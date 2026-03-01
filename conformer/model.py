from flax import nnx
import jax
import jax.numpy as jnp
from .mel import AudioToMelSpectrogram
import numpy as np


class RelativePositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, max_len: int = 500, dtype=jnp.float32):
        self.d_model = d_model
        self.dtype = dtype
        pe = np.zeros((2 * max_len, d_model))
        position = np.arange(-max_len, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe, dtype=dtype)

    def __call__(self, seq_len: int):
        # returns positional embeddings for relative distances -(seq_len-1) to (seq_len-1)
        max_len = (self.pe.shape[0]) // 2

        # Handle edge case: if seq_len is 0 or negative, return empty
        if seq_len <= 0:
            return jnp.zeros((0, self.d_model), dtype=self.dtype)

        # Ensure we don't go out of bounds
        start_idx = max(0, max_len - (seq_len - 1))
        end_idx = min(self.pe.shape[0], max_len + seq_len)
        pe_slice = self.pe[start_idx:end_idx]

        # If we had to clip, pad with zeros
        if start_idx > max_len - (seq_len - 1):
            pad_before = jnp.zeros(
                (start_idx - (max_len - (seq_len - 1)), self.d_model), dtype=self.dtype
            )
            pe_slice = jnp.concatenate([pad_before, pe_slice], axis=0)
        if end_idx < max_len + seq_len:
            pad_after = jnp.zeros(
                (max_len + seq_len - end_idx, self.d_model), dtype=self.dtype
            )
            pe_slice = jnp.concatenate([pe_slice, pad_after], axis=0)

        # Final check: ensure output has correct length
        expected_len = 2 * seq_len - 1
        if pe_slice.shape[0] != expected_len:
            # This shouldn't happen, but handle it gracefully
            if pe_slice.shape[0] < expected_len:
                pad_needed = expected_len - pe_slice.shape[0]
                pe_slice = jnp.concatenate(
                    [pe_slice, jnp.zeros((pad_needed, self.d_model), dtype=self.dtype)],
                    axis=0,
                )
            else:
                pe_slice = pe_slice[:expected_len]

        return pe_slice


class ConformerMultiHeadAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_features: int,
        out_features: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = qkv_features // num_heads
        self.dtype = dtype

        self.q_proj = nnx.Linear(d_model, qkv_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(
            d_model, qkv_features, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.v_proj = nnx.Linear(
            d_model, qkv_features, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.pos_proj = nnx.Linear(
            d_model, qkv_features, use_bias=False, rngs=rngs, dtype=dtype
        )

        self.out_proj = nnx.Linear(qkv_features, out_features, rngs=rngs, dtype=dtype)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # Learned biases for relative positioning
        self.u = nnx.Param(jnp.zeros((num_heads, self.head_dim), dtype=dtype))
        self.v = nnx.Param(jnp.zeros((num_heads, self.head_dim), dtype=dtype))

    def __call__(self, x, pos_emb, mask=None, training=True):
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).reshape(B, T, H, d)
        k = self.k_proj(x).reshape(B, T, H, d)
        v = self.v_proj(x).reshape(B, T, H, d)

        # Reverse pos_emb so the skew operation produces R[i-j] (not R[j-i]).
        # pos_emb is ordered from distance -(T-1) to +(T-1); the standard
        # relative-shift / skew trick expects the reversed order (+(T-1) to -(T-1)).
        pos_emb_rev = jnp.flip(pos_emb, axis=0)
        p = self.pos_proj(pos_emb_rev).reshape(-1, H, d)  # (2T-1, H, d)

        # Term (a): content-content
        content_q = (q + self.u).transpose(0, 2, 1, 3)  # (B, H, T, d)
        content_scores = jnp.matmul(content_q, k.transpose(0, 2, 3, 1))

        # Term (b) & (d): content-position and bias-position
        pos_q = (q + self.v).transpose(0, 2, 1, 3)  # (B, H, T, d)
        pos_scores = jnp.matmul(pos_q, p.transpose(1, 2, 0))  # (B, H, T, 2T-1)

        # Relative shift trick to get (B, H, T, T)
        pos_scores = self._relative_shift(pos_scores)

        scores = (content_scores + pos_scores) / jnp.sqrt(d)

        if mask is not None:
            # mask is (B, 1, 1, T), expand to (B, 1, T, T) for attention masking
            # Mask out positions where either query or key is padding
            mask_query = mask[:, 0, 0, :, None]  # (B, T, 1)
            mask_key = mask[:, 0, 0, None, :]  # (B, 1, T)
            mask_expanded = (mask_query & mask_key)[:, None, :, :]  # (B, 1, T, T)
            scores = jnp.where(mask_expanded, scores, -1e9)

        probs = nnx.softmax(scores, axis=-1)
        probs = self.dropout(probs, deterministic=not training)

        attn_out = jnp.matmul(probs, v.transpose(0, 2, 1, 3))  # (B, H, T, d)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        if mask is not None:
            q_mask = mask[:, 0, 0, :, None].astype(attn_out.dtype)
            attn_out = attn_out * q_mask

        return self.out_proj(attn_out)

    def _relative_shift(self, x):
        # x shape: (B, H, T, 2T-1)
        B, H, T, W = x.shape
        # Pad one column of zeros on the left: (B, H, T, 2T)
        zero_pad = jnp.zeros((B, H, T, 1), dtype=x.dtype)
        x = jnp.concatenate([zero_pad, x], axis=-1)
        # Reshape to (B, H, 2T, T)
        x = x.reshape(B, H, 2 * T, T)
        # Slice and reshape to (B, H, T, 2T-1) then take first T
        x = x[:, :, 1:, :]  # (B, H, 2T-1, T)
        x = x.reshape(B, H, T, 2 * T - 1)
        return x[:, :, :, :T]


class Conv2dSubSampler(nnx.Module):
    def __init__(self, d_model, rngs: nnx.Rngs, dtype=jnp.float32):
        self.module = nnx.Sequential(
            nnx.Conv(
                in_features=1,
                out_features=d_model,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                rngs=rngs,
                dtype=dtype,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=d_model,
                out_features=d_model,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                rngs=rngs,
                dtype=dtype,
            ),
            nnx.relu,
        )

    def get_length(self, seq_len):
        # Two layers of Conv2d with kernel_size=3, stride=2, padding="VALID"
        # Formula: L_out = floor((L_in - K) / S) + 1
        seq_len = (seq_len - 3) // 2 + 1
        seq_len = (seq_len - 3) // 2 + 1
        return jnp.maximum(seq_len, 0)

    def __call__(self, x):
        # B, T, D, 1(C)
        output = self.module(x)
        batch_size, subsampled_time, subsampled_freq, d_model = output.shape
        return output.reshape(batch_size, subsampled_time, subsampled_freq * d_model)


class ConformerFeedForwardBlock(nnx.Module):
    def __init__(
        self, d_model, expansion_factor, dropout, rngs: nnx.Rngs, dtype=jnp.float32
    ):
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        self.lin1 = nnx.Linear(
            d_model, d_model * expansion_factor, rngs=rngs, dtype=dtype
        )
        self.drop1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.lin2 = nnx.Linear(
            d_model * expansion_factor, d_model, rngs=rngs, dtype=dtype
        )
        self.drop2 = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, training=True):
        x = self.ln(x)
        x = self.lin1(x)
        x = nnx.silu(x)
        x = self.drop1(x, deterministic=not training)
        x = self.lin2(x)
        x = self.drop2(x, deterministic=not training)
        return x


class ConformerConvBlock(nnx.Module):
    def __init__(self, d_model, dropout, rngs: nnx.Rngs, dtype=jnp.float32):
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=d_model,
            out_features=d_model * 2,
            kernel_size=1,
            rngs=rngs,
            dtype=dtype,
        )
        self.conv2 = nnx.Conv(
            in_features=d_model,
            out_features=d_model,
            kernel_size=31,
            feature_group_count=d_model,
            rngs=rngs,
            dtype=dtype,
        )
        self.batch_norm = nnx.BatchNorm(d_model, rngs=rngs)
        self.conv3 = nnx.Conv(
            in_features=d_model,
            out_features=d_model,
            kernel_size=1,
            rngs=rngs,
            dtype=dtype,
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, mask=None, training=True):
        x = self.layer_norm(x)

        x = self.conv1(x)
        x = nnx.glu(x)

        if mask is not None:
            # mask is (B, 1, 1, T), x is (B, T, D)
            m = mask[:, 0, 0, :, None].astype(x.dtype)
            x = x * m
            x = self.conv2(x)
            x = x * m
            x = self.batch_norm(x, use_running_average=not training)
            x = x * m
        else:
            x = self.conv2(x)
            x = self.batch_norm(x, use_running_average=not training)

        x = nnx.silu(x)
        x = self.conv3(x)
        x = self.dropout(x, deterministic=not training)
        return x


class ConformerBlock(nnx.Module):
    def __init__(
        self,
        d_model=144,
        feed_forward_residual_factor=0.5,
        feed_forward_expansion_factor=4,
        num_head=4,
        dropout=0.1,
        rngs: nnx.Rngs | None = None,
        dtype=jnp.float32,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.residual_factor = feed_forward_residual_factor
        self.ff1 = ConformerFeedForwardBlock(
            d_model, feed_forward_expansion_factor, dropout, rngs=rngs, dtype=dtype
        )
        self.ln_before_attention = nnx.LayerNorm(d_model, rngs=rngs)
        self.attention = ConformerMultiHeadAttention(
            num_head,
            d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=dropout,
            rngs=rngs,
            dtype=dtype,
        )
        self.attn_dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.conv_block = ConformerConvBlock(
            d_model, dropout=dropout, rngs=rngs, dtype=dtype
        )
        self.ff2 = ConformerFeedForwardBlock(
            d_model, feed_forward_expansion_factor, dropout, rngs=rngs, dtype=dtype
        )
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, pos_emb, mask=None, training=True):
        x = x + (self.residual_factor * self.ff1(x, training=training))
        attn_out = self.attention(
            self.ln_before_attention(x),
            pos_emb,
            mask=mask,
            training=training,
        )
        x = x + self.attn_dropout(attn_out, deterministic=not training)
        x = x + self.conv_block(x, mask=mask, training=training)
        x = x + (self.residual_factor * self.ff2(x, training=training))
        x = self.layer_norm(x)

        if mask is not None:
            # Final mask to kill any residual noise in padding
            m = mask[:, 0, 0, :, None].astype(x.dtype)
            x = x * m

        return x


class ConformerEncoder(nnx.Module):
    def __init__(
        self,
        token_count,
        d_input=80,
        d_model=144,
        num_layers=16,
        feed_forward_residual_factor=0.5,
        feed_forward_expansion_factor=4,
        num_head=4,
        dropout=0.1,
        rngs: nnx.Rngs | None = None,
        dtype=jnp.float32,
        sample_rate=16000,
        n_fft=512,
        n_window_size=400,
        n_window_stride=160,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mel_spectogram = AudioToMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            n_mels=d_input,
            rng=rngs,
        )
        self.mel_spectogram.normalize = True
        self.mel_spectogram.spec_augment = True
        self.conv_subsampler = Conv2dSubSampler(d_model=d_model, rngs=rngs, dtype=dtype)
        # Calculate frequency dimension after two conv layers: (d_input - 3) // 2 + 1, then ((freq_dim - 3) // 2 + 1)
        freq_dim_after_conv1 = (d_input - 3) // 2 + 1
        freq_dim_after_conv2 = (freq_dim_after_conv1 - 3) // 2 + 1
        self.linear_proj = nnx.Linear(
            d_model * freq_dim_after_conv2, d_model, rngs=rngs, dtype=dtype
        )
        self.rel_pos_encoding = RelativePositionalEncoding(d_model=d_model, dtype=dtype)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

        self.layers = nnx.List(
            [
                ConformerBlock(
                    d_model=d_model,
                    feed_forward_residual_factor=feed_forward_residual_factor,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    num_head=num_head,
                    dropout=dropout,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nnx.Linear(d_model, token_count, rngs=rngs, dtype=dtype)
        self.d_model = d_model

    def compute_mask(self, lengths, max_length):
        mask = jnp.arange(max_length)[None, :] < lengths[:, None]
        return mask[:, None, None, :]

    def __call__(self, x, mask=None, training=True, inputs_lengths=None):
        x, seq_len = self.mel_spectogram(x, lengths=inputs_lengths, training=training)
        x = jnp.transpose(x, (0, 2, 1))

        # Calculate subsampled length before convolution
        seq_len = self.conv_subsampler.get_length(seq_len)

        x = self.conv_subsampler(x[:, :, :, None])
        x = self.linear_proj(x)
        x = x * jnp.sqrt(self.d_model)  # Standard scaling for Conformer/Transformer

        # Generate relative positional embeddings for the current sequence length
        pos_emb = self.rel_pos_encoding(x.shape[1])
        x = self.dropout(x, deterministic=not training)

        if mask is None and inputs_lengths is not None:
            mask = self.compute_mask(seq_len, x.shape[1])

        for layer in self.layers:
            x = layer(x, pos_emb, mask, training=training)

        return self.decoder(x), seq_len

    def initialize_weights(self, rng_key: jax.Array):
        """Initialize weights following NeMo/Conformer best practices."""

        def init_fn(module):
            if isinstance(module, nnx.Linear):
                d_in = module.in_features
                stddev = 1.0 / jnp.sqrt(d_in)
                module.kernel.value = jax.random.uniform(
                    rng_key, module.kernel.shape, minval=-stddev, maxval=stddev
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        rng_key, module.bias.shape, minval=-stddev, maxval=stddev
                    )
            elif isinstance(module, nnx.Conv):
                # Simple uniform init for conv as well
                k_size = np.prod(module.kernel_size)
                d_in = module.in_features
                stddev = 1.0 / jnp.sqrt(k_size * d_in)
                module.kernel.value = jax.random.uniform(
                    rng_key, module.kernel.shape, minval=-stddev, maxval=stddev
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        rng_key, module.bias.shape, minval=-stddev, maxval=stddev
                    )

        # Apply to all submodules
        for _, module in self.iter_modules():
            init_fn(module)


# ====================== ZIPFORMER INNOVATIONS ======================
class BiasNorm(nnx.Module):
    """Exact BiasNorm from the Zipformer paper (retains length info)."""

    def __init__(self, dim: int, rngs: nnx.Rngs, dtype=jnp.float32):
        self.dim = dim
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))
        self.log_scale = nnx.Param(jnp.zeros((), dtype=dtype))  # gamma

    def __call__(self, x):
        x_centered = x - self.bias.value
        rms = jnp.sqrt(jnp.mean(x_centered**2, axis=-1, keepdims=True) + 1e-6)
        scale = jnp.exp(self.log_scale.value)
        return (x_centered / rms) * scale


class SwooshL(nnx.Module):  # used in FFN (normally-off)
    def __call__(self, x):
        return jnp.log1p(jnp.exp(x - 4.0)) - 0.08 * x - 0.035


class SwooshR(nnx.Module):  # used after conv
    def __call__(self, x):
        return jnp.log1p(jnp.exp(x - 1.0)) - 0.08 * x - 0.313261687


class Bypass(nnx.Module):
    """Zipformer bypass: learnable mixing of residual."""

    def __init__(self, dim: int, rngs: nnx.Rngs, init_value: float = 0.95):
        self.alpha = nnx.Param(jnp.full((dim,), init_value))

    def __call__(self, x, y):
        c = nnx.sigmoid(self.alpha.value)  # stays between 0-1
        return (1 - c) * x + c * y


class RotaryEmbedding(nnx.Module):
    """RoPE – the newest positional encoding (replaces your old relative one)."""

    def __init__(self, dim: int, max_len: int = 2048, dtype=jnp.float32, rngs=None):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
        self.inv_freq = jnp.array(inv_freq, dtype=dtype)  # not a Param (static)

    def __call__(self, x):  # x: (B, T, H, D)
        seq_len = x.shape[1]
        t = jnp.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)[None, :, None, :]  # (1, T, 1, D)
        sin = jnp.sin(emb)[None, :, None, :]
        return cos, sin


def apply_rotary_emb(q, k, cos, sin):
    # q/k shape: (B, T, H, D), cos/sin shape: (1, T, 1, D)
    # Standard RoPE: split into two halves along last dim and rotate
    d = q.shape[-1] // 2
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    cos_half = cos[..., :d]  # (1, T, 1, D/2)
    sin_half = sin[..., :d]  # (1, T, 1, D/2)
    q_out = jnp.concatenate(
        [q1 * cos_half - q2 * sin_half, q1 * sin_half + q2 * cos_half], axis=-1
    )
    k_out = jnp.concatenate(
        [k1 * cos_half - k2 * sin_half, k1 * sin_half + k2 * cos_half], axis=-1
    )
    return q_out, k_out


# ====================== UPDATED BLOCKS ======================
class ZipformerFeedForwardBlock(nnx.Module):  # SwiGLU + Bypass
    def __init__(
        self, d_model, expansion_factor, dropout, rngs: nnx.Rngs, dtype=jnp.float32
    ):
        self.norm = BiasNorm(d_model, rngs=rngs, dtype=dtype)
        self.lin1 = nnx.Linear(
            d_model, d_model * expansion_factor * 2, rngs=rngs, dtype=dtype
        )
        self.drop = nnx.Dropout(dropout, rngs=rngs)
        self.lin2 = nnx.Linear(
            d_model * expansion_factor, d_model, rngs=rngs, dtype=dtype
        )
        self.bypass = Bypass(d_model, rngs=rngs)

    def __call__(self, x, training=True):
        residual = x
        x = self.norm(x)
        x = self.lin1(x)
        a, b = jnp.split(x, 2, axis=-1)
        x = a * nnx.silu(b)  # SwiGLU
        x = self.drop(x, deterministic=not training)
        x = self.lin2(x)
        x = self.drop(x, deterministic=not training)
        return self.bypass(residual, x)


class ZipformerMultiHeadAttention(nnx.Module):  # upgraded with RoPE
    def __init__(
        self, num_heads, d_model, dropout_rate, rngs: nnx.Rngs, dtype=jnp.float32
    ):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dtype = dtype

        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(
            d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.v_proj = nnx.Linear(
            d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.rope = RotaryEmbedding(self.head_dim, rngs=rngs)

    def __call__(self, x, mask=None, training=True):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, d)
        k = self.k_proj(x).reshape(B, T, H, d)
        v = self.v_proj(x).reshape(B, T, H, d)

        cos, sin = self.rope(q)  # RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(d)

        if mask is not None:
            # mask is already (B, 1, 1, T), broadcasts with scores (B, H, T, T)
            scores = jnp.where(mask, scores, -1e9)

        attn = nnx.softmax(scores, axis=-1)
        attn = self.dropout(attn, deterministic=not training)

        out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, D)
        return self.out_proj(out)


class ZipformerConvBlock(nnx.Module):  # kept almost same, but with SwooshR + BiasNorm
    def __init__(self, d_model, dropout, rngs: nnx.Rngs, dtype=jnp.float32):
        self.norm = BiasNorm(d_model, rngs=rngs, dtype=dtype)
        self.conv1 = nnx.Conv(d_model, d_model * 2, (1,), rngs=rngs, dtype=dtype)
        self.conv2 = nnx.Conv(
            d_model, d_model, (31,), feature_group_count=d_model, rngs=rngs, dtype=dtype
        )
        self.conv3 = nnx.Conv(d_model, d_model, (1,), rngs=rngs, dtype=dtype)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.bypass = Bypass(d_model, rngs=rngs)

    def __call__(self, x, mask=None, training=True):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = nnx.glu(x)

        if mask is not None:
            m = mask[:, 0, 0, :, None].astype(x.dtype)
            x = x * m
        x = self.conv2(x)
        x = SwooshR()(x)
        if mask is not None:
            x = x * m
        x = self.conv3(x)
        x = self.dropout(x, deterministic=not training)
        return self.bypass(residual, x)


class ZipformerBlock(nnx.Module):
    def __init__(
        self,
        d_model=256,
        feed_forward_expansion_factor=4,
        num_head=4,
        dropout=0.1,
        rngs=None,
        dtype=jnp.float32,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.ff1 = ZipformerFeedForwardBlock(
            d_model, feed_forward_expansion_factor, dropout, rngs, dtype
        )
        self.attn = ZipformerMultiHeadAttention(num_head, d_model, dropout, rngs, dtype)
        self.conv = ZipformerConvBlock(d_model, dropout, rngs, dtype)
        self.ff2 = ZipformerFeedForwardBlock(
            d_model, feed_forward_expansion_factor, dropout, rngs, dtype
        )

    def __call__(self, x, mask=None, training=True):
        x = self.ff1(x, training=training)
        x = x + self.attn(x, mask=mask, training=training)
        x = x + self.conv(x, mask=mask, training=training)
        x = self.ff2(x, training=training)
        return x


# ====================== MAIN ENCODER ======================
class ZipformerEncoder(nnx.Module):
    def __init__(
        self,
        token_count,
        d_input=128,
        d_model=256,
        num_layers=4,
        feed_forward_expansion_factor=4,
        num_head=4,
        dropout=0.1,
        rngs=None,
        dtype=jnp.float32,
        **featurizer_kwargs,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mel_spectogram = AudioToMelSpectrogram(
            n_mels=d_input, rng=rngs, **featurizer_kwargs
        )
        self.mel_spectogram.normalize = True
        self.mel_spectogram.spec_augment = True

        # Same subsampler as before (you can upgrade later)
        self.conv_subsampler = Conv2dSubSampler(d_model, rngs, dtype)
        # Calculate frequency dimension after two conv layers: (d_input - 3) // 2 + 1, then ((freq_dim - 3) // 2 + 1)
        freq_dim_after_conv1 = (d_input - 3) // 2 + 1
        freq_dim_after_conv2 = (freq_dim_after_conv1 - 3) // 2 + 1
        self.linear_proj = nnx.Linear(
            d_model * freq_dim_after_conv2, d_model, rngs=rngs, dtype=dtype
        )

        self.layers = nnx.List(
            [
                ZipformerBlock(
                    d_model,
                    feed_forward_expansion_factor,
                    num_head,
                    dropout,
                    rngs,
                    dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nnx.Linear(d_model, token_count, rngs=rngs, dtype=dtype)
        self.d_model = d_model

    def compute_mask(self, lengths, max_length):
        return (jnp.arange(max_length)[None, :] < lengths[:, None])[:, None, None, :]

    def __call__(self, x, mask=None, training=True, inputs_lengths=None):
        x, seq_len = self.mel_spectogram(x, lengths=inputs_lengths, training=training)
        x = jnp.transpose(x, (0, 2, 1))  # B, T, F

        seq_len = self.conv_subsampler.get_length(seq_len)
        x = self.conv_subsampler(x[:, :, :, None])
        x = self.linear_proj(x)
        x = x * jnp.sqrt(self.d_model)

        if mask is None and inputs_lengths is not None:
            mask = self.compute_mask(seq_len, x.shape[1])

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return self.decoder(x), seq_len

    def initialize_weights(self, rng_key: jax.Array):
        """Initialize weights following NeMo best practices."""

        def init_fn(module):
            if isinstance(module, nnx.Linear):
                d_in = module.in_features
                stddev = 1.0 / jnp.sqrt(d_in)
                module.kernel.value = jax.random.uniform(
                    rng_key, module.kernel.shape, minval=-stddev, maxval=stddev
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        rng_key, module.bias.shape, minval=-stddev, maxval=stddev
                    )
            elif isinstance(module, nnx.Conv):
                k_size = np.prod(module.kernel_size)
                d_in = module.in_features
                stddev = 1.0 / jnp.sqrt(k_size * d_in)
                module.kernel.value = jax.random.uniform(
                    rng_key, module.kernel.shape, minval=-stddev, maxval=stddev
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        rng_key, module.bias.shape, minval=-stddev, maxval=stddev
                    )

        # Apply to all submodules
        for _, module in self.iter_modules():
            init_fn(module)
