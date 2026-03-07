from flax import nnx
import jax
import jax.numpy as jnp
from .mel import AudioToMelSpectrogram
import numpy as np


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

    def __init__(self, dim: int, rngs: nnx.Rngs, init_value: float = 0.0):
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

        self.norm = BiasNorm(d_model, rngs=rngs, dtype=dtype)
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(
            d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.v_proj = nnx.Linear(
            d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.bypass = Bypass(d_model, rngs=rngs)

        self.rope = RotaryEmbedding(self.head_dim, rngs=rngs)

    def __call__(self, x, mask=None, training=True):
        residual = x
        x = self.norm(x)
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
        return self.bypass(residual, self.out_proj(out))


class ZipformerConvBlock(nnx.Module):  # kept almost same, but with SwooshR + BiasNorm
    def __init__(self, d_model, dropout, rngs: nnx.Rngs, dtype=jnp.float32):
        self.norm = BiasNorm(d_model, rngs=rngs, dtype=dtype)
        self.conv1 = nnx.Conv(d_model, d_model * 2, (1,), rngs=rngs, dtype=dtype)
        self.conv2 = nnx.Conv(
            d_model, d_model, (31,), feature_group_count=d_model, rngs=rngs, dtype=dtype
        )
        self.norm2 = BiasNorm(d_model, rngs=rngs, dtype=dtype)
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
        x = self.norm2(x)
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
        x = self.attn(x, mask=mask, training=training)
        x = self.conv(x, mask=mask, training=training)
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

    def encode_from_mel(self, mel, seq_len, training=False):
        """Run encoder from pre-computed mel spectrogram.

        Designed for ONNX export: avoids dynamic-shape masking that cannot be
        traced by jax2onnx.  For batched inference the caller should zero-pad
        the mel spectrogram so that no mask is needed, or use batch_size=1.

        Args:
            mel: Pre-computed mel spectrogram of shape (B, n_mels, T) — the
                 output format of AudioToMelSpectrogram.
            seq_len: Integer array of shape (B,) with valid frame counts.
            training: Whether to run in training mode (enables dropout).

        Returns:
            (logits, output_seq_len) — same as __call__.
        """
        x = jnp.transpose(mel, (0, 2, 1))  # B, T, F

        seq_len = self.conv_subsampler.get_length(seq_len)
        x = self.conv_subsampler(x[:, :, :, None])
        x = self.linear_proj(x)
        x = x * jnp.sqrt(self.d_model)

        # Skip masking for ONNX export — jnp.arange(x.shape[1]) cannot be
        # traced when x.shape[1] is a symbolic dynamic dimension.
        for layer in self.layers:
            x = layer(x, mask=None, training=training)

        return self.decoder(x), seq_len

    def initialize_weights(self, rng_key: jax.Array):
        """Initialize weights following NeMo best practices."""

        def init_fn(module):
            nonlocal rng_key
            if isinstance(module, nnx.Linear):
                rng_key, k1, k2 = jax.random.split(rng_key, 3)
                d_in = module.in_features
                limit = jnp.sqrt(3.0 / d_in)
                module.kernel.value = jax.random.uniform(
                    k1, module.kernel.shape, minval=-limit, maxval=limit
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        k2, module.bias.shape, minval=-limit, maxval=limit
                    )
            elif isinstance(module, nnx.Conv):
                rng_key, k1, k2 = jax.random.split(rng_key, 3)
                k_size = np.prod(module.kernel_size)
                d_in = module.in_features
                fan_in = k_size * d_in
                limit = jnp.sqrt(3.0 / fan_in)
                module.kernel.value = jax.random.uniform(
                    k1, module.kernel.shape, minval=-limit, maxval=limit
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        k2, module.bias.shape, minval=-limit, maxval=limit
                    )

        for _, module in self.iter_modules():
            init_fn(module)
