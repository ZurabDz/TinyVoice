import math

import jax
import jax.numpy as jnp
from flax import nnx

from .mel import AudioToMelSpectrogram


def _rope_table(head_dim: int, max_len: int, dtype):
    inv = 1.0 / (10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _apply_rope(x, cos, sin):  # x: (B, T, H, D)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_h = cos[None, :, None, :half]
    sin_h = sin[None, :, None, :half]
    return jnp.concatenate(
        [x1 * cos_h - x2 * sin_h, x1 * sin_h + x2 * cos_h], axis=-1
    )


class DropPath(nnx.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""

    def __init__(self, drop_prob: float, rngs: nnx.Rngs):
        self.drop_prob = drop_prob
        self.rngs = rngs

    def __call__(self, x, training: bool):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jax.random.uniform(self.rngs.dropout(), shape, dtype=x.dtype)
        binary_tensor = jnp.floor(random_tensor)
        return x / keep_prob * binary_tensor


class ConvSubsampler(nnx.Module):
    """Two stride-2 2D convs giving a 4× time downsample."""

    def __init__(self, d_model: int, *, rngs: nnx.Rngs, dtype):
        self.conv1 = nnx.Conv(
            1, d_model // 4, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs, dtype=dtype
        )
        self.conv2 = nnx.Conv(
            d_model // 4, d_model, (3, 3), strides=(2, 2), padding="VALID", rngs=rngs, dtype=dtype
        )

    @staticmethod
    def output_length(t):
        t = (t - 3) // 2 + 1
        t = (t - 3) // 2 + 1
        return jnp.maximum(t, 0)

    def __call__(self, x):  # (B, T, F)
        x = nnx.silu(self.conv1(x[..., None]))
        x = nnx.silu(self.conv2(x))
        B, T, F, C = x.shape
        return x.reshape(B, T, F * C)


class SwiGLUFFN(nnx.Module):
    """Pre-norm SwiGLU feed-forward — half-residual macaroon contribution."""

    def __init__(self, d_model: int, expansion: int, dropout: float, *, rngs: nnx.Rngs, dtype):
        hidden = d_model * expansion // 2
        self.norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.gate = nnx.Linear(d_model, hidden, use_bias=False, rngs=rngs, dtype=dtype)
        self.up = nnx.Linear(d_model, hidden, use_bias=False, rngs=rngs, dtype=dtype)
        self.down = nnx.Linear(hidden, d_model, use_bias=False, rngs=rngs, dtype=dtype)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, training: bool):
        h = self.norm(x)
        h = nnx.silu(self.gate(h)) * self.up(h)
        return self.drop(self.down(h), deterministic=not training)


class FlashAttention(nnx.Module):
    """Pre-norm MHSA over fused QKV with RoPE and cuDNN flash attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, *, rngs: nnx.Rngs, dtype):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.qkv = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs, dtype=dtype)
        self.out = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, cos, sin, lengths, training: bool):
        B, T, _ = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q = _apply_rope(qkv[:, :, 0], cos, sin)
        k = _apply_rope(qkv[:, :, 1], cos, sin)
        v = qkv[:, :, 2]
        out = jax.nn.dot_product_attention(
            q, k, v,
            query_seq_lengths=lengths,
            key_value_seq_lengths=lengths,
            implementation="cudnn",
        )
        out = out.reshape(B, T, -1)
        return self.drop(self.out(out), deterministic=not training)


class ConvModule(nnx.Module):
    """Pre-norm pointwise→GLU→depthwise→activation→pointwise convolution module."""

    def __init__(self, d_model: int, kernel: int, dropout: float, *, rngs: nnx.Rngs, dtype):
        self.norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.pw1 = nnx.Linear(d_model, 2 * d_model, use_bias=False, rngs=rngs, dtype=dtype)
        self.dw = nnx.Conv(
            d_model,
            d_model,
            (kernel,),
            feature_group_count=d_model,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
        )
        self.act_norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.pw2 = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, mask1d, training: bool):
        h = self.norm(x)
        h = nnx.glu(self.pw1(h), axis=-1)
        h = h * mask1d
        h = self.dw(h)
        h = nnx.silu(self.act_norm(h))
        h = self.pw2(h)
        return self.drop(h, deterministic=not training)


class FastConformerBlock(nnx.Module):
    """Macaroon: ½·FFN → MHSA → Conv → ½·FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expansion: int,
        kernel: int,
        dropout: float,
        stochastic_depth_prob: float,
        *,
        rngs: nnx.Rngs,
        dtype,
    ):
        self.ff1 = SwiGLUFFN(d_model, expansion, dropout, rngs=rngs, dtype=dtype)
        self.attn = FlashAttention(d_model, num_heads, dropout, rngs=rngs, dtype=dtype)
        self.conv = ConvModule(d_model, kernel, dropout, rngs=rngs, dtype=dtype)
        self.ff2 = SwiGLUFFN(d_model, expansion, dropout, rngs=rngs, dtype=dtype)
        self.drop_path = DropPath(stochastic_depth_prob, rngs=rngs)

    def __call__(self, x, cos, sin, lengths, mask1d, training: bool):
        x = x + self.drop_path(0.5 * self.ff1(x, training), training)
        x = x + self.drop_path(self.attn(x, cos, sin, lengths, training), training)
        x = x + self.drop_path(self.conv(x, mask1d, training), training)
        x = x + self.drop_path(0.5 * self.ff2(x, training), training)
        return x


class FastConformerEncoder(nnx.Module):
    """FastConformer encoder: log-mel → 4× subsample → N scanned blocks → CTC head."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        expansion: int,
        kernel: int,
        dropout: float,
        stochastic_depth_prob: float,
        interctc_layer: int,
        n_mels: int,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.frontend = AudioToMelSpectrogram(
            sample_rate=sample_rate,
            n_window_size=win_length,
            n_window_stride=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            rngs=rngs,
            spec_augment=True,
        )
        self.subsampler = ConvSubsampler(d_model, rngs=rngs, dtype=dtype)
        freq_dim = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.proj = nnx.Linear(d_model * freq_dim, d_model, rngs=rngs, dtype=dtype)

        # Stochastic depth linear decay: 0 at layer 0, max at last layer
        def get_drop_prob(layer_idx):
            return stochastic_depth_prob * (layer_idx + 1) / num_layers

        self.blocks = nnx.List([])
        for i in range(num_layers):
            self.blocks.append(
                FastConformerBlock(
                    d_model,
                    num_heads,
                    expansion,
                    kernel,
                    dropout,
                    get_drop_prob(i),
                    rngs=rngs,
                    dtype=dtype,
                )
            )

        self.final_norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs, dtype=dtype)

        # InterCTC head
        self.num_layers = num_layers
        self.interctc_layer = interctc_layer
        if 0 < interctc_layer < num_layers:
            self.interctc_head = nnx.Linear(d_model, vocab_size, rngs=rngs, dtype=dtype)
        else:
            self.interctc_head = None

        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.dtype = dtype

    def __call__(self, audio, audio_lengths, training: bool = True):
        mel, mel_lengths = self.frontend(audio, audio_lengths, training=training)
        x = jnp.transpose(mel, (0, 2, 1))  # (B, T_mel, F_mel)
        seq_len = self.subsampler.output_length(mel_lengths)
        x = self.subsampler(x)
        x = self.proj(x) * math.sqrt(self.d_model)

        T = x.shape[1]
        cos, sin = _rope_table(self.head_dim, T, self.dtype)
        mask1d = (jnp.arange(T)[None, :] < seq_len[:, None])[:, :, None].astype(x.dtype)

        inter_logits = None
        
        # Manually iterate to capture intermediate state and avoid nnx.scan issues
        for i, block in enumerate(self.blocks):
            # Wrap in remat for memory efficiency
            def block_fn(x_in, b=block):
                return b(x_in, cos, sin, seq_len, mask1d, training)
            
            x = nnx.remat(block_fn)(x)
            
            if self.interctc_head is not None and i == self.interctc_layer - 1:
                inter_logits = self.interctc_head(self.final_norm(x))

        x_enc = self.final_norm(x)
        logits = self.head(x_enc)

        return {
            "logits": logits,
            "inter_logits": inter_logits,
            "encoder_states": x_enc,
            "output_lengths": seq_len,
        }


class TransformerDecoderLayer(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, expansion: int, dropout: float, *, rngs: nnx.Rngs, dtype):
        self.self_attn_norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.self_attn = nnx.MultiHeadAttention(num_heads, d_model, dropout_rate=dropout, decode=False, rngs=rngs, dtype=dtype)
        
        self.cross_attn_norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.cross_attn = nnx.MultiHeadAttention(num_heads, d_model, dropout_rate=dropout, decode=False, rngs=rngs, dtype=dtype)
        
        self.ffn = SwiGLUFFN(d_model, expansion, dropout, rngs=rngs, dtype=dtype)

    def __call__(self, x, enc_states, self_mask, cross_mask, training: bool):
        # Self-attention
        h = self.self_attn_norm(x)
        x = x + self.self_attn(h, mask=self_mask, deterministic=not training)
        
        # Cross-attention
        h = self.cross_attn_norm(x)
        x = x + self.cross_attn(h, enc_states, mask=cross_mask, deterministic=not training)
        
        # FFN
        x = x + self.ffn(x, training)
        return x


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        expansion: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
        dtype,
    ):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs, dtype=dtype)
        self.pos_embed = nnx.Embed(2048, d_model, rngs=rngs, dtype=dtype) # Max seq len
        
        self.layers = nnx.List([])
        for _ in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(d_model, num_heads, expansion, dropout, rngs=rngs, dtype=dtype)
            )
            
        self.final_norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs, dtype=dtype)
        self.d_model = d_model

    def __call__(self, labels, enc_states, enc_lengths, training: bool):
        B, T_dec = labels.shape
        T_enc = enc_states.shape[1]
        
        # Embeddings
        x = self.embed(labels) * math.sqrt(self.d_model)
        x = x + self.pos_embed(jnp.arange(T_dec)[None, :])
        
        # Masks
        causal_mask = jnp.tril(jnp.ones((T_dec, T_dec)))
        self_mask = causal_mask[None, None, :, :]
        
        cross_mask = (jnp.arange(T_enc)[None, :] < enc_lengths[:, None])[:, None, None, :]
        
        for layer in self.layers:
            # Wrap in remat for memory efficiency
            def layer_fn(x_in, l=layer):
                return l(x_in, enc_states, self_mask, cross_mask, training)
            x = nnx.remat(layer_fn)(x)
            
        x = self.final_norm(x)
        return self.head(x)


class FastConformer(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        expansion: int,
        kernel: int,
        dropout: float,
        stochastic_depth_prob: float,
        interctc_layer: int,
        n_mels: int,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        dtype,
        rngs: nnx.Rngs,
    ):
        self.encoder = FastConformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            expansion=expansion,
            kernel=kernel,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            interctc_layer=interctc_layer,
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            dtype=dtype,
            rngs=rngs,
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, audio, audio_lengths, labels=None, training: bool = True):
        enc_out = self.encoder(audio, audio_lengths, training=training)
        
        if labels is not None:
            dec_logits = self.decoder(labels, enc_out["encoder_states"], enc_out["output_lengths"], training=training)
        else:
            dec_logits = None
            
        return {
            **enc_out,
            "dec_logits": dec_logits,
        }

