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
        self.down = nnx.Linear(hidden, d_model, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=jax.nn.initializers.zeros)
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
        self.out = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=jax.nn.initializers.zeros)
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
        self.pw2 = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=jax.nn.initializers.zeros)
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
        *,
        rngs: nnx.Rngs,
        dtype,
    ):
        self.ff1 = SwiGLUFFN(d_model, expansion, dropout, rngs=rngs, dtype=dtype)
        self.attn = FlashAttention(d_model, num_heads, dropout, rngs=rngs, dtype=dtype)
        self.conv = ConvModule(d_model, kernel, dropout, rngs=rngs, dtype=dtype)
        self.ff2 = SwiGLUFFN(d_model, expansion, dropout, rngs=rngs, dtype=dtype)

    def __call__(self, x, cos, sin, lengths, mask1d, training: bool):
        x = x + 0.5 * self.ff1(x, training)
        x = x + self.attn(x, cos, sin, lengths, training)
        x = x + self.conv(x, mask1d, training)
        x = x + 0.5 * self.ff2(x, training)
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

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=0, out_axes=0)
        def make_block(rngs):
            return FastConformerBlock(
                d_model, num_heads, expansion, kernel, dropout, rngs=rngs, dtype=dtype
            )

        self.blocks = make_block(rngs)
        self.final_norm = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype, param_dtype=jnp.float32)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs, dtype=dtype)

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

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        @nnx.remat
        def run(x, block):
            return block(x, cos, sin, seq_len, mask1d, training)

        x = run(x, self.blocks)
        x = self.final_norm(x)
        return self.head(x), seq_len
