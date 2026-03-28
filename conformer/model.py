from flax import nnx
import jax
import jax.numpy as jnp
from .mel import AudioToMelSpectrogram
import numpy as np


class Conv2dSubSampler(nnx.Module):
    def __init__(self, d_model, rngs: nnx.Rngs, dtype=jnp.float32):
        self.conv1 = nnx.Conv(
            1,
            d_model // 4,
            (3, 3),
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
            dtype=dtype,
        )
        self.conv2 = nnx.Conv(
            d_model // 4,
            d_model,
            (3, 3),
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
            dtype=dtype,
        )

    def get_length(self, seq_len):
        seq_len = (seq_len - 3) // 2 + 1
        seq_len = (seq_len - 3) // 2 + 1
        return jnp.maximum(seq_len, 0)

    def __call__(self, x):
        x = nnx.silu(self.conv1(x))
        x = nnx.silu(self.conv2(x))
        B, T, F, C = x.shape
        return x.reshape(B, T, F * C)


class RotaryEmbedding(nnx.Module):
    def __init__(self, dim: int):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.inv_freq = jnp.array(inv_freq, dtype=jnp.float32)

    def __call__(self, x):  # x: (B, T, H, D)
        t = jnp.arange(x.shape[1], dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)[None, :, None, :].astype(x.dtype)
        sin = jnp.sin(emb)[None, :, None, :].astype(x.dtype)
        return cos, sin


def apply_rotary_emb(q, k, cos, sin):
    d = q.shape[-1] // 2
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    cos_h, sin_h = cos[..., :d], sin[..., :d]
    q_out = jnp.concatenate([q1 * cos_h - q2 * sin_h, q1 * sin_h + q2 * cos_h], axis=-1)
    k_out = jnp.concatenate([k1 * cos_h - k2 * sin_h, k1 * sin_h + k2 * cos_h], axis=-1)
    return q_out, k_out


class FastConformerFFN(nnx.Module):
    def __init__(self, d_model, expansion, dropout, rngs: nnx.Rngs, dtype=jnp.float32):
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.lin1 = nnx.Linear(d_model, d_model * expansion, rngs=rngs, dtype=dtype)
        self.lin2 = nnx.Linear(d_model * expansion, d_model, rngs=rngs, dtype=dtype)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, training=True):
        residual = x
        x = self.norm(x)
        x = nnx.silu(self.lin1(x))
        x = self.drop(x, deterministic=not training)
        x = self.lin2(x)
        x = self.drop(x, deterministic=not training)
        return residual + 0.5 * x


class FastConformerMHSA(nnx.Module):
    def __init__(self, d_model, num_heads, dropout, rngs: nnx.Rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(
            d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.v_proj = nnx.Linear(
            d_model, d_model, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs, dtype=dtype)
        self.drop = nnx.Dropout(dropout, rngs=rngs)
        self.rope = RotaryEmbedding(self.head_dim)

    def __call__(self, x, mask=None, training=True):
        residual = x
        x = self.norm(x)
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, d)
        k = self.k_proj(x).reshape(B, T, H, d)
        v = self.v_proj(x).reshape(B, T, H, d)

        cos, sin = self.rope(q)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Accumulate in float32 to prevent overflow
        scores = jnp.einsum(
            "bthd,bshd->bhts", q, k, preferred_element_type=jnp.float32
        ) / jnp.sqrt(jnp.float32(d))
        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min / 2)

        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.drop(attn, deterministic=not training)

        out = (
            jnp.einsum("bhts,bshd->bthd", attn, v, preferred_element_type=jnp.float32)
            .reshape(B, T, D)
            .astype(q.dtype)
        )
        return residual + self.drop(self.out_proj(out), deterministic=not training)


class FastConformerConvModule(nnx.Module):
    def __init__(
        self, d_model, kernel_size, dropout, rngs: nnx.Rngs, dtype=jnp.float32
    ):
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.pw1 = nnx.Conv(d_model, d_model * 2, (1,), rngs=rngs, dtype=dtype)
        self.dw = nnx.Conv(
            d_model,
            d_model,
            (kernel_size,),
            feature_group_count=d_model,
            padding="SAME",
            rngs=rngs,
            dtype=dtype,
        )
        self.post_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.pw2 = nnx.Conv(d_model, d_model, (1,), rngs=rngs, dtype=dtype)
        self.drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, mask=None, training=True):
        residual = x
        x = self.norm(x)
        x = nnx.glu(self.pw1(x))
        if mask is not None:
            m = mask[:, 0, 0, :, None].astype(x.dtype)
            x = x * m
        x = nnx.silu(self.post_norm(self.dw(x)))
        if mask is not None:
            x = x * m
        x = self.drop(self.pw2(x), deterministic=not training)
        return residual + x


class FastConformerBlock(nnx.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        expansion,
        conv_kernel,
        dropout,
        drop_prob,
        rngs,
        dtype=jnp.float32,
        drop_anneal_steps=5000,
    ):
        self.ff1 = FastConformerFFN(d_model, expansion, dropout, rngs, dtype)
        self.attn = FastConformerMHSA(d_model, num_heads, dropout, rngs, dtype)
        self.conv = FastConformerConvModule(d_model, conv_kernel, dropout, rngs, dtype)
        self.ff2 = FastConformerFFN(d_model, expansion, dropout, rngs, dtype)
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.drop_prob = drop_prob
        self.drop_anneal_steps = drop_anneal_steps
        self.rngs = rngs

    def __call__(self, x, mask=None, training=True, step=None):
        x_in = x
        x = self.ff1(x, training=training)
        x = self.attn(x, mask=mask, training=training)
        x = self.conv(x, mask=mask, training=training)
        x = self.ff2(x, training=training)
        x = self.norm(x)
        if training and self.drop_prob > 0.0:
            drop_prob = self.drop_prob
            if step is not None:
                drop_prob = self.drop_prob * jnp.minimum(
                    step / float(self.drop_anneal_steps), 1.0
                )
            keep = jax.random.bernoulli(self.rngs.dropout(), 1.0 - drop_prob)
            return jnp.where(keep, x, x_in)
        return x


@nnx.remat(static_argnums=(3,))
def remat_block_forward(block, x, mask, training, step):
    return block(x, mask=mask, training=training, step=step)


class FastConformerEncoder(nnx.Module):
    def __init__(
        self,
        token_count,
        d_input=128,
        d_model=256,
        num_layers=6,
        feed_forward_expansion_factor=4,
        num_head=4,
        conv_kernel_size=9,
        dropout=0.1,
        layer_drop_prob=0.1,
        rngs=None,
        dtype=jnp.float32,
        layer_drop_anneal_steps=5000,
        **featurizer_kwargs,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mel_spectogram = AudioToMelSpectrogram(
            n_mels=d_input,
            rng=rngs,
            normalize=True,
            spec_augment=True,
            pitch_shift=True,
            **featurizer_kwargs,
        )

        self.conv_subsampler = Conv2dSubSampler(d_model, rngs, dtype)
        freq_dim = ((d_input - 3) // 2 + 1 - 3) // 2 + 1
        self.linear_proj = nnx.Linear(
            d_model * freq_dim, d_model, rngs=rngs, dtype=dtype
        )

        self.layers = nnx.List(
            [
                FastConformerBlock(
                    d_model,
                    num_head,
                    feed_forward_expansion_factor,
                    conv_kernel_size,
                    dropout,
                    layer_drop_prob,
                    rngs,
                    dtype,
                    drop_anneal_steps=layer_drop_anneal_steps,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nnx.Linear(d_model, token_count, rngs=rngs, dtype=dtype)
        self.d_model = d_model

    def compute_mask(self, lengths, max_length):
        return (jnp.arange(max_length)[None, :] < lengths[:, None])[:, None, None, :]

    def __call__(self, x, mask=None, training=True, inputs_lengths=None, step=None):
        x, seq_len = self.mel_spectogram(x, lengths=inputs_lengths, training=training)
        x = jnp.transpose(x, (0, 2, 1))

        seq_len = self.conv_subsampler.get_length(seq_len)
        x = self.conv_subsampler(x[:, :, :, None])
        compute_dtype = self.linear_proj.kernel.value.dtype
        x = self.linear_proj(x).astype(compute_dtype)
        x = x * jnp.sqrt(self.d_model).astype(compute_dtype)

        if mask is None and inputs_lengths is not None:
            mask = self.compute_mask(seq_len, x.shape[1])

        for layer in self.layers:
            x = remat_block_forward(layer, x, mask, training, step)

        return self.decoder(x), seq_len

    def encode_from_mel(self, mel, seq_len, training=False):
        x = jnp.transpose(mel, (0, 2, 1))
        seq_len = self.conv_subsampler.get_length(seq_len)
        x = self.conv_subsampler(x[:, :, :, None])
        x = self.linear_proj(x)
        x = x * jnp.sqrt(self.d_model)
        for layer in self.layers:
            x = remat_block_forward(layer, x, None, training, None)
        return self.decoder(x), seq_len

    def initialize_weights(self, rng_key: jax.Array):
        num_layers = len(self.layers)
        residual_scale = 1.0 / jnp.sqrt(2.0 * num_layers)

        # Collect modules that are inside conformer blocks (residual branches)
        residual_modules = set()
        for layer in self.layers:
            for _, mod in layer.iter_modules():
                residual_modules.add(id(mod))

        def init_fn(module):
            nonlocal rng_key
            if isinstance(module, nnx.Linear):
                rng_key, k1, k2 = jax.random.split(rng_key, 3)
                limit = jnp.sqrt(3.0 / module.in_features)
                if id(module) in residual_modules:
                    limit = limit * residual_scale
                module.kernel.value = jax.random.uniform(
                    k1, module.kernel.shape, minval=-limit, maxval=limit
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        k2, module.bias.shape, minval=-limit, maxval=limit
                    )
            elif isinstance(module, nnx.Conv):
                rng_key, k1, k2 = jax.random.split(rng_key, 3)
                fan_in = np.prod(module.kernel_size) * module.in_features
                limit = jnp.sqrt(3.0 / fan_in)
                if id(module) in residual_modules:
                    limit = limit * residual_scale
                module.kernel.value = jax.random.uniform(
                    k1, module.kernel.shape, minval=-limit, maxval=limit
                )
                if hasattr(module, "bias") and isinstance(module.bias, nnx.Param):
                    module.bias.value = jax.random.uniform(
                        k2, module.bias.shape, minval=-limit, maxval=limit
                    )

        for _, module in self.iter_modules():
            init_fn(module)
