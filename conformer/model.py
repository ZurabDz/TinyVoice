from flax import nnx
import jax
import jax.numpy as jnp
from .mel import MelSpectrogram
import numpy as np


class RelativePositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        pe = np.zeros((2 * max_len, d_model))
        position = np.arange(0, 2 * max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe, dtype=jnp.float32)

    def __call__(self, seq_len: int):
        # returns positional embeddings for relative distances -(seq_len-1) to (seq_len-1)
        max_len = (self.pe.shape[0]) // 2
        return self.pe[max_len - (seq_len - 1) : max_len + seq_len]


class RelativeMultiHeadAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_features: int,
        out_features: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = qkv_features // num_heads

        self.q_proj = nnx.Linear(d_model, qkv_features, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, qkv_features, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, qkv_features, use_bias=False, rngs=rngs)
        self.pos_proj = nnx.Linear(d_model, qkv_features, use_bias=False, rngs=rngs)

        self.out_proj = nnx.Linear(qkv_features, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # Learned biases for relative positioning
        self.u = nnx.Param(
            jax.random.normal(rngs.params(), (num_heads, self.head_dim))
        )
        self.v = nnx.Param(
            jax.random.normal(rngs.params(), (num_heads, self.head_dim))
        )

    def __call__(self, x, pos_emb, mask=None, training=True):
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).reshape(B, T, H, d)
        k = self.k_proj(x).reshape(B, T, H, d)
        v = self.v_proj(x).reshape(B, T, H, d)
        p = self.pos_proj(pos_emb).reshape(-1, H, d)  # (2T-1, H, d)

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
            scores = jnp.where(mask, scores, -1e9)

        probs = nnx.softmax(scores, axis=-1)
        probs = self.dropout(probs, deterministic=not training)

        attn_out = jnp.matmul(probs, v.transpose(0, 2, 1, 3))  # (B, H, T, d)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)

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
        x = x[:, :, 1:, :] # (B, H, 2T-1, T)
        x = x.reshape(B, H, T, 2 * T - 1)
        return x[:, :, :, :T]


class Conv2dSubSampler(nnx.Module):
    def __init__(self, d_model, rngs: nnx.Rngs):
        self.module = nnx.Sequential(
            nnx.Conv(
                in_features=1,
                out_features=d_model,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=d_model,
                out_features=d_model,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                rngs=rngs,
            ),
            nnx.relu,
        )

    def __call__(self, x):
        # B, T, D, 1(C)
        output = self.module(x)
        batch_size, subsampled_time, subsampled_freq, d_model = output.shape
        return output.reshape(batch_size, subsampled_time, subsampled_freq * d_model)


class FeedForwardBlock(nnx.Module):
    def __init__(self, d_model, expansion_factor, dropout, rngs: nnx.Rngs):
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        self.lin1 = nnx.Linear(d_model, d_model * expansion_factor, rngs=rngs)
        self.drop1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.lin2 = nnx.Linear(d_model * expansion_factor, d_model, rngs=rngs)
        self.drop2 = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, training=True):
        x = self.ln(x)
        x = self.lin1(x)
        x = nnx.silu(x)
        x = self.drop1(x, deterministic=not training)
        x = self.lin2(x)
        x = self.drop2(x, deterministic=not training)
        return x


class ConvBlock(nnx.Module):
    def __init__(self, d_model, dropout, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=d_model, out_features=d_model * 2, kernel_size=1, rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=d_model,
            out_features=d_model,
            kernel_size=31,
            feature_group_count=d_model,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(d_model, rngs=rngs)
        self.conv3 = nnx.Conv(
            in_features=d_model, out_features=d_model, kernel_size=1, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, training=True):
        x = self.layer_norm(x)

        x = self.conv1(x)
        x = nnx.glu(x)

        x = self.conv2(x)
        x = self.bn(x, use_running_average=not training)
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
        rngs: nnx.Rngs = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.residual_factor = feed_forward_residual_factor
        self.ff1 = FeedForwardBlock(
            d_model, feed_forward_expansion_factor, dropout, rngs=rngs
        )
        self.ln_before_attention = nnx.LayerNorm(d_model, rngs=rngs)
        self.attention = RelativeMultiHeadAttention(
            num_head,
            d_model,
            qkv_features=512,
            out_features=d_model,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.conv_block = ConvBlock(d_model, dropout=dropout, rngs=rngs)
        self.ff2 = FeedForwardBlock(
            d_model, feed_forward_expansion_factor, dropout, rngs=rngs
        )
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, pos_emb, mask=None, training=True):
        x = x + (self.residual_factor * self.ff1(x, training=training))
        x = x + self.attention(
            self.ln_before_attention(x),
            pos_emb,
            mask=mask,
            training=training,
        )
        x = x + self.conv_block(x, training=training)
        x = x + (self.residual_factor * self.ff2(x, training=training))
        return self.layer_norm(x)


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
        rngs: nnx.Rngs = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mel_spectogram = MelSpectrogram(rngs=rngs)
        self.conv_subsampler = Conv2dSubSampler(d_model=d_model, rngs=rngs)
        self.linear_proj = nnx.Linear(
            d_model * (((d_input - 1) // 2 - 1) // 2), d_model, rngs=rngs
        )
        self.rel_pos_encoding = RelativePositionalEncoding(d_model=d_model)
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
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nnx.Linear(d_model, token_count, rngs=rngs)
        self.d_model = d_model

    def __call__(self, x, mask=None, training=True, inputs_lengths=None):
        x = self.mel_spectogram(x, training, lengths=inputs_lengths)
        x = self.conv_subsampler(x[:, :, :, None])
        x = self.linear_proj(x)
        x = x * jnp.sqrt(self.d_model)

        # Generate relative positional embeddings for the current sequence length
        pos_emb = self.rel_pos_encoding(x.shape[1])
        x = self.dropout(x, deterministic=not training)

        for layer in self.layers:
            x = layer(x, pos_emb, mask, training=training)

        return self.decoder(x)

