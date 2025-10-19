from flax import nnx
from flax.nnx import initializers
from typing import Optional, Tuple
import jax.numpy as jnp
import jax
from .config import ConformerConfig, FeaturizerConfig
from .feedforward import FeedForwardModule
from .conv_subsampler import ConvolutionSubsampling
from .conv_module import ConvolutionModule
from .mel import MelSpectrogram
from .positional_encoder import PositionalEncoding


class ConformerBlock(nnx.Module):
    """A single block of the Conformer encoder."""

    def __init__(self, config: ConformerConfig, *, rngs: nnx.Rngs):
        self.ffn1 = FeedForwardModule(
            config.encoder_dim,
            config.feed_forward_expansion_factor,
            config.feed_forward_dropout_p,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.self_attn = nnx.MultiHeadAttention(
            config.num_attention_heads,
            config.encoder_dim,
            config.encoder_dim // config.num_attention_heads,
            dtype=config.dtype,
            dropout_rate=config.attention_dropout_p,
            rngs=rngs,
        )
        self.conv_module = ConvolutionModule(
            config.encoder_dim,
            config.conv_kernel_size,
            config.conv_expansion_factor,
            config.conv_dropout_p,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.ffn2 = FeedForwardModule(
            config.encoder_dim,
            config.feed_forward_expansion_factor,
            config.feed_forward_dropout_p,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.layer_norm = nnx.LayerNorm(
            config.encoder_dim, dtype=config.dtype, rngs=rngs
        )
        self.dropout = nnx.Dropout(config.attention_dropout_p, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, pad_mask: jnp.ndarray, *, training
    ) -> jnp.ndarray:
        x = x + 0.5 * self.ffn1(x, training=training)
        x = x + self.dropout(
            self.self_attn(x, mask=pad_mask, decode=False, deterministic=not training),
            deterministic=not training,
        )
        x = x + self.conv_module(x, training=training)
        x = x + 0.5 * self.ffn2(x, training=training)
        x = self.layer_norm(x)
        return x


class ConformerEncoder(nnx.Module):
    """The main Conformer model for ASR."""

    def __init__(
        self,
        config: ConformerConfig,
        feat_cfg: FeaturizerConfig,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.mel_feature = MelSpectrogram(
            feat_cfg.sampling_rate,
            feat_cfg.n_fft,
            feat_cfg.win_length,
            feat_cfg.hop_length,
            feat_cfg.n_mels,
            rngs=rngs,
            dither=feat_cfg.dither,
        )
        self.pos_encoder = PositionalEncoding(config.encoder_dim, 0.1, 2000, rngs=rngs)
        self.conv_subsampling = ConvolutionSubsampling(config, rngs=rngs)
        self.encoder_blocks = nnx.List(
            [
                ConformerBlock(config, rngs=rngs)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.output_linear = nnx.Linear(
            config.encoder_dim, num_classes, dtype=config.dtype, rngs=rngs
        )

    def __call__(
        self, x: jnp.ndarray, input_lengths: jnp.ndarray, *, training
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.mel_feature(x, training)
        x = self.conv_subsampling(x, training=training)
        x, _ = self.pos_encoder(x, training=training)

        # output_lengths = input_lengths // 160
        output_lengths = input_lengths
        output_lengths = output_lengths // 4

        max_len = x.shape[1]
        pad_mask = jnp.arange(max_len)[None, :] < output_lengths[:, None]
        pad_mask = jnp.expand_dims(pad_mask, axis=1)
        pad_mask = jnp.expand_dims(pad_mask, axis=1)

        for block in self.encoder_blocks:
            x = block(x, pad_mask, training=training)

        logits = self.output_linear(x)
        # log_probs = jax.nn.log_softmax(logits, axis=-1)

        return logits, output_lengths
