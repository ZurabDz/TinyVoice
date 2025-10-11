from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 16
    warmup_steps: int = 10000


@dataclass
class FeaturizerConfig:
    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    dither: float = 0.00001


@dataclass
class SpecAugmentConfig:
    freq_mask_param: int = 27
    n_freq_masks: int = 2
    time_mask_param: int = 100
    n_time_masks: int = 2


@dataclass
class ConformerConfig:
    input_dim: int = 80
    num_encoder_layers: int = 4
    encoder_dim: int = 128
    num_attention_heads: int = 2
    feed_forward_expansion_factor: int = 2
    conv_expansion_factor: int = 2
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    subsampling_factor: int = 4
    dtype: jnp.dtype = jnp.bfloat16
