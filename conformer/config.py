from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.98
    num_epochs: int = 4
    batch_size: int = 16
    val_every_n_steps: int = 500
    lr_init_value: float = 1e-7
    lr_peak_value: float = 1e-3
    lr_warmup_steps: int = 500
    lr_decay_steps: int = 20000
    lr_end_value: float = 1e-6


@dataclass
class FeaturizerConfig:
    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80

@dataclass
class ConformerConfig:
    input_dim: int = 80
    num_encoder_layers: int = 4
    encoder_dim: int = 256
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    subsampling_factor: int = 4
    dtype: jnp.dtype = jnp.float32

@dataclass
class DataConfig:
    checkpoints_path: str = '/home/penguin/data/ka/checkpoints'
    tokenizer_path: str = '/home/penguin/data/ka/packed_dataset/tokenizer.pkl'
    train_data_path: str = '/home/penguin/data/ka/packed_dataset/train.array_record'
    test_data_path: str = '/home/penguin/data/ka/packed_dataset/test.array_record'
    batch_size: int = 16
    worker_count: int = 8
    prefetch_buffer_size: int = 16
    bucket_sizes: list[tuple[int, int]] = None

    def __post_init__(self):
        if self.bucket_sizes is None:
            self.bucket_sizes = [
                (64000, 64),
                (96000, 96),
                (128000, 128),
                (160000, 160),
                (192000, 192),
                (224000, 224),
                (256000, 256),
            ]
