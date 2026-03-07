from dataclasses import dataclass, field
from typing import Optional
import jax.numpy as jnp


def _default_buckets():
    return [
        (16000, 35),
        (32000, 60),
        (48000, 94),
        (80000, 120),
        (128000, 154),
        (192000, 200),
    ]


@dataclass
class TrainingArguments:
    learning_rate: float = 5e-4
    dtype: jnp.dtype = jnp.bfloat16
    weight_decay: float = 0.01
    grad_clip: float = 5.0

    num_epochs: int = 50
    batch_size: int = 16
    grad_accumulation_steps: int = 1

    lr_init_value: float = 1e-7
    lr_peak_value: float = 5e-4
    lr_warmup_steps: int = 2500
    lr_decay_steps: int = 190000
    lr_end_value: float = 1e-6

    log_steps: int = 5
    val_every: int = 500

    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 400
    save_total_limit: int = 5

    d_model: int = 256
    num_encoder_layers: int = 4
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    conv_expansion_factor: int = 2
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 31
    subsampling_factor: int = 4
    layer_drop_prob: float = 0.1

    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128
    d_input: int = 128

    data_dir: str = "/home/penguin/data/ka"
    worker_count: int = 8
    prefetch_buffer_size: int = 16
    bucket_sizes: list[tuple[int, int]] = field(default_factory=_default_buckets)

    def __post_init__(self):
        if self.bucket_sizes and self.bucket_sizes != sorted(
            self.bucket_sizes, key=lambda x: x[0]
        ):
            self.bucket_sizes = sorted(self.bucket_sizes, key=lambda x: x[0])
        if self.lr_warmup_steps >= self.lr_decay_steps:
            self.lr_warmup_steps = self.lr_decay_steps // 10


@dataclass
class FeaturizerConfig:
    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128


@dataclass
class ConformerConfig:
    input_dim: int = 128
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
    layer_drop_prob: float = 0.1


@dataclass
class DataConfig:
    checkpoints_path: str = "/home/penguin/data/ka/checkpoints"
    tokenizer_path: str = "/home/penguin/data/ka/packed_dataset/tokenizer.pkl"
    train_data_path: str = "/home/penguin/data/ka/packed_dataset/train.array_record"
    test_data_path: str = "/home/penguin/data/ka/packed_dataset/test.array_record"
    batch_size: int = 16
    worker_count: int = 8
    prefetch_buffer_size: int = 16
    bucket_sizes: Optional[list[tuple[int, int]]] = None

    def __post_init__(self):
        if self.bucket_sizes is None:
            self.bucket_sizes = [
                (16000, 35),
                (32000, 60),
                (48000, 94),
                (80000, 120),
                (128000, 154),
                (192000, 200),
            ]
