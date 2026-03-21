from dataclasses import dataclass, field
import jax.numpy as jnp


def _round_up_to_multiple(x, m):
    return ((x + m - 1) // m) * m


def _default_buckets():
    # Bucket sizes: (audio_frames, label_length)
    # Ensure audio_frames >= label_length * 704 (approx subsampling factor 640 + 10% margin)
    # Subsampling: hop_length=160 * conv_stride=4 = 640
    # All dimensions padded to multiples of 8 for XLA vectorization
    raw_buckets = [
        (67040,  94),
        (87520,  120),
        (128480, 154),
        (176000, 200),
    ]
    adjusted = []
    for audio_frames, label_len in raw_buckets:
        label_len = _round_up_to_multiple(label_len, 8)
        min_audio = int(label_len * 704)
        audio_frames = _round_up_to_multiple(max(audio_frames, min_audio), 8)
        adjusted.append((audio_frames, label_len))
    return adjusted


@dataclass
class TrainingArguments:
    learning_rate: float = 5e-4
    dtype: jnp.dtype = jnp.float16
    weight_decay: float = 0.01
    grad_clip: float = 5.0

    num_epochs: int = 50
    batch_size: int = 24
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
    num_encoder_layers: int = 8
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 9
    subsampling_factor: int = 4
    layer_drop_prob: float = 0.1
    layer_drop_anneal_steps: int = 5000
    entropy_weight: float = 0.1

    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128

    data_dir: str = "/home/penguin/data/ka"
    worker_count: int = 16
    prefetch_buffer_size: int = 64
    bucket_sizes: list[tuple[int, int]] = field(default_factory=_default_buckets)

    def __post_init__(self):
        if self.bucket_sizes and self.bucket_sizes != sorted(
            self.bucket_sizes, key=lambda x: x[0]
        ):
            self.bucket_sizes = sorted(self.bucket_sizes, key=lambda x: x[0])
        if self.lr_warmup_steps >= self.lr_decay_steps:
            self.lr_warmup_steps = self.lr_decay_steps // 10
