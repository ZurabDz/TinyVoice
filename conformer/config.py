from dataclasses import dataclass, field
import jax.numpy as jnp


def _default_buckets():
    # Bucket sizes: (audio_frames, label_length)
    # Ensure audio_frames >= label_length * 704 (approx subsampling factor 640 + 10% margin)
    # Subsampling: hop_length=160 * conv_stride=4 = 640
    raw_buckets = [
        (16000, 35),
        (32000, 60),
        (48000, 94),
        (80000, 120),
        (128000, 154),
        (192000, 200),
    ]
    # Adjust audio_frames to be at least label_length * 704
    adjusted = []
    for audio_frames, label_len in raw_buckets:
        min_audio = int(label_len * 704)
        adjusted.append((max(audio_frames, min_audio), label_len))
    return adjusted


@dataclass
class TrainingArguments:
    learning_rate: float = 5e-4
    dtype: jnp.dtype = jnp.float16  
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
    num_encoder_layers: int = 6
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    feed_forward_dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    conv_dropout_p: float = 0.1
    conv_kernel_size: int = 9
    subsampling_factor: int = 4
    layer_drop_prob: float = 0.1

    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128

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
