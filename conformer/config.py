from dataclasses import dataclass, field

import jax.numpy as jnp


def _default_buckets() -> list[tuple[int, int]]:
    # (audio_frames, label_length) buckets sorted ascending by audio_frames.
    # Constraint: audio_frames >= label_length * 704 (hop_length 160 * conv stride
    # 4 = 640, +10% margin), so CTC always has at least one frame per label.
    return [
        (67584, 96),
        (87520, 120),
        (128480, 160),
        (176000, 200),
    ]


@dataclass
class TrainingArguments:
    # Optimizer
    grad_accumulation_steps: int = 1
    grad_clip: float = 5.0
    weight_decay: float = 0.01
    lr_init_value: float = 1e-9
    lr_peak_value: float = 2e-4
    lr_warmup_steps: int = 5000
    lr_decay_steps: int = 190000
    lr_end_value: float = 1e-6

    # Model
    d_model: int = 256
    num_encoder_layers: int = 16
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    feed_forward_dropout_p: float = 0.1
    conv_kernel_size: int = 9
    layer_drop_prob: float = 0.1
    layer_drop_anneal_steps: int = 20000
    entropy_weight: float = 0.02

    # Feature extraction
    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128

    # Trainer
    dtype: jnp.dtype = jnp.bfloat16
    num_epochs: int = 50
    batch_size: int = 24
    log_steps: int = 20
    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 500
    save_total_limit: int = 5
    loss_sync_steps: int = 50
    device_prefetch_batches: int = 4

    # Data loader
    data_dir: str = "/home/penguin/data/ka"
    worker_count: int = 16
    prefetch_buffer_size: int = 64
    enable_speed_perturb: bool = True
    enable_additive_noise: bool = True
    enable_reverb: bool = True
    bucket_sizes: list[tuple[int, int]] = field(default_factory=_default_buckets)

    def __post_init__(self):
        self.bucket_sizes = sorted(self.bucket_sizes, key=lambda b: b[0])
        if self.lr_warmup_steps >= self.lr_decay_steps:
            self.lr_warmup_steps = self.lr_decay_steps // 10
        self.loss_sync_steps = max(int(self.loss_sync_steps), 1)
        self.device_prefetch_batches = max(int(self.device_prefetch_batches), 1)
