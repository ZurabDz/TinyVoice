from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class TrainingArguments:
    # Model
    d_model: int = 256
    num_encoder_layers: int = 16
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    dropout: float = 0.1
    conv_kernel_size: int = 9

    # Frontend
    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128

    # Optimizer
    grad_clip: float = 5.0
    weight_decay: float = 0.01
    lr_init_value: float = 1e-9
    lr_peak_value: float = 2e-4
    lr_warmup_steps: int = 5000
    lr_end_value: float = 1e-6

    # Training
    dtype: jnp.dtype = jnp.bfloat16
    num_epochs: int = 50
    batch_size: int = 24
    log_steps: int = 20
    save_steps: int = 500
    save_total_limit: int = 5
    checkpoint_dir: str = "./checkpoints"

    # Data — single fixed shape, no buckets
    data_dir: str = "/home/penguin/data/ka/packed_dataset"
    audio_frames_max: int = 176000  # 11 s @ 16 kHz
    label_length_max: int = 200
    min_audio_seconds: float = 1.0
    max_audio_seconds: float = 11.0
    enable_speed_perturb: bool = True
    enable_additive_noise: bool = True
    enable_reverb: bool = True
    mp_prefetch_workers: int = 8
    mp_prefetch_buffer: int = 4
