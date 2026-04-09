import math

import jax
import jax.numpy as jnp
import librosa
from flax import nnx
from jax.scipy.signal import stft


def normalize_batch(x, seq_len):
    """Per-sample (mean, std) normalize over the time axis, ignoring padding."""
    _, _, max_time = x.shape
    valid = (jnp.arange(max_time) < seq_len[:, None])[:, None, :]  # [B, 1, T]

    n = seq_len.astype(x.dtype)[:, None]  # [B, 1]
    masked = jnp.where(valid, x, 0.0)
    mean = masked.sum(axis=-1) / n  # [B, C]
    diff = jnp.where(valid, x - mean[:, :, None], 0.0)
    var = (diff**2).sum(axis=-1) / jnp.maximum(n - 1.0, 1.0)  # Bessel
    std = jnp.sqrt(jnp.nan_to_num(var, nan=0.0)) + 1e-5
    return (x - mean[:, :, None]) / std[:, :, None]


def pitch_shift_mel(mel_specs, rng, semitones=2.0, prob=0.3, f_min=0.0, f_max=8000.0):
    """Pitch-shift via per-bin mel-scale interpolation, applied per sample."""
    batch_size, n_mels, _ = mel_specs.shape

    rng, apply_key, shift_key = jax.random.split(rng, 3)
    apply_mask = jax.random.uniform(apply_key, (batch_size,)) < prob
    n_steps = jax.random.uniform(shift_key, (batch_size,), minval=-semitones, maxval=semitones)

    mel_min = jnp.float32(2595.0 * math.log10(1.0 + f_min / 700.0))
    mel_max = jnp.float32(2595.0 * math.log10(1.0 + f_max / 700.0))
    log10 = jnp.float32(math.log(10.0))
    bin_idx = jnp.arange(n_mels, dtype=jnp.float32)

    def shift_one(mel_item, n_shift, apply):
        alpha = jnp.float32(2.0) ** (n_shift / 12.0)
        dest_mels = mel_min + bin_idx / (n_mels - 1) * (mel_max - mel_min)
        dest_freqs = 700.0 * (jnp.exp(dest_mels / 2595.0 * log10) - 1.0)
        src_freqs = dest_freqs / alpha
        valid = (src_freqs >= f_min) & (src_freqs <= f_max)
        src_freqs_c = jnp.clip(src_freqs, f_min, f_max)
        src_mels = 2595.0 * jnp.log(src_freqs_c / 700.0 + 1.0) / log10
        src_bins = (src_mels - mel_min) / (mel_max - mel_min) * (n_mels - 1)

        floor_b = jnp.clip(jnp.floor(src_bins).astype(jnp.int32), 0, n_mels - 2)
        frac = (src_bins - floor_b.astype(jnp.float32))[:, None]
        low = mel_item[floor_b, :]
        high = mel_item[floor_b + 1, :]
        shifted = low + frac * (high - low)
        shifted = jnp.where(valid[:, None], shifted, 0.0)
        return jnp.where(apply, shifted, mel_item)

    return jax.vmap(shift_one)(mel_specs, n_steps, apply_mask)


def spec_augment(
    x,
    seq_len,
    rng,
    n_freq_masks=2,
    n_time_masks=10,
    freq_mask_param=27,
    time_mask_ratio=0.05,
):
    """SpecAugment with adaptive per-sample time mask widths."""
    batch_size, n_mels, n_frames = x.shape

    for _ in range(n_freq_masks):
        rng, k1, k2 = jax.random.split(rng, 3)
        f_size = jax.random.randint(k1, (batch_size,), 0, freq_mask_param)
        f_start = jax.random.randint(k2, (batch_size,), 0, jnp.maximum(n_mels - f_size, 1))
        f_idx = jnp.arange(n_mels)[None, :]
        f_mask = (f_idx >= f_start[:, None]) & (f_idx < (f_start + f_size)[:, None])
        x = jnp.where(f_mask[:, :, None], 0.0, x)

    seq_len_i32 = seq_len.astype(jnp.int32)
    for _ in range(n_time_masks):
        rng, k1, k2 = jax.random.split(rng, 3)
        max_t = jnp.maximum((time_mask_ratio * seq_len_i32).astype(jnp.int32), 1)
        t_size = jax.random.randint(k1, (batch_size,), 0, jnp.maximum(max_t, 1))
        t_start = jax.random.randint(k2, (batch_size,), 0, jnp.maximum(seq_len_i32 - t_size, 1))
        t_idx = jnp.arange(n_frames)[None, :]
        t_mask = (t_idx >= t_start[:, None]) & (t_idx < (t_start + t_size)[:, None])
        x = jnp.where(t_mask[:, None, :], 0.0, x)

    return x


class AudioToMelSpectrogram(nnx.Module):
    def __init__(
        self,
        sample_rate,
        n_window_size,
        n_window_stride,
        n_fft,
        n_mels=80,
        rng=None,
        normalize=False,
        spec_augment=False,
        pitch_shift=False,
        pitch_shift_semitones=2.0,
        pitch_shift_prob=0.3,
    ):
        self.rngs = rng if rng else nnx.Rngs(0)
        self.sample_rate = sample_rate
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.n_fft = n_fft if n_fft else 2 ** math.ceil(math.log2(n_window_size))
        self.normalize = normalize
        self.spec_augment = spec_augment
        self.pitch_shift = pitch_shift
        self.pitch_shift_semitones = pitch_shift_semitones
        self.pitch_shift_prob = pitch_shift_prob
        self.log_zero_guard = 2**-24

        self.filterbanks = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=n_mels,
            fmin=0,
            fmax=self.sample_rate / 2,
        )[None, :]

    def get_length(self, seq_len):
        pad = self.n_fft // 2 * 2
        return jnp.floor_divide(seq_len + pad - self.n_fft, self.n_window_stride) + 1

    def __call__(self, signal, lengths, training=True):
        seq_len = self.get_length(lengths)

        _, _, Zxx = stft(
            signal,
            fs=self.sample_rate,
            nperseg=self.n_window_size,
            noverlap=self.n_window_size - self.n_window_stride,
            nfft=self.n_fft,
        )
        x = jnp.abs(Zxx) ** 2  # power spectrogram
        x = jnp.matmul(self.filterbanks, x)
        x = jnp.log(x + self.log_zero_guard)

        if self.normalize:
            x = normalize_batch(x, seq_len)

        if self.pitch_shift and training:
            x = pitch_shift_mel(
                x,
                self.rngs.dropout(),
                semitones=self.pitch_shift_semitones,
                prob=self.pitch_shift_prob,
                f_min=0.0,
                f_max=self.sample_rate / 2,
            )

        if self.spec_augment and training:
            x = spec_augment(x, seq_len, self.rngs.dropout())

        # Zero out frames beyond seq_len.
        max_len = x.shape[-1]
        pad_mask = jnp.arange(max_len)[None, None, :] >= seq_len[:, None, None]
        x = jnp.where(pad_mask, 0.0, x)
        return x, seq_len
