import jax
import jax.numpy as jnp
import librosa
import numpy as np
from flax import nnx
from jax.scipy.signal import stft


def _normalize(x, lengths):
    """Per-sample mean/std over the time axis, masked by length."""
    valid = (jnp.arange(x.shape[-1]) < lengths[:, None])[:, None, :]
    n = lengths.astype(x.dtype)[:, None, None]
    masked = jnp.where(valid, x, 0.0)
    mean = masked.sum(axis=-1, keepdims=True) / n
    var = jnp.where(valid, (x - mean) ** 2, 0.0).sum(axis=-1, keepdims=True) / n
    return (x - mean) / (jnp.sqrt(var) + 1e-5)


def _spec_augment(
    x,
    lengths,
    rng,
    n_freq_masks: int = 2,
    n_time_masks: int = 10,
    freq_mask_param: int = 27,
    time_mask_ratio: float = 0.05,
):
    """Vectorized SpecAugment: build all freq/time masks in one shot per type."""
    B, F, T = x.shape

    rng, k1, k2 = jax.random.split(rng, 3)
    f_size = jax.random.randint(k1, (B, n_freq_masks), 0, freq_mask_param)
    f_start = jax.random.randint(k2, (B, n_freq_masks), 0, jnp.maximum(F - f_size, 1))
    f_idx = jnp.arange(F)
    f_mask = (
        (f_idx[None, None, :] >= f_start[:, :, None])
        & (f_idx[None, None, :] < (f_start + f_size)[:, :, None])
    ).any(axis=1)
    x = jnp.where(f_mask[:, :, None], 0.0, x)

    rng, k1, k2 = jax.random.split(rng, 3)
    t_max = jnp.maximum((time_mask_ratio * lengths).astype(jnp.int32), 1)
    t_size = jax.random.randint(k1, (B, n_time_masks), 0, jnp.maximum(t_max[:, None], 1))
    t_start = jax.random.randint(
        k2, (B, n_time_masks), 0, jnp.maximum(lengths[:, None] - t_size, 1)
    )
    t_idx = jnp.arange(T)
    t_mask = (
        (t_idx[None, None, :] >= t_start[:, :, None])
        & (t_idx[None, None, :] < (t_start + t_size)[:, :, None])
    ).any(axis=1)
    return jnp.where(t_mask[:, None, :], 0.0, x)


class AudioToMelSpectrogram(nnx.Module):
    """Log-mel frontend with on-device SpecAugment."""

    def __init__(
        self,
        sample_rate: int,
        n_window_size: int,
        n_window_stride: int,
        n_fft: int,
        n_mels: int,
        *,
        rngs: nnx.Rngs,
        spec_augment: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.n_fft = n_fft
        self.spec_augment_enabled = spec_augment
        self.rngs = rngs

        filterbank = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sample_rate / 2
        )
        self.filterbank = nnx.Variable(jnp.asarray(filterbank, dtype=jnp.float32))

    def output_length(self, audio_lengths):
        pad = (self.n_fft // 2) * 2
        return jnp.floor_divide(audio_lengths + pad - self.n_fft, self.n_window_stride) + 1

    def __call__(self, signal, lengths, training: bool = True):
        spec_lengths = self.output_length(lengths)

        _, _, Zxx = stft(
            signal,
            fs=self.sample_rate,
            nperseg=self.n_window_size,
            noverlap=self.n_window_size - self.n_window_stride,
            nfft=self.n_fft,
        )
        power = jnp.abs(Zxx) ** 2  # (B, F, T)
        mel = jnp.matmul(self.filterbank.value, power)
        mel = jnp.log(mel + 2.0**-24)
        mel = _normalize(mel, spec_lengths)

        if training and self.spec_augment_enabled:
            mel = _spec_augment(mel, spec_lengths, self.rngs.spec_augment())

        pad_mask = jnp.arange(mel.shape[-1])[None, None, :] >= spec_lengths[:, None, None]
        mel = jnp.where(pad_mask, 0.0, mel)
        return mel, spec_lengths
