from flax import nnx
import librosa
import math
import jax
import jax.numpy as jnp
from jax.scipy.signal import stft


def normalize_batch(x, seq_len):
    constant = 1e-5
    batch_size, num_features, max_time = x.shape
    
    # time_indices [T] vs seq_len [B, 1] -> broadcasts to [B, T]
    valid_mask = jnp.arange(max_time) < seq_len[:, None]  # [B, T]

    # Expand mask for [B, C, T] -> [B, 1, T]
    x_masked = jnp.where(valid_mask[:, None, :], x, 0.0)
    x_mean_numerator = x_masked.sum(axis=-1)  # [B, C]
    x_mean_denominator = seq_len  # [B]
    x_mean = x_mean_numerator / x_mean_denominator[:, None]  # [B, C]

    # Subtract 1 for Bessel's correction
    diff_masked = jnp.where(valid_mask[:, None, :], x - x_mean[:, :, None], 0.0)
    sum_sq = (diff_masked ** 2).sum(axis=-1)  # [B, C]
    x_std = jnp.sqrt(sum_sq / (x_mean_denominator[:, None] - 1.0))

    # Replace NaN (from seq_len=1) with 0, then add CONSTANT.
    x_std = jnp.nan_to_num(x_std, nan=0.0)
    x_std = x_std + constant

    normalized_x = (x - x_mean[:, :, None]) / x_std[:, :, None]
    
    return normalized_x, x_mean, x_std  

class AudioToMelSpectrogram(nnx.Module):
    def __init__(self, sample_rate, n_window_size, n_window_stride, n_fft, rng=None):
        self.rng = rng if rng else nnx.Rngs(0)
        self.sample_rate = sample_rate
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.n_fft = n_fft if n_fft else 2 ** math.ceil(math.log2(self.n_window_size))
        self.log = True
        self.pad_value = 0
        self.normalize = False

        self.log_zero_guard_value = 2 ** -24

        self.filterbanks = librosa.filters.mel(
            sr=16_000, n_fft=self.n_fft, n_mels=80, fmin=0, fmax=self.sample_rate / 2,
        )[None, :]

    def get_length(self, seq_len):
        pad_amount = self.n_fft // 2 * 2
        return jnp.floor_divide((seq_len + pad_amount - self.n_fft), self.n_window_stride)

    @nnx.jit
    def __call__(self, signal, lengths):
        # Mask to zero values beyond seq_len
        seq_len = self.get_length(lengths)

        # TODO: disable autocas
        f, t, Zxx = stft(signal, fs=self.sample_rate, nperseg=self.n_window_size, 
            noverlap=self.n_window_size - self.n_window_stride, nfft=self.n_fft)

        # convert complex number tensor into magnitude with guard for sqrt(if its grad?)
        x = jnp.abs(Zxx)
        x = jnp.pow(x, 2)

        max_len = x.shape[-1]

        # convert to human like mels
        x = jnp.matmul(self.filterbanks, x)

        if self.log:
            x = jnp.log(x + self.log_zero_guard_value)

        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len)

        # Create mask: (batch, 1, n_frames)
        mask = jnp.arange(max_len)[None, None, :] >= seq_len[:, None, None]
        x = jnp.where(mask, self.pad_value, x)
        
        return x, seq_len

        
