from flax import nnx
import librosa
import math
import jax
import jax.numpy as jnp
from jax.scipy.signal import stft


def normalize_batch(x, seq_len):
    constant = 1e-5
    batch_size, num_features, max_time = x.shape
    
    # Cast to float32 for stable statistics computation in fp16 training
    compute_dtype = x.dtype
    if compute_dtype == jnp.float16:
        x = x.astype(jnp.float32)

    # time_indices [T] vs seq_len [B, 1] -> broadcasts to [B, T]
    valid_mask = jnp.arange(max_time) < seq_len[:, None]  # [B, T]

    # Expand mask for [B, C, T] -> [B, 1, T]
    x_masked = jnp.where(valid_mask[:, None, :], x, 0.0)
    x_mean_numerator = x_masked.sum(axis=-1)  # [B, C]
    x_mean_denominator = seq_len.astype(x.dtype)  # [B]
    x_mean = x_mean_numerator / x_mean_denominator[:, None]  # [B, C]

    # Subtract 1 for Bessel's correction
    diff_masked = jnp.where(valid_mask[:, None, :], x - x_mean[:, :, None], 0.0)
    sum_sq = (diff_masked**2).sum(axis=-1)  # [B, C]
    x_std = jnp.sqrt(sum_sq / (x_mean_denominator[:, None] - 1.0))

    # Replace NaN (from seq_len=1) with 0, then add CONSTANT.
    x_std = jnp.nan_to_num(x_std, nan=0.0)
    x_std = x_std + constant

    normalized_x = (x - x_mean[:, :, None]) / x_std[:, :, None]
    
    # Cast back to original dtype if needed
    if compute_dtype == jnp.float16:
        normalized_x = normalized_x.astype(compute_dtype)
        x_mean = x_mean.astype(compute_dtype)
        x_std = x_std.astype(compute_dtype)

    return normalized_x, x_mean, x_std


def spec_augment(
    x,
    seq_len,
    rng,
    n_freq_masks=2,
    n_time_masks=5,
    freq_mask_param=27,
    time_mask_ratio=0.10,
):
    """SpecAugment: Time and Frequency Masking.

    Uses adaptive time masking where the maximum mask width per sample is
    ``time_mask_ratio * seq_len`` (clamped to at least 1). This avoids masking
    too large a fraction of short utterances while still being effective on
    long ones.
    """
    batch_size, n_mels, n_frames = x.shape

    # Frequency masking
    for _ in range(n_freq_masks):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        f_size = jax.random.randint(subkey1, (batch_size,), 0, freq_mask_param)
        f_start = jax.random.randint(
            subkey2, (batch_size,), 0, jnp.maximum(n_mels - f_size, 1)
        )

        # Create mask: [B, F]
        f_indices = jnp.arange(n_mels)[None, :]
        f_mask = (f_indices >= f_start[:, None]) & (
            f_indices < (f_start + f_size)[:, None]
        )
        x = jnp.where(f_mask[:, :, None], 0.0, x)

    # Adaptive time masking (per-sample mask width proportional to seq_len)
    seq_len_i32 = seq_len.astype(jnp.int32)
    for _ in range(n_time_masks):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        # max mask width is time_mask_ratio * actual seq length, at least 1
        max_t = jnp.maximum((time_mask_ratio * seq_len_i32).astype(jnp.int32), 1)
        t_size = jax.random.randint(subkey1, (batch_size,), 0, jnp.maximum(max_t, 1))
        t_start = jax.random.randint(
            subkey2, (batch_size,), 0, jnp.maximum(seq_len_i32 - t_size, 1)
        )

        # Create mask: [B, T]
        t_indices = jnp.arange(n_frames)[None, :]
        t_mask = (t_indices >= t_start[:, None]) & (
            t_indices < (t_start + t_size)[:, None]
        )
        x = jnp.where(t_mask[:, None, :], 0.0, x)

    return x, rng


class AudioToMelSpectrogram(nnx.Module):
    def __init__(
        self,
        sample_rate,
        n_window_size,
        n_window_stride,
        n_fft,
        n_mels=80,
        rng=None,
    ):
        self.rngs = rng if rng else nnx.Rngs(0)
        self.sample_rate = sample_rate
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.n_fft = n_fft if n_fft else 2 ** math.ceil(math.log2(self.n_window_size))
        self.log = True
        self.pad_value = 0
        self.normalize = False
        self.spec_augment = False

        self.log_zero_guard_value = 2**-24

        self.filterbanks = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=n_mels,
            fmin=0,
            fmax=self.sample_rate / 2,
        )[None, :]

    def get_length(self, seq_len):
        pad_amount = self.n_fft // 2 * 2
        return (
            jnp.floor_divide((seq_len + pad_amount - self.n_fft), self.n_window_stride)
            + 1
        )

    def __call__(self, signal, lengths, training=True):
        # Mask to zero values beyond seq_len
        seq_len = self.get_length(lengths)

        # TODO: disable autocas
        f, t, Zxx = stft(
            signal,
            fs=self.sample_rate,
            nperseg=self.n_window_size,
            noverlap=self.n_window_size - self.n_window_stride,
            nfft=self.n_fft,
        )

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

        if self.spec_augment and training:
            # We use nnx.Rngs to get a fresh key for each call
            x, _ = spec_augment(x, seq_len, self.rngs.dropout())

        # Create mask: (batch, 1, n_frames)
        mask = jnp.arange(max_len)[None, None, :] >= seq_len[:, None, None]
        x = jnp.where(mask, self.pad_value, x)

        return x, seq_len
