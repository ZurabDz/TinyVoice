from flax import nnx
import librosa
import math
import jax
import jax.numpy as jnp
from jax.scipy.signal import stft


def pitch_shift_mel(mel_specs, rng, semitones=2.0, prob=0.3, f_min=0.0, f_max=8000.0):
    """Pitch-shift mel spectrograms via frequency-bin interpolation.

    For each item in the batch, samples a random semitone shift in
    [-semitones, +semitones] with probability ``prob``, then remaps each
    destination bin k to its correct source bin via the exact mel-scale
    inversion:  src_bin = mel_bin(dest_freq / alpha).

    This runs entirely in JAX (vmapped over the batch), so it executes on
    device inside the JIT-compiled training step with no CPU overhead.

    Args:
        mel_specs: [B, n_mels, T] mel spectrogram (any dtype).
        rng: JAX random key.
        semitones: maximum shift magnitude in semitones.
        prob: per-sample application probability.
        f_min: lower frequency bound used when building the mel filterbank.
        f_max: upper frequency bound used when building the mel filterbank.

    Returns:
        mel_specs of the same shape and dtype with pitch shifts applied.
    """
    batch_size, n_mels, _ = mel_specs.shape
    orig_dtype = mel_specs.dtype
    # Compute in float32 for numerical stability
    x = mel_specs.astype(jnp.float32)

    rng, apply_key, shift_key = jax.random.split(rng, 3)
    apply_mask = jax.random.uniform(apply_key, (batch_size,)) < prob
    n_steps = jax.random.uniform(
        shift_key, (batch_size,), minval=-semitones, maxval=semitones
    )

    # Precompute mel-scale constants (scalar Python floats → traced as constants)
    mel_min = jnp.float32(2595.0 * math.log10(1.0 + f_min / 700.0))
    mel_max = jnp.float32(2595.0 * math.log10(1.0 + f_max / 700.0))
    log10 = jnp.float32(math.log(10.0))
    bin_idx = jnp.arange(n_mels, dtype=jnp.float32)

    def shift_one(mel_item, n_shift, apply):
        alpha = jnp.float32(2.0) ** (n_shift / 12.0)

        # Mel values at each destination bin centre
        dest_mels = mel_min + bin_idx / (n_mels - 1) * (mel_max - mel_min)
        # Corresponding frequencies (mel → Hz)
        dest_freqs = 700.0 * (jnp.exp(dest_mels / 2595.0 * log10) - 1.0)
        # Source frequency: pitch shift is a frequency scaling
        src_freqs = dest_freqs / alpha
        # Bins whose source frequency falls outside [f_min, f_max] are zeroed
        valid = (src_freqs >= f_min) & (src_freqs <= f_max)
        src_freqs_c = jnp.clip(src_freqs, f_min, f_max)
        # Source frequencies back to mel-bin indices (Hz → mel → bin)
        src_mels = 2595.0 * jnp.log(src_freqs_c / 700.0 + 1.0) / log10
        src_bins = (src_mels - mel_min) / (mel_max - mel_min) * (n_mels - 1)

        # Linear interpolation over the mel axis for all T frames simultaneously
        floor_b = jnp.clip(jnp.floor(src_bins).astype(jnp.int32), 0, n_mels - 2)
        frac = (src_bins - floor_b.astype(jnp.float32))[:, None]  # [n_mels, 1]
        low = mel_item[floor_b, :]  # [n_mels, T]
        high = mel_item[floor_b + 1, :]  # [n_mels, T]
        shifted = low + frac * (high - low)
        shifted = jnp.where(valid[:, None], shifted, 0.0)

        return jnp.where(apply, shifted, mel_item)

    result = jax.vmap(shift_one)(x, n_steps, apply_mask)
    return result.astype(orig_dtype)


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
    n_time_masks=10,
    freq_mask_param=27,
    time_mask_ratio=0.05,
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
        self.n_fft = n_fft if n_fft else 2 ** math.ceil(math.log2(self.n_window_size))
        self.log = True
        self.pad_value = 0
        self.normalize = normalize
        self.spec_augment = spec_augment
        self.pitch_shift = pitch_shift
        self.pitch_shift_semitones = pitch_shift_semitones
        self.pitch_shift_prob = pitch_shift_prob

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
            # We use nnx.Rngs to get a fresh key for each call
            x, _ = spec_augment(x, seq_len, self.rngs.dropout())

        # Create mask: (batch, 1, n_frames)
        mask = jnp.arange(max_len)[None, None, :] >= seq_len[:, None, None]
        x = jnp.where(mask, self.pad_value, x)

        return x, seq_len
