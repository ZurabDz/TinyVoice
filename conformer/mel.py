import librosa
import jax
import jax.numpy as jnp
from jax.scipy.signal import stft
from flax import nnx
from typing import Optional
import jax


class MelSpectrogram(nnx.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        power: float = 2.0,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        log_epsilon: float = 1e-10,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
        dither: float = None,
    ):

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2.0
        self.log_epsilon = log_epsilon
        self.dtype = dtype
        self.rngs = rngs
        self.dither = dither

        # Create and store the mel filterbank as a static parameter
        mel_fb = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            dtype=self.dtype,
        )
        self.mel_filterbank = jnp.array(mel_fb, dtype=self.dtype)

    def __call__(self, waveforms: jnp.ndarray, training) -> jnp.ndarray:
        if training and self.dither > 0:
            rand_waves = jax.random.normal(
                self.rngs.fork().default.key.value, shape=waveforms.shape
            )
            waveforms += rand_waves * 0.00001

        def process_single_waveform(waveform):
            _, _, stft_matrix = stft(
                waveform,
                fs=self.sample_rate,
                nperseg=self.win_length,
                noverlap=self.win_length - self.hop_length,
                nfft=self.n_fft,
                window="hann",
                boundary="constant",
                padded=False,
            )
            stft_matrix = jnp.transpose(stft_matrix)

            power_spectrogram = jnp.abs(stft_matrix) ** self.power

            # (Time, Freqs) @ (Freqs, Mels) -> (Time, Mels)
            mel_spectrogram = jnp.dot(power_spectrogram, self.mel_filterbank.T)

            log_mel_spectrogram = jnp.log(mel_spectrogram + self.log_epsilon)

            return log_mel_spectrogram

        # Use jax.vmap to process the batch of waveforms
        batched_mel_spectrogram_fn = jax.vmap(process_single_waveform)
        return batched_mel_spectrogram_fn(waveforms)
