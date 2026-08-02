"""NumPy log-mel frontend matching the training-time JAX implementation.

The NPU graph starts at the mel spectrogram rather than at the waveform.  The
STFT and the per-utterance mean/variance normalisation are numerically
sensitive -- they involve ``log`` of very small magnitudes and a reduction over
the valid frames only -- so quantising them to int8/int16 would dominate the
total error budget.  They are also cheap relative to the 16 encoder blocks.

Keeping the frontend in NumPy means the same preprocessing runs on the target
board, where JAX and librosa are not available.  The mel filterbank itself is
exported alongside the ONNX graph so nothing here has to re-derive it.
"""

from __future__ import annotations

import numpy as np


def periodic_hann(window_size: int) -> np.ndarray:
    """``scipy.signal.get_window("hann", N, fftbins=True)``, used by JAX's STFT."""
    n = np.arange(window_size, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / window_size)


def stft_power(
    audio: np.ndarray,
    *,
    win_length: int,
    hop_length: int,
    n_fft: int,
) -> np.ndarray:
    """One-sided power spectrogram matching ``jax.scipy.signal.stft`` defaults.

    Reproduces ``boundary="zeros"`` (half-window zero padding on both sides),
    ``padded=True`` (tail padded to a whole number of hops) and
    ``scaling="spectrum"`` (amplitude divided by the window sum).

    Returns ``(n_fft // 2 + 1, frames)`` in float64 -- the caller downcasts
    after the log, which is where precision actually matters.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError(f"expected a mono waveform, got shape {audio.shape}")

    # boundary="zeros": half a window of zeros at each end.
    boundary = win_length // 2
    padded = np.pad(audio, (boundary, boundary))

    # padded=True: extend the tail so the frames tile the signal exactly.
    remainder = (len(padded) - win_length) % hop_length
    if remainder:
        padded = np.pad(padded, (0, hop_length - remainder))

    frames = 1 + (len(padded) - win_length) // hop_length
    starts = np.arange(frames) * hop_length
    segments = padded[starts[:, None] + np.arange(win_length)[None, :]]

    window = periodic_hann(win_length)
    spectrum = np.fft.rfft(segments * window, n=n_fft, axis=-1) / window.sum()
    return (spectrum.real**2 + spectrum.imag**2).T


def mel_frame_count(num_samples: int, *, hop_length: int) -> int:
    """Valid (unpadded) mel frames, matching ``AudioToMelSpectrogram.output_length``.

    The model defines this as ``floor(samples / hop) + 1``.  ``stft_power`` can
    return one frame more when ``samples`` is not a multiple of ``hop``; that
    extra frame is padding and is masked out.
    """
    return num_samples // hop_length + 1


def log_mel(
    audio: np.ndarray,
    filterbank: np.ndarray,
    *,
    win_length: int,
    hop_length: int,
    n_fft: int,
    num_frames: int | None = None,
    valid_frames: int | None = None,
) -> np.ndarray:
    """Normalised log-mel spectrogram, shaped ``(n_mels, num_frames)``.

    ``valid_frames`` selects which frames feed the mean/variance statistics --
    the model normalises over real audio only and then zeroes the padding.
    Pass ``num_frames`` to pad or trim the time axis to the fixed window the
    exported graph expects.
    """
    power = stft_power(
        audio, win_length=win_length, hop_length=hop_length, n_fft=n_fft
    )
    mel = np.matmul(np.asarray(filterbank, dtype=np.float64), power)
    mel = np.log(mel + 2.0**-24)

    if valid_frames is None:
        valid_frames = mel_frame_count(len(audio), hop_length=hop_length)
    valid_frames = int(np.clip(valid_frames, 1, mel.shape[-1]))

    # Statistics over the valid frames only, then normalise every frame.
    window = mel[:, :valid_frames]
    mean = window.mean(axis=-1, keepdims=True)
    var = np.square(window - mean).mean(axis=-1, keepdims=True)
    mel = (mel - mean) / (np.sqrt(var) + 1e-5)
    mel[:, valid_frames:] = 0.0

    if num_frames is not None:
        if mel.shape[-1] < num_frames:
            mel = np.pad(mel, ((0, 0), (0, num_frames - mel.shape[-1])))
        else:
            mel = mel[:, :num_frames]
    return mel.astype(np.float32)
