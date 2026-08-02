"""Canonical audio loading used by preparation, training, and inference."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def canonicalize_audio(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Return finite mono float32 audio at ``target_rate``.

    Keeping this operation in one place prevents subtle train/inference drift
    caused by different audio libraries choosing different resamplers or by
    stereo files being handled differently.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"expected mono or stereo waveform, got shape {audio.shape}")
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, copy=False)
    if sample_rate != target_rate:
        audio = soxr.resample(audio, sample_rate, target_rate, quality="HQ")
    return np.asarray(audio, dtype=np.float32)


def load_audio_file(path: str | Path, target_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return canonicalize_audio(audio, sample_rate, target_rate)


def load_audio_bytes(audio_bytes: bytes, target_rate: int) -> np.ndarray:
    with io.BytesIO(audio_bytes) as fh:
        audio, sample_rate = sf.read(fh, dtype="float32", always_2d=False)
    return canonicalize_audio(audio, sample_rate, target_rate)
