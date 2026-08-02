"""Runtime helpers for the exported NPU model.

Deliberately depends on NumPy only -- no JAX, no flax, no tokenizer pickle --
so the same code runs on the target board.  Everything it needs comes out of
the artifact directory written by ``scripts/export_onnx.py``:

    <name>.onnx            the graph (host-side reference execution)
    <name>_<qtype>.nb      the NPU network binary
    <name>_meta.json       frame geometry, vocabulary, frontend parameters
    filterbank.npy         mel filterbank lifted from the trained model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .frontend_np import log_mel, mel_frame_count


def subsampled_length(frames: int) -> int:
    """Frames surviving ``ConvSubsampler``: two 3x3 stride-2 VALID convolutions."""
    for _ in range(2):
        frames = (frames - 3) // 2 + 1
    return max(frames, 0)


def collapse_ctc(token_ids, blank_id: int = 0) -> list[int]:
    """Collapse repeats and drop blanks -- the NumPy twin of ``decode.collapse_ctc_ids``."""
    collapsed: list[int] = []
    previous = blank_id
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


class Artifacts:
    """The exported model directory, plus the preprocessing it implies."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        # The conversion also drops <name>_<qtype>_nbg_meta.json here, which
        # describes the network binary rather than the export -- skip it.
        candidates = sorted(
            p for p in self.root.glob("*_meta.json") if not p.name.endswith("_nbg_meta.json")
        )
        if not candidates:
            raise SystemExit(f"no *_meta.json in {self.root} -- run scripts/export_onnx.py")
        self.meta = json.loads(candidates[0].read_text())
        self.name = candidates[0].name.removesuffix("_meta.json")
        self.filterbank = np.load(self.root / self.meta["filterbank"])

    @property
    def onnx_path(self) -> Path:
        return self.root / f"{self.name}.onnx"

    def nb_path(self, qtype: str) -> Path:
        return self.root / f"{self.name}_{qtype}.nb"

    @property
    def window(self) -> int:
        return int(self.meta["window_samples"])

    def prepare(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Waveform (<= one window) -> (mel, mask, valid encoder frames).

        Both tensors are 4-D NCHW to match the exported graph's inputs.
        """
        meta = self.meta
        audio = np.asarray(audio, dtype=np.float32)[: self.window]

        valid_mel = min(
            mel_frame_count(len(audio), hop_length=meta["hop_length"]), meta["mel_frames"]
        )
        mel = log_mel(
            audio,
            self.filterbank,
            win_length=meta["win_length"],
            hop_length=meta["hop_length"],
            n_fft=meta["n_fft"],
            num_frames=meta["mel_frames"],
            valid_frames=valid_mel,
        )

        valid_out = int(np.clip(subsampled_length(valid_mel), 1, meta["seq_len"]))
        mask = np.zeros((1, 1, meta["seq_len"], 1), dtype=np.float32)
        mask[..., :valid_out, :] = 1.0
        return mel.T[None, None].astype(np.float32), mask, valid_out

    def chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split audio into window-sized pieces.

        The graph has a fixed sequence length, so anything longer than the
        window has to be cut.  Chunks are non-overlapping: attention cannot see
        across a boundary, which is the usual cost of a static-shape encoder.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) <= self.window:
            return [audio]
        return [audio[i : i + self.window] for i in range(0, len(audio), self.window)]

    def token_ids(self, logits: np.ndarray, valid: int) -> np.ndarray:
        return np.asarray(logits)[:valid].argmax(axis=-1)

    def decode(self, token_ids) -> str:
        vocab = self.meta["vocab"]
        blank = int(self.meta["blank_id"])
        skip = {blank, int(self.meta["pad_id"])}
        return "".join(
            vocab[i] for i in collapse_ctc(token_ids, blank_id=blank) if i not in skip
        )
