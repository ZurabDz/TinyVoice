#!/usr/bin/env python
"""Measure what quantisation cost the model.

``scripts/npu_convert.sh`` runs the calibration utterance through the ACUITY
simulator twice -- once in float32, once quantised -- and leaves both logit
dumps under ``inf/``.  This compares them, and, more usefully than any tensor
metric, decodes both so the actual transcription difference is visible.

    python scripts/npu_compare.py artifacts/npu/tinyvoice
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer.npu_runtime import Artifacts  # noqa: E402


def load_logits(directory: Path) -> np.ndarray:
    """Read an ACUITY ``.tensor`` dump (plain text, shape encoded in the name)."""
    matches = [p for p in directory.glob("*attach_logits*.tensor") if ".qnt." not in p.name]
    if not matches:
        raise SystemExit(f"no logits dump in {directory}")
    shape = [int(part) for part in matches[0].stem.split("_")[-3:]]
    return np.loadtxt(matches[0], dtype=np.float32).reshape(shape)[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", nargs="?", type=Path, default=Path("artifacts/npu/tinyvoice"))
    args = p.parse_args()

    artifacts = Artifacts(args.model_dir)
    inference_dir = args.model_dir / "inf"
    reference = load_logits(inference_dir / "float")

    others = sorted(d for d in inference_dir.iterdir() if d.is_dir() and d.name != "float")
    if not others:
        raise SystemExit(f"no quantised runs under {inference_dir}")

    # The dumps come from calibration sample 0. Its real length is recorded in
    # the calibration mask, so use that rather than guessing -- frames past it
    # hold unmasked garbage that would make the comparison meaningless.
    mask_path = args.model_dir / "calib_mask.npy"
    if mask_path.exists():
        active = int(np.load(mask_path)[0].reshape(-1).sum())
    else:
        active = reference.shape[0]
    print(f"comparing {active} valid frames of {reference.shape[0]}\n")
    print(f"float32 : {artifacts.decode(reference[:active].argmax(-1))}\n")

    for directory in others:
        candidate = load_logits(directory)
        span = min(len(reference), len(candidate))

        a, b = reference[:span].ravel(), candidate[:span].ravel()
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        agreement = float(
            (reference[:active].argmax(-1) == candidate[:active].argmax(-1)).mean()
        )

        print(f"{directory.name:8}: {artifacts.decode(candidate[:active].argmax(-1))}")
        print(
            f"          cosine={cosine:.6f}  "
            f"frame_argmax_agreement={agreement:.2%}  "
            f"max_abs_err={float(np.abs(a - b).max()):.4f}\n"
        )


if __name__ == "__main__":
    main()
