#!/usr/bin/env python
"""Check the device C frontend against the Python reference.

Builds device/host_test.c natively, runs it over a wav, and compares the mel it
produces with ``conformer.frontend_np``.  A mismatch here would show up on the
board as unexplained accuracy loss with nothing obvious to blame, so it is
worth checking before flashing anything.

    python scripts/check_device_frontend.py sample.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conformer.audio import load_audio_file  # noqa: E402
from conformer.frontend_np import log_mel, mel_frame_count  # noqa: E402
from conformer.npu_runtime import Artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("--model-dir", type=Path, default=Path("artifacts/npu/tinyvoice"))
    args = p.parse_args()

    artifacts = Artifacts(args.model_dir)
    assets = args.model_dir / "device"
    if not (assets / "tinyvoice_model.h").exists():
        raise SystemExit(f"{assets} missing -- run scripts/export_device_assets.py first")

    device = ROOT / "device"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        binary = tmp / "host_test"
        subprocess.run(
            ["cc", "-O2", "-o", str(binary),
             str(device / "host_test.c"), str(device / "tinyvoice_frontend.c"),
             f"-I{device}", f"-I{assets}", "-lm"],
            check=True,
        )

        # The C reader only handles 16-bit PCM, so normalise the input first.
        audio = load_audio_file(args.audio, artifacts.meta["sampling_rate"])
        wav = tmp / "input.wav"
        import soundfile as sf

        sf.write(wav, audio, artifacts.meta["sampling_rate"], subtype="PCM_16")

        mel_bin = tmp / "mel.bin"
        result = subprocess.run(
            [str(binary), str(assets / "filterbank.bin"), str(wav), str(mel_bin)],
            capture_output=True, text=True, check=True,
        )
        print(result.stdout.strip())

        meta = artifacts.meta
        c_mel = np.fromfile(mel_bin, dtype=np.float32).reshape(meta["mel_frames"], meta["n_mels"])

    # Feed the reference the same samples the C code saw: 16-bit quantised.
    reference_audio = np.round(np.asarray(audio, dtype=np.float32) * 32768.0) / 32768.0
    reference_audio = reference_audio[: meta["window_samples"]]

    valid = min(
        mel_frame_count(len(reference_audio), hop_length=meta["hop_length"]), meta["mel_frames"]
    )
    py_mel = log_mel(
        reference_audio,
        artifacts.filterbank,
        win_length=meta["win_length"],
        hop_length=meta["hop_length"],
        n_fft=meta["n_fft"],
        num_frames=meta["mel_frames"],
        valid_frames=valid,
    ).T

    a, b = py_mel[:valid], c_mel[:valid]
    error = float(np.abs(a - b).max())
    scale = max(float(np.abs(a).max()), 1e-6)
    print(f"valid frames        : {valid}")
    print(f"max abs difference  : {error:.3e}  ({error / scale:.2e} relative)")

    # The int16 quantisation step that follows is 2^-10, so anything well under
    # that is invisible to the model.
    step = 2.0**-10
    verdict = "OK" if error < step / 4 else "TOO LARGE - the C port has drifted"
    print(f"int16 input step    : {step:.3e}  [{verdict}]")
    if error >= step / 4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
