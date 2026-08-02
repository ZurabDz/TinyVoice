#!/usr/bin/env python
"""Transcribe audio with the exported NPU model.

Two backends, both driving the *same* graph and the same CPU frontend:

  onnx    ONNX Runtime on the exported float graph.  Fast, runs anywhere, and
          matches the trained JAX model to ~1e-2 relative error.
  acuity  The ACUITY simulator inside the SDK container, executing the
          quantised network.  This is what the NPU itself will compute, so it
          is the backend to trust when judging accuracy after quantisation.

    python scripts/npu_infer.py sample.wav
    python scripts/npu_infer.py sample.wav --backend acuity --container ed2eab5

The board itself runs the .nb through the VIPLite/OVX C runtime; the frontend
and CTC decode there are the NumPy code in ``conformer.npu_runtime``, which is
kept free of JAX for exactly that reason.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer.audio import load_audio_file  # noqa: E402
from conformer.npu_runtime import Artifacts  # noqa: E402

ACUITY = "/usr/local/acuity_command_line_tools"
SDK_MODELS = "/home/ai-sdk/models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="+", type=Path)
    p.add_argument("--model-dir", type=Path, default=Path("artifacts/npu/tinyvoice"))
    p.add_argument("--backend", choices=("onnx", "acuity"), default="onnx")
    p.add_argument("--container", help="SDK container id (required by --backend acuity)")
    p.add_argument("--qtype", default="int16", help="quantisation used by --backend acuity")
    return p.parse_args()


class OnnxBackend:
    """ONNX Runtime over the exported float graph."""

    def __init__(self, artifacts: Artifacts):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(artifacts.onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.label = "onnxruntime (float32)"

    def __call__(self, mel, mask):
        return self.session.run(["logits"], {"mel": mel, "mask": mask})[0][0]


class AcuityBackend:
    """Quantised execution through the ACUITY simulator in the SDK container.

    Bit-accurate to the NPU, and correspondingly slow -- each call pays for a
    container round trip plus a full graph rebuild.  Worth it when the question
    is "what does quantisation cost", not "how fast is this".
    """

    def __init__(self, artifacts: Artifacts, container: str, qtype: str):
        self.artifacts = artifacts
        self.container = container
        self.qtype = qtype
        self.remote = f"{SDK_MODELS}/{artifacts.root.name}"
        self.label = f"acuity simulator ({qtype})"

        quantize_file = f"{artifacts.name}_{qtype}.quantize"
        probe = subprocess.run(
            ["docker", "exec", container, "test", "-f", f"{self.remote}/{quantize_file}"]
        )
        if probe.returncode != 0:
            raise SystemExit(
                f"{self.remote}/{quantize_file} not found in container {container}.\n"
                f"Run: scripts/npu_convert.sh {container} <platform> {qtype}"
            )

    def __call__(self, mel, mask):
        name = self.artifacts.name
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "infer_mel.npy", mel)
            np.save(tmp / "infer_mask.npy", mask)
            for f in ("infer_mel.npy", "infer_mask.npy"):
                subprocess.run(
                    ["docker", "cp", str(tmp / f), f"{self.container}:{self.remote}/{f}"],
                    check=True,
                )

            # A copy of the inputmeta pointed at this single utterance.
            script = (
                f"cd {self.remote} && "
                f"cp {name}_inputmeta.yml infer_inputmeta.yml && "
                f"python3 npu_inputmeta.py infer_inputmeta.yml "
                f"mel=infer_mel.npy mask=infer_mask.npy >/dev/null && "
                f"python3 {ACUITY}/pegasus.py inference "
                f"--model {name}.json --model-data {name}.data "
                f"--dtype quantized --model-quantize {name}_{self.qtype}.quantize "
                f"--device CPU --iterations 1 "
                f"--with-input-meta infer_inputmeta.yml --output-dir ./inf/run"
            )
            result = subprocess.run(
                ["docker", "exec", "-e", f"ACUITY_PATH={ACUITY}", self.container, "bash", "-c", script],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                sys.stderr.write(result.stdout[-3000:] + result.stderr[-3000:])
                raise SystemExit("acuity inference failed")

            listing = subprocess.run(
                ["docker", "exec", self.container, "bash", "-c",
                 f"ls {self.remote}/inf/run/*attach_logits*.tensor | grep -v qnt"],
                capture_output=True, text=True, check=True,
            ).stdout.split()
            if not listing:
                raise SystemExit("acuity produced no logits tensor")

            remote_tensor = listing[0]
            subprocess.run(
                ["docker", "cp", f"{self.container}:{remote_tensor}", str(tmp / "logits.tensor")],
                check=True,
            )
            # Filename encodes the shape: ..._out0_1_274_36.tensor
            shape = [int(part) for part in Path(remote_tensor).stem.split("_")[-3:]]
            values = np.loadtxt(tmp / "logits.tensor", dtype=np.float32)
            return values.reshape(shape)[0]


def main() -> None:
    args = parse_args()
    artifacts = Artifacts(args.model_dir)

    if args.backend == "acuity":
        if not args.container:
            raise SystemExit("--backend acuity needs --container")
        backend = AcuityBackend(artifacts, args.container, args.qtype)
    else:
        backend = OnnxBackend(artifacts)

    meta = artifacts.meta
    print(
        f"model   {artifacts.name}  (checkpoint step {meta['checkpoint_step']}, "
        f"{meta['window_seconds']:.1f}s window)\n"
        f"backend {backend.label}\n"
    )

    for path in args.audio:
        audio = load_audio_file(path, meta["sampling_rate"])
        duration = len(audio) / meta["sampling_rate"]

        started = time.perf_counter()
        token_ids: list[int] = []
        for chunk in artifacts.chunks(audio):
            mel, mask, valid = artifacts.prepare(chunk)
            logits = backend(mel, mask)
            token_ids.extend(artifacts.token_ids(logits, valid).tolist())
        elapsed = time.perf_counter() - started

        text = artifacts.decode(token_ids)
        print(f"{path.name}  [{duration:.2f}s audio, {elapsed:.2f}s compute, {duration / elapsed:.1f}x realtime]")
        print(f"  {text}\n")


if __name__ == "__main__":
    main()
