#!/usr/bin/env python
"""Export the trained FastConformer encoder to a fixed-shape ONNX graph.

Produces an ACUITY-ready model directory: the ``.onnx`` graph, the calibration
tensors the quantiser needs, and a metadata file describing everything the
runtime has to reproduce on the CPU side (mel filterbank, frame geometry,
vocabulary).  Feed the directory to ``scripts/npu_convert.sh``.

    python scripts/export_onnx.py --seconds 11 --calib-samples 64

The exported graph covers mel -> logits only; see ``conformer/frontend_np`` for
why the STFT stays on the CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer.audio import load_audio_file  # noqa: E402
from conformer.config import TrainingArguments  # noqa: E402
from conformer.factory import build_model, load_checkpoint  # noqa: E402
from conformer.frontend_np import log_mel, mel_frame_count  # noqa: E402
from conformer.onnx_builder import build_encoder_onnx  # noqa: E402
from conformer.tokenizer import Tokenizer  # noqa: E402

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seconds", type=float, default=11.0,
        help="fixed audio window baked into the graph (default: training max)",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/npu"))
    p.add_argument("--name", default="tinyvoice")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument(
        "--calib-samples", type=int, default=64,
        help="utterances used to calibrate the quantiser (0 disables)",
    )
    p.add_argument(
        "--calib-dir", type=Path, default=None,
        help="directory of audio files; defaults to the packed dev split",
    )
    p.add_argument(
        "--verify", type=int, default=3,
        help="utterances to check ONNX against the JAX model (0 disables)",
    )
    return p.parse_args()


def collect_calibration_audio(args, targs, count: int) -> list[np.ndarray]:
    """Real utterances for calibration -- silence or noise would mis-scale the encoder."""
    if count <= 0:
        return []

    if args.calib_dir is not None:
        paths = sorted(
            p for p in args.calib_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES
        )
        if not paths:
            raise SystemExit(f"no audio files under {args.calib_dir}")
        return [load_audio_file(p, targs.sampling_rate) for p in paths[:count]]

    record = Path(targs.data_dir) / "dev.array_record"
    if not record.exists():
        raise SystemExit(
            f"{record} not found -- pass --calib-dir with a folder of audio files"
        )

    import grain

    from conformer.audio import load_audio_bytes
    from conformer.dataset import unpack_speech_data

    source = grain.sources.ArrayRecordDataSource(str(record))
    limit = int(targs.max_audio_seconds * targs.sampling_rate)
    audios = []
    for index in range(len(source)):
        if len(audios) >= count:
            break
        metadata, audio_bytes = unpack_speech_data(source[index])
        if not (targs.sampling_rate <= metadata["frames"] <= limit):
            continue
        audios.append(load_audio_bytes(audio_bytes, targs.sampling_rate))
    return audios


def make_inputs(audio, filterbank, targs, meta) -> tuple[np.ndarray, np.ndarray, int]:
    """Waveform -> (mel, mask, valid encoder frames) for the exported graph."""
    from conformer.model import ConvSubsampler

    audio = np.asarray(audio, dtype=np.float32)
    window = meta["window_samples"]
    audio = audio[:window]

    valid_mel = min(mel_frame_count(len(audio), hop_length=targs.hop_length), meta["mel_frames"])
    mel = log_mel(
        audio,
        filterbank,
        win_length=targs.win_length,
        hop_length=targs.hop_length,
        n_fft=targs.n_fft,
        num_frames=meta["mel_frames"],
        valid_frames=valid_mel,
    )

    valid_out = int(np.clip(ConvSubsampler.output_length(np.int64(valid_mel)), 1, meta["seq_len"]))
    mask = np.zeros((1, 1, meta["seq_len"], 1), dtype=np.float32)
    mask[..., :valid_out, :] = 1.0
    return mel.T[None, None].astype(np.float32), mask, valid_out


def verify(onnx_path, model, audios, filterbank, targs, meta) -> None:
    """Compare the ONNX graph against the JAX model on real utterances."""
    import jax.numpy as jnp
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    window = meta["window_samples"]

    worst = 0.0
    for audio in audios:
        audio = np.asarray(audio, dtype=np.float32)[:window]
        padded = np.pad(audio, (0, window - len(audio)))

        reference, ref_len = model(
            jnp.asarray(padded[None]),
            jnp.asarray([len(audio)], dtype=jnp.int32),
            training=False,
        )
        reference = np.asarray(reference[0], dtype=np.float32)

        mel, mask, valid = make_inputs(audio, filterbank, targs, meta)
        logits = session.run(["logits"], {"mel": mel, "mask": mask})[0][0]

        span = min(valid, int(ref_len[0]), reference.shape[0])
        scale = max(np.abs(reference[:span]).max(), 1e-6)
        error = np.abs(logits[:span] - reference[:span]).max() / scale
        worst = max(worst, float(error))

        agree = (
            logits[:span].argmax(-1) == reference[:span].argmax(-1)
        ).mean()
        print(f"  frames={span:4d}  rel_err={error:.2e}  argmax_agreement={agree:.3%}")

    verdict = "OK" if worst < 2e-2 else "SUSPICIOUS - inspect before quantising"
    print(f"  worst relative error: {worst:.2e}  [{verdict}]")


def main() -> None:
    args = parse_args()
    targs = TrainingArguments(attn_impl="xla")
    if args.checkpoint_dir:
        targs.checkpoint_dir = args.checkpoint_dir

    tokenizer = Tokenizer.load_tokenizer(Path(targs.data_dir) / "tokenizer.pkl")
    model = build_model(targs, tokenizer)
    model, step = load_checkpoint(
        model, targs.checkpoint_dir, args=targs, tokenizer=tokenizer
    )
    if step is None:
        raise SystemExit(f"no checkpoints in {targs.checkpoint_dir}")
    print(f"Restored checkpoint step {step}")

    # Round the window down to a whole number of hops so the frame count is exact.
    window = int(args.seconds * targs.sampling_rate) // targs.hop_length * targs.hop_length
    mel_frames = mel_frame_count(window, hop_length=targs.hop_length)

    onnx_model, meta = build_encoder_onnx(
        model, mel_frames=mel_frames, n_mels=targs.n_mels
    )
    meta.update(
        window_samples=window,
        window_seconds=window / targs.sampling_rate,
        sampling_rate=targs.sampling_rate,
        n_fft=targs.n_fft,
        win_length=targs.win_length,
        hop_length=targs.hop_length,
        blank_id=int(tokenizer.blank_id),
        pad_id=int(tokenizer.pad_id),
        vocab=[tokenizer.id_to_char[i] for i in range(tokenizer.vocab_size)],
        checkpoint_step=int(step),
    )

    # ACUITY expects models/<NAME>/<NAME>.onnx, so mirror that layout.
    root = args.out / args.name
    root.mkdir(parents=True, exist_ok=True)

    onnx_path = root / f"{args.name}.onnx"
    import onnx

    onnx.save(onnx_model, str(onnx_path))
    print(
        f"\nONNX graph  {onnx_path}"
        f"\n  mel   (1, 1, {meta['mel_frames']}, {meta['n_mels']})"
        f"\n  mask  (1, 1, {meta['seq_len']}, 1)"
        f"\n  logits(1, {meta['seq_len']}, {meta['vocab_size']})"
        f"\n  window {meta['window_seconds']:.2f}s, {meta['num_layers']} layers, "
        f"{len(onnx_model.graph.node)} nodes"
    )

    filterbank = np.asarray(model.frontend.filterbank.value, dtype=np.float32)
    np.save(root / "filterbank.npy", filterbank)
    meta["filterbank"] = "filterbank.npy"
    (root / f"{args.name}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    audios = collect_calibration_audio(args, targs, max(args.calib_samples, args.verify))

    if args.calib_samples > 0:
        # ACUITY's NPY provider handles one port per database and np.load()s the
        # database path directly, so each input needs its own stacked array.
        mels, masks = [], []
        for audio in audios[: args.calib_samples]:
            mel, mask, _ = make_inputs(audio, filterbank, targs, meta)
            mels.append(mel[0])
            masks.append(mask[0])
        np.save(root / "calib_mel.npy", np.stack(mels).astype(np.float32))
        np.save(root / "calib_mask.npy", np.stack(masks).astype(np.float32))
        print(f"Calibration  {len(mels)} utterances -> calib_mel.npy / calib_mask.npy")

    (root / "inputs_outputs.txt").write_text(
        f"--inputs 'mel mask' "
        f"--input-size-list '1,{meta['mel_frames']},{meta['n_mels']}#1,{meta['seq_len']},1' "
        f"--outputs logits\n"
    )

    if args.verify > 0 and audios:
        print("\nVerifying ONNX against the JAX model:")
        verify(onnx_path, model, audios[: args.verify], filterbank, targs, meta)


if __name__ == "__main__":
    main()
