import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import librosa
import numpy as np
from flax import nnx

from conformer.config import TrainingArguments
from conformer.decode import greedy_ctc_decode_text
from conformer.factory import build_model, load_checkpoint
from conformer.tokenizer import Tokenizer


@nnx.jit
def forward(model, audios, audio_lengths):
    return model(audios, audio_lengths, training=False)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    args = TrainingArguments()
    tokenizer = Tokenizer.load_tokenizer(
        Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
    )

    model = build_model(args, tokenizer)
    model, latest_step = load_checkpoint(model, args.checkpoint_dir)
    if latest_step is None:
        print("No checkpoints found.")
        sys.exit(1)
    print(f"Restored checkpoint step {latest_step}")

    audio, _ = librosa.load(audio_path, sr=args.sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    print(f"Audio: {len(audio) / args.sampling_rate:.2f}s")

    audios = jnp.asarray(audio[None, :])
    audio_lengths = jnp.asarray([len(audio)], dtype=jnp.int32)
    out = forward(model, audios, audio_lengths)

    text = greedy_ctc_decode_text(
        np.asarray(out["logits"][0]), int(out["output_lengths"][0]), tokenizer
    )
    print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
