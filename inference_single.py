import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import librosa
import numpy as np

from conformer.config import TrainingArguments
from conformer.decode import greedy_ctc_decode_text
from conformer.factory import build_model, load_checkpoint
from conformer.tokenizer import Tokenizer


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    args = TrainingArguments()
    tokenizer_path = Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
    tokenizer = Tokenizer.load_tokenizer(tokenizer_path)

    model = build_model(args, tokenizer)
    model, latest_step = load_checkpoint(model, args.checkpoint_dir)
    if latest_step is None:
        print("No checkpoints found.")
        sys.exit(1)
    print(f"Restored checkpoint step {latest_step}")

    audio, _ = librosa.load(audio_path, sr=args.sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    print(f"Audio: {len(audio) / args.sampling_rate:.2f}s")

    padded_audios = jnp.zeros((1, len(audio)), dtype=jnp.float32).at[0, : len(audio)].set(audio)
    audio_lengths = jnp.array([len(audio)], dtype=jnp.int32)
    logits, output_lengths = model(padded_audios, training=False, inputs_lengths=audio_lengths)

    text = greedy_ctc_decode_text(
        np.asarray(logits[0]),
        int(output_lengths[0]),
        tokenizer,
    )
    print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
