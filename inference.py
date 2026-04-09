import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jiwer import cer, wer
from tqdm.auto import tqdm

from conformer.config import TrainingArguments
from conformer.dataset import build_data_sources, build_eval_loader
from conformer.decode import decode_token_ids, greedy_ctc_decode_text
from conformer.factory import build_model, load_checkpoint
from conformer.tokenizer import Tokenizer


@nnx.jit
def forward(model, audios, audio_lengths):
    return model(audios, audio_lengths, training=False)


def main():
    args = TrainingArguments()
    tokenizer = Tokenizer.load_tokenizer(
        Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
    )

    model = build_model(args, tokenizer)
    model, latest_step = load_checkpoint(model, args.checkpoint_dir)
    if latest_step is None:
        print("No checkpoints found.")
        return
    print(f"Restored checkpoint step {latest_step}")

    _, test_source, _ = build_data_sources(args, eval_split="test")
    test_loader = build_eval_loader(test_source, tokenizer, args)

    refs, hyps = [], []
    with open("transcribe.txt", "w", encoding="utf-8") as fh:
        for batch_index, (audios, labels, audio_lengths, label_lengths) in enumerate(
            tqdm(test_loader, desc="Inference")
        ):
            logits, output_lengths = forward(model, jnp.asarray(audios), jnp.asarray(audio_lengths))
            logits_np = np.asarray(logits)
            out_lens = np.asarray(output_lengths)
            for i in range(logits_np.shape[0]):
                pred = greedy_ctc_decode_text(logits_np[i], int(out_lens[i]), tokenizer)
                gt = decode_token_ids(labels[i, : int(label_lengths[i])], tokenizer)
                fh.write(f"\nSample {batch_index * args.batch_size + i + 1}:\n")
                fh.write(f"Ground Truth: {gt}\n")
                fh.write(f"Prediction:   {pred}\n")
                refs.append(gt)
                hyps.append(pred)

    print("WER:", wer(refs, hyps))
    print("CER:", cer(refs, hyps))


if __name__ == "__main__":
    main()
