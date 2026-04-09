import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from jiwer import cer, wer
from tqdm.auto import tqdm

from conformer.config import TrainingArguments
from conformer.dataset import build_data_sources, build_test_loader
from conformer.decode import decode_token_ids, greedy_ctc_decode_text
from conformer.factory import build_model, load_checkpoint
from conformer.tokenizer import Tokenizer


def to_jax_batch(batch):
    audios, labels, audio_lengths, label_lengths = batch
    return (
        jnp.asarray(audios, dtype=jnp.float32),
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray(audio_lengths, dtype=jnp.int32),
        jnp.asarray(label_lengths, dtype=jnp.int32),
    )


def main():
    args = TrainingArguments()
    tokenizer_path = Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
    tokenizer = Tokenizer.load_tokenizer(tokenizer_path)

    model = build_model(args, tokenizer)
    model, latest_step = load_checkpoint(model, args.checkpoint_dir)
    if latest_step is None:
        print("No checkpoints found.")
        return
    print(f"Restored checkpoint step {latest_step}")

    _, test_source, _ = build_data_sources(
        args.data_dir,
        args.sampling_rate,
        args.batch_size,
        eval_split="test",
    )
    test_loader = build_test_loader(test_source, tokenizer, args)

    refs = []
    hyps = []

    with open("transcribe.txt", "w", encoding="utf-8") as fh:
        for batch_index, batch in enumerate(tqdm(test_loader, desc="Inference")):
            audios, labels, audio_lengths, label_lengths = to_jax_batch(batch)
            logits, output_lengths = model(audios, training=False, inputs_lengths=audio_lengths)

            logits_np = np.asarray(logits)
            output_lengths_np = np.asarray(output_lengths)
            labels_np = np.asarray(labels)
            label_lengths_np = np.asarray(label_lengths)

            for sample_index in range(logits_np.shape[0]):
                pred_text = greedy_ctc_decode_text(
                    logits_np[sample_index],
                    int(output_lengths_np[sample_index]),
                    tokenizer,
                )
                gt_text = decode_token_ids(
                    labels_np[sample_index, : int(label_lengths_np[sample_index])],
                    tokenizer,
                )
                sample_number = batch_index * args.batch_size + sample_index + 1
                fh.write(f"\nSample {sample_number}:\n")
                fh.write(f"Ground Truth: {gt_text}\n")
                fh.write(f"Prediction:   {pred_text}\n")
                refs.append(gt_text)
                hyps.append(pred_text)

    print("WER:", wer(refs, hyps))
    print("CER:", cer(refs, hyps))


if __name__ == "__main__":
    main()
