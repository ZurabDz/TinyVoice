import os
import functools

os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import grain
import jax.numpy as jnp
from jiwer import cer, wer
from tqdm.auto import tqdm

from conformer.config import TrainingArguments
from conformer.dataset import ProcessAudioData, FilterByDuration, batch_fn
from conformer.factory import build_model, load_checkpoint
from conformer.tokenizer import Tokenizer


def ctc_decode(ids, tokenizer):
    result, prev = [], tokenizer.blank_id
    for _id in ids:
        _id = int(_id)
        if _id != prev and _id != tokenizer.blank_id:
            result.append(_id)
        prev = _id
    if hasattr(tokenizer, "id_to_char"):
        return "".join([tokenizer.id_to_char.get(i, f"[{i}]") for i in result])
    return tokenizer.decode(result).replace("\ufffd", "")


def decode_gt(tokens, tokenizer):
    ids = [
        int(t)
        for t in tokens
        if int(t) not in (tokenizer.blank_id, tokenizer.label_pad_token)
    ]
    if hasattr(tokenizer, "id_to_char"):
        return "".join([tokenizer.id_to_char.get(i, "") for i in ids])
    return tokenizer.decode(ids)


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

    test_source = grain.sources.ArrayRecordDataSource(
        args.data_dir + "/packed_dataset/test.array_record"
    )
    read_options = grain.ReadOptions(
        num_threads=args.worker_count,
        prefetch_buffer_size=args.prefetch_buffer_size * args.batch_size,
    )
    processed_test_dataset = (
        grain.MapDataset.source(test_source)
        .filter(
            FilterByDuration(sample_rate=args.sampling_rate, min_sec=1, max_sec=12.0)
        )
        .map(ProcessAudioData(tokenizer))
        .to_iter_dataset(read_options=read_options)
        .batch(
            batch_size=args.batch_size,
            batch_fn=functools.partial(
                batch_fn,
                bucket_sizes=args.bucket_sizes,
                pad_token_id=tokenizer.label_pad_token,
            ),
        )
    )

    ground_truth_texts, predicted_texts = [], []

    with open("transcribe.txt", "w", encoding="utf-8") as f:
        for count, batch in enumerate(tqdm(processed_test_dataset, desc="Inference")):
            padded_audios, frames, padded_labels, label_lengths = batch
            padded_audios = jnp.array(padded_audios, dtype=jnp.float32)
            frames = jnp.array(frames, dtype=jnp.int32)

            logits, out_lengths = model(
                padded_audios, training=False, inputs_lengths=frames
            )

            import numpy as np

            pred_ids = np.argmax(np.array(logits), axis=-1)
            for i in range(len(pred_ids)):
                pred_text = ctc_decode(pred_ids[i, : int(out_lengths[i])], tokenizer)
                gt_text = decode_gt(
                    padded_labels[i][: int(label_lengths[i])], tokenizer
                )
                f.write(f"\nSample {count * args.batch_size + i + 1}:\n")
                f.write(f"Ground Truth: {gt_text}\n")
                f.write(f"Prediction:   {pred_text}\n")
                ground_truth_texts.append(gt_text)
                predicted_texts.append(pred_text)

    print("WER:", wer(ground_truth_texts, predicted_texts))
    print("CER:", cer(ground_truth_texts, predicted_texts))


if __name__ == "__main__":
    main()
