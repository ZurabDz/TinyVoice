import os

os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from conformer.tokenizer import Tokenizer, HuggingFaceBPETokenizer
from pathlib import Path
from conformer.config import (
    DataConfig,
    ConformerConfig,
    FeaturizerConfig,
)
from conformer.model import ZipformerEncoder
from flax import nnx
import optax
import grain
from conformer.dataset import batch_fn, ProcessAudioData
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


def main():
    data_config = DataConfig()

    print(f"JAX background devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")

    # 1. Load Tokenizer
    tokenizer_path = Path(data_config.tokenizer_path)
    if tokenizer_path.suffix == ".json":
        tokenizer = HuggingFaceBPETokenizer.from_pretrained(tokenizer_path.parent)
    else:
        tokenizer = Tokenizer.load_tokenizer(tokenizer_path)

    featurizer_config = FeaturizerConfig()
    conformer_config = ConformerConfig()
    token_count = (
        tokenizer.vocab_size
        if hasattr(tokenizer, "vocab_size")
        else len(tokenizer.id_to_char)
    )

    model = ZipformerEncoder(
        token_count=token_count,
        num_layers=conformer_config.num_encoder_layers,
        d_model=conformer_config.encoder_dim,
        num_head=conformer_config.num_attention_heads,
        dropout=conformer_config.feed_forward_dropout_p,
        feed_forward_expansion_factor=conformer_config.feed_forward_expansion_factor,
        d_input=featurizer_config.n_mels,
        sample_rate=featurizer_config.sampling_rate,
        n_fft=featurizer_config.n_fft,
        n_window_size=featurizer_config.win_length,
        n_window_stride=featurizer_config.hop_length,
        dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )

    # Setup Checkpoint Manager (using refactored API)
    checkpoint_dir = os.path.abspath("./checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return

    options = ocp.CheckpointManagerOptions()
    mngr = ocp.CheckpointManager(checkpoint_dir, options=options)

    latest_step = mngr.latest_step()
    if latest_step is None:
        print("No checkpoints found.")
        return

    print(f"Restoring checkpoint from step {latest_step}...")

    # Setup model structure for restoration
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-4,
        warmup_steps=100,
        decay_steps=1000,
        end_value=0.0,
    )

    # Weight decay mask: apply weight decay only to weights (ndim > 1), not biases (ndim == 1)
    def weight_decay_mask(params):
        return jax.tree_util.tree_map(lambda x: x.ndim > 1, params)

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=0.9,
                b2=0.98,
                weight_decay=1e-3,
                mask=weight_decay_mask,
            ),
        ),
        wrt=nnx.Param,
    )

    # Restore checkpoint using refactored API
    restored = mngr.restore(
        latest_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model)),
            optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
        ),
    )

    # Update model with restored state
    nnx.update(model, restored.model)
    print("Model restored successfully.")
    mngr.close()

    # 5. Load Test Data
    try:
        test_audio_source = grain.sources.ArrayRecordDataSource(
            data_config.test_data_path
        )
    except Exception as e:
        print(f"Error loading test data: {e}. Please ensure data is generated.")
        return

    map_test_audio_dataset = grain.MapDataset.source(test_audio_source)

    processed_test_dataset = (
        map_test_audio_dataset.map(ProcessAudioData(tokenizer)).batch(
            batch_size=1, batch_fn=batch_fn
        )  # Batch size 1 for inference demo
    )

    # 6. Run Inference
    iterator = iter(processed_test_dataset)

    def only_georgian_chars(text):
        # Keep Georgian characters, spaces, and common punctuation
        return text

    def ctc_decode(ids, tokenizer):
        """CTC greedy decode: collapse repeated tokens and remove blanks."""
        collapsed_ids = []
        prev = -1
        for _id in ids:
            _id = int(_id)
            if _id != prev:
                if _id != tokenizer.blank_id and _id != getattr(
                    tokenizer, "padding_id", -1
                ):
                    collapsed_ids.append(_id)
            prev = _id

        # Use tokenizer's decode method (works for both char-level and BPE)
        if hasattr(tokenizer, "id_to_char"):
            return "".join(
                [tokenizer.id_to_char.get(_id, f"[{_id}]") for _id in collapsed_ids]
            )
        else:
            return tokenizer.decode(collapsed_ids).replace("\ufffd", "")

    def decode_gt(tokens, tokenizer):
        """Decode ground truth token IDs."""
        ids = [int(t) for t in tokens if int(t) != tokenizer.blank_id]
        if hasattr(tokenizer, "id_to_char"):
            return "".join([tokenizer.id_to_char.get(_id, "") for _id in ids])
        else:
            return tokenizer.decode(ids)

    print("\nStarting Inference on Test Set (Top 10 samples)...")

    ground_truth_texts = []
    predicted_texts = []

    count = 0
    from tqdm.auto import tqdm

    with open("transcribe.txt", "w", encoding="utf-8") as f:
        for batch in tqdm(iterator, desc="Inference"):
            # if count >= 10:
            #     break

            padded_audios, frames, padded_labels, label_lengths = batch

            # Forward pass
            logits, output_seq_len = model(
                padded_audios, training=False, inputs_lengths=frames
            )

            # Greedy decoding: argmax with sequence length masking
            seq_len = int(output_seq_len[0])
            predicted_ids = jnp.argmax(logits[0, :seq_len], axis=-1)

            # Batch size is 1, take first element
            pred_tokens = predicted_ids
            gt_tokens = padded_labels[0][
                : int(label_lengths[0])
            ]  # Slice to actual length

            pred_text = ctc_decode(pred_tokens, tokenizer)
            gt_text = decode_gt(gt_tokens, tokenizer)

            f.write(f"\nSample {count + 1}:\n")
            f.write(f"Ground Truth: {gt_text}\n")
            f.write(f"Prediction:   {pred_text}\n")

            ground_truth_texts.append(only_georgian_chars(gt_text))
            predicted_texts.append(only_georgian_chars(pred_text))

            count += 1

    # print("WER: ", wer(ground_truth_texts, predicted_texts))
    # print("CER: ", cer(ground_truth_texts, predicted_texts))
    # print(ground_truth_texts)
    # print(predicted_texts)


if __name__ == "__main__":
    main()
