import os
from jiwer import wer, cer
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from conformer.tokenizer import Tokenizer
from pathlib import Path
from conformer.config import DataConfig, TrainingConfig, ConformerConfig
from conformer.model import ConformerEncoder
from flax import nnx
import optax
import grain
from conformer.dataset import batch_fn, ProcessAudioData, unpack_speech_data
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


def main():
    data_config = DataConfig()

    print(f"JAX background devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")

    # 1. Load Tokenizer
    tokenizer = Tokenizer.load_tokenizer(Path(data_config.tokenizer_path))

    conformer_config = ConformerConfig()
    model = ConformerEncoder(
        token_count=len(tokenizer.id_to_char),
        num_layers=conformer_config.num_encoder_layers,
        d_model=conformer_config.encoder_dim,
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

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=0.9,
                b2=0.98,
                weight_decay=1e-3,
                mask=0.1,
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
        return "".join([c for c in text if c in tokenizer.id_to_char.values()])

    def ctc_decode(ids, tokenizer):
        res = []
        prev = -1
        for _id in ids:
            _id = int(_id)
            if _id != prev:
                if _id != tokenizer.blank_id and _id != tokenizer.padding_id:
                    res.append(tokenizer.id_to_char.get(_id, f"[{_id}]"))
            prev = _id
        return "".join(res)

    print("\nStarting Inference on Test Set (Top 10 samples)...")

    ground_truth_texts = []
    predicted_texts = []

    count = 0
    from tqdm.auto import tqdm
    for batch in tqdm(iterator, desc="Inference"):
        if count >= 500:
            break

        padded_audios, frames, padded_labels, label_lengths = batch

        # Forward pass
        logits, output_seq_len = model(
            padded_audios, training=False, inputs_lengths=frames
        )

        # Greedy decoding: argmax
        predicted_ids = jnp.argmax(logits, axis=-1)

        # Batch size is 1, take first element
        pred_tokens = predicted_ids[0]
        gt_tokens = padded_labels[0][: int(label_lengths[0])]  # Slice to actual length

        pred_text = ctc_decode(pred_tokens, tokenizer)
        gt_text = "".join(
            [
                tokenizer.id_to_char.get(int(t), "")
                for t in gt_tokens
                if int(t) != tokenizer.blank_id
            ]
        )  # Skip blank/padding

        # print(f"\nSample {count + 1}:")
        # print(f"Ground Truth: {gt_text}")
        # print(f"Prediction:   {pred_text}")

        ground_truth_texts.append(only_georgian_chars(gt_text))
        predicted_texts.append(only_georgian_chars(pred_text))

        count += 1

    print("WER: ", wer(ground_truth_texts, predicted_texts))
    print("CER: ", cer(ground_truth_texts, predicted_texts))
    print(ground_truth_texts)
    print(predicted_texts)

if __name__ == "__main__":
    main()
