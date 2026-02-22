import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

from conformer.tokenizer import Tokenizer, HuggingFaceBPETokenizer
from conformer.config import (
    DataConfig,
    TrainingConfig,
    ConformerConfig,
    FeaturizerConfig,
)
from conformer.model import ConformerEncoder
from flax import nnx
import optax
import grain
from conformer.dataset import batch_fn, ProcessAudioData
import jax
import jax.numpy as jnp
import jax.profiler
from pathlib import Path
import functools

# Enable JAX compilation caching
cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))

from tqdm.auto import tqdm

data_config = DataConfig()
train_config = TrainingConfig()

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

model = ConformerEncoder(
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
model.initialize_weights(jax.random.PRNGKey(0))

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=train_config.lr_init_value,
    peak_value=train_config.lr_peak_value,
    warmup_steps=train_config.lr_warmup_steps,
    decay_steps=train_config.lr_decay_steps,
    end_value=train_config.lr_end_value,
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
            eps=1e-7,
            weight_decay=1e-2,
            mask=weight_decay_mask,
        ),
    ),
    wrt=nnx.Param,
)

import orbax.checkpoint as ocp

# Checkpoint Setup (using refactored API)
checkpoint_dir = os.path.abspath("./checkpoints")
options = ocp.CheckpointManagerOptions(
    max_to_keep=5, save_interval_steps=400, enable_async_checkpointing=True
)
mngr = ocp.CheckpointManager(checkpoint_dir, options=options)

train_audio_source = grain.sources.ArrayRecordDataSource(data_config.train_data_path)

test_audio_source = grain.sources.ArrayRecordDataSource(data_config.test_data_path)


map_train_audio_dataset = grain.MapDataset.source(train_audio_source)
map_test_audio_dataset = grain.MapDataset.source(test_audio_source)


# Configure prefetching with threads (not multiprocess to avoid GPU init issues)
read_options = grain.ReadOptions(
    num_threads=data_config.worker_count,
    prefetch_buffer_size=data_config.prefetch_buffer_size * data_config.batch_size,
)

processed_train_dataset = (
    map_train_audio_dataset.shuffle(seed=42)
    .map(ProcessAudioData(tokenizer))
    .batch(
        batch_size=data_config.batch_size,
        batch_fn=functools.partial(
            batch_fn,
            bucket_sizes=data_config.bucket_sizes,
            pad_token_id=tokenizer.blank_id,
        ),
    )
    .to_iter_dataset(read_options=read_options)
)

processed_test_dataset = (
    map_test_audio_dataset.map(ProcessAudioData(tokenizer))
    .batch(
        batch_size=data_config.batch_size,
        batch_fn=functools.partial(
            batch_fn,
            bucket_sizes=data_config.bucket_sizes,
            pad_token_id=tokenizer.blank_id,
        ),
    )
    .to_iter_dataset(read_options=read_options)
)


@jax.jit(static_argnums=(0, 2), donate_argnums=(1, 3))
def train_step(
    model_graphdef,
    model_state,
    optimizer_graphdef,
    optimizer_state,
    padded_audios,
    padded_labels,
    frames,
    label_lengths,
):
    """Training step using Functional API"""
    model = nnx.merge(model_graphdef, model_state)
    optimizer = nnx.merge(optimizer_graphdef, optimizer_state)

    def loss_fn(model):
        logits, real_times = model(padded_audios, training=True, inputs_lengths=frames)

        logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(
            jnp.float32
        )
        label_paddings = (
            jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]
        ).astype(jnp.float32)

        per_sample_loss = optax.ctc_loss(
            logits.astype(jnp.float32),
            logit_paddings,
            padded_labels,
            label_paddings,
            blank_id=tokenizer.blank_id,
        )
        is_finite = jnp.isfinite(per_sample_loss)
        finite_loss_ratio = is_finite.mean()

        # Clamp infinite CTC losses (e.g. from impossible alignments)
        # to prevent NaN gradients from corrupting the entire batch.
        per_sample_loss = jnp.where(is_finite, per_sample_loss, 0.0)
        loss = per_sample_loss.mean()

        return loss, finite_loss_ratio

    (loss, finite_loss_ratio), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model=model, grads=grads)

    _, new_model_state = nnx.split(model)
    _, new_optimizer_state = nnx.split(optimizer)

    return loss, finite_loss_ratio, new_model_state, new_optimizer_state


@jax.jit(static_argnums=(0,))
def eval_step(
    model_graphdef, model_state, padded_audios, padded_labels, frames, label_lengths
):
    """Evaluation step"""
    model = nnx.merge(model_graphdef, model_state)

    logits, real_times = model(padded_audios, training=False, inputs_lengths=frames)

    logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(
        jnp.float32
    )
    label_paddings = (
        jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]
    ).astype(jnp.float32)

    loss = optax.ctc_loss(
        logits.astype(jnp.float32),
        logit_paddings,
        padded_labels,
        label_paddings,
        blank_id=tokenizer.blank_id,
    ).mean()

    return loss


# Initial Split
model_graphdef, model_state = nnx.split(model)
optimizer_graphdef, optimizer_state = nnx.split(optimizer)

# mngr.save(
#     0,
#     args=ocp.args.Composite(
#         model=ocp.args.StandardSave(model_state),
#         optimizer=ocp.args.StandardSave(optimizer_state),
#     ),
# )
# mngr.wait_until_finished()

# Pre-compilation (Warmup)
# Use jax.jit(...).lower().compile() to trigger AOT compilation without
# executing the function, so the optimizer state and model weights stay clean.
print("Pre-compiling for all buckets...")
for b_frames, b_label in tqdm(data_config.bucket_sizes, desc="Compiling buckets"):
    # Dummy inputs for shape/dtype inference only
    d_audios = jnp.zeros((data_config.batch_size, b_frames), dtype=jnp.float32)
    d_labels = jnp.zeros((data_config.batch_size, b_label), dtype=jnp.int32)
    d_frames = jnp.full((data_config.batch_size,), b_frames, dtype=jnp.int32)
    d_label_lengths = jnp.full((data_config.batch_size,), b_label, dtype=jnp.int32)

    # Trigger AOT compilation without running the function
    jax.jit(train_step, static_argnums=(0, 2), donate_argnums=(1, 3)).lower(
        model_graphdef,
        model_state,
        optimizer_graphdef,
        optimizer_state,
        d_audios,
        d_labels,
        d_frames,
        d_label_lengths,
    ).compile()
    print(f"compiling {b_frames} {b_label} is done")

# Training Loop
global_step = 0

for epoch in range(train_config.num_epochs):
    train_loss_sum = 0.0
    train_steps = 0

    pbar = tqdm(
        processed_train_dataset, desc=f"Epoch {epoch + 1}/{train_config.num_epochs}"
    )
    for element in pbar:
        # 1. Start profiling at global_step 10
        # if global_step == 10:
        #     print("Starting JAX trace...")
        #     jax.profiler.start_trace("./logs")

        padded_audios, frames, padded_labels, label_lengths = element
        # Convert numpy arrays to JAX arrays
        padded_audios = jnp.array(padded_audios, dtype=jnp.float32)
        padded_labels = jnp.array(padded_labels, dtype=jnp.int32)
        frames = jnp.array(frames, dtype=jnp.int32)
        label_lengths = jnp.array(label_lengths, dtype=jnp.int32)

        # Basic validation
        if padded_audios.shape[0] == 0:
            print(f"Warning: Empty batch at step {global_step}, skipping...")
            continue

        try:
            loss, finite_ratio, model_state, optimizer_state = train_step(
                model_graphdef,
                model_state,
                optimizer_graphdef,
                optimizer_state,
                padded_audios,
                padded_labels,
                frames,
                label_lengths,
            )

            # Validation checks outside JIT
            loss_val = float(loss)
            if not jnp.isfinite(loss):
                print(f"Warning: Non-finite loss {loss_val} at step {global_step}")
            if loss_val > 1000:
                print(f"Warning: Very large loss {loss_val} at step {global_step}")

        except Exception as e:
            print(f"Error at step {global_step}: {e}")
            print(
                f"  Batch shape: audio={padded_audios.shape}, labels={padded_labels.shape}"
            )
            print(f"  Frames: {frames}, Label lengths: {label_lengths}")
            raise

        # 2. Stop profiling at global_step 20
        # if global_step == 20:
        #     print("Stopping JAX trace...")
        #     jax.profiler.stop_trace()

        # Convert loss to float to avoid accumulating JAX arrays
        train_loss_sum += float(loss)
        train_steps += 1
        global_step += 1

        # Save checkpoint only when needed (skip step 1)
        if global_step > 1 and mngr.should_save(global_step):
            mngr.save(
                global_step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_state),
                    optimizer=ocp.args.StandardSave(optimizer_state),
                ),
            )

        # Update tqdm only every 5 steps to reduce CPU-GPU sync
        if global_step % 5 == 0 or global_step < 10:
            avg_train_loss = train_loss_sum / train_steps
            pbar.set_postfix(
                {
                    "loss": f"{avg_train_loss:.4f}",
                    "finite": f"{float(finite_ratio):.2f}",
                    "step": global_step,
                }
            )

    # Validation after each epoch
    print(f"\nRunning validation after epoch {epoch + 1}...")
    val_loss_sum = 0.0
    val_steps = 0
    for element in tqdm(processed_test_dataset, desc="Validation"):
        padded_audios, frames, padded_labels, label_lengths = element
        # Convert numpy arrays to JAX arrays
        padded_audios = jnp.array(padded_audios, dtype=jnp.float32)
        padded_labels = jnp.array(padded_labels, dtype=jnp.int32)
        frames = jnp.array(frames, dtype=jnp.int32)
        label_lengths = jnp.array(label_lengths, dtype=jnp.int32)

        val_loss = eval_step(
            model_graphdef,
            model_state,
            padded_audios,
            padded_labels,
            frames,
            label_lengths,
        )
        val_loss_sum += float(val_loss)
        val_steps += 1

    avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
    print(
        f"Epoch {epoch + 1} - Train Loss: {train_loss_sum / train_steps:.4f}, Val Loss: {avg_val_loss:.4f}\n"
    )

# Save final checkpoint if not already saved
if not mngr.should_save(global_step):
    mngr.save(
        global_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(model_state),
            optimizer=ocp.args.StandardSave(optimizer_state),
        ),
    )
    mngr.wait_until_finished()
    print(f"Final checkpoint saved at step {global_step}")
