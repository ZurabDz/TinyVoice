import os

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# TODO: Maybe I need to recheck epoch suffling instead of simple for loop there
# TODO: Which ones are necessary
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_command_buffer= "
    "--xla_gpu_strict_conv_algorithm_picker=false "
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

from conformer.tokenizer import Tokenizer, HuggingFaceBPETokenizer
from conformer.config import TrainingArguments
from conformer.utils.timing import Timer, Timers
from conformer.model import ZipformerEncoder
from flax import nnx
import optax
import grain
from conformer.dataset import (
    batch_fn,
    ProcessAudioData,
    SpeedPerturb,
    AddNoise,
    FilterByDuration,
)
import jax
import numpy as np
import jax.numpy as jnp
import jax.profiler
from pathlib import Path
import functools


# Enable JAX compilation caching
cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))

from tqdm.auto import tqdm

args = TrainingArguments()

tokenizer_path = Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
tokenizer = Tokenizer.load_tokenizer(tokenizer_path)
token_count = (
    tokenizer.vocab_size
    if hasattr(tokenizer, "vocab_size")
    else len(tokenizer.id_to_char)
)

model = ZipformerEncoder(
    token_count=token_count,
    num_layers=args.num_encoder_layers,
    d_model=args.d_model,
    num_head=args.num_attention_heads,
    dropout=args.feed_forward_dropout_p,
    feed_forward_expansion_factor=args.feed_forward_expansion_factor,
    layer_drop_prob=args.layer_drop_prob,
    d_input=args.n_mels,
    sample_rate=args.sampling_rate,
    n_fft=args.n_fft,
    n_window_size=args.win_length,
    n_window_stride=args.hop_length,
    dtype=args.dtype,
    rngs=nnx.Rngs(0),
)
model.initialize_weights(jax.random.PRNGKey(0))

accum_steps = args.grad_accumulation_steps

# LR schedule counts in optimizer steps (total_steps / accum_steps)
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=args.lr_init_value,
    peak_value=args.lr_peak_value,
    warmup_steps=args.lr_warmup_steps // accum_steps,
    decay_steps=args.lr_decay_steps // accum_steps,
    end_value=args.lr_end_value,
)


# Weight decay mask: apply weight decay only to weights (ndim > 1), not biases (ndim == 1)
def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim > 1, params)


inner_optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip),
    optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.98,
        eps=1e-7,
        weight_decay=args.weight_decay,
        mask=weight_decay_mask,
    ),
)

if accum_steps > 1:
    tx = optax.MultiSteps(inner_optimizer, every_k_schedule=accum_steps)
else:
    tx = inner_optimizer

optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

import tensorflow as tf
import orbax.checkpoint as ocp

# TensorBoard Setup
tb_writer = tf.summary.create_file_writer("./runs")

# Checkpoint Setup (using refactored API)
checkpoint_dir = os.path.abspath(args.checkpoint_dir)
options = ocp.CheckpointManagerOptions(
    max_to_keep=args.save_total_limit,
    save_interval_steps=args.save_steps,
    enable_async_checkpointing=True,
)
mngr = ocp.CheckpointManager(checkpoint_dir, options=options)

train_audio_source = grain.sources.ArrayRecordDataSource(
    args.data_dir + "/packed_dataset/train.array_record"
)

test_audio_source = grain.sources.ArrayRecordDataSource(
    args.data_dir + "/packed_dataset/test.array_record"
)


duration_filter = FilterByDuration(
    sample_rate=args.sampling_rate, min_sec=1, max_sec=12.0
)
map_train_audio_dataset = grain.MapDataset.source(train_audio_source).filter(
    duration_filter
)
map_test_audio_dataset = grain.MapDataset.source(test_audio_source).filter(
    duration_filter
)

steps_per_epoch = len(map_train_audio_dataset) // args.batch_size
total_steps = steps_per_epoch * args.num_epochs
print(
    f"Filtered train samples: {len(map_train_audio_dataset)}, steps/epoch: {steps_per_epoch}, total steps: {total_steps}"
)
print(
    f"LR decay_steps (config): {args.lr_decay_steps}, should be close to total steps ({total_steps})"
)


# Configure prefetching with threads (not multiprocess to avoid GPU init issues)
read_options = grain.ReadOptions(
    num_threads=args.worker_count,
    prefetch_buffer_size=args.prefetch_buffer_size * args.batch_size,
)

processed_test_dataset = (
    map_test_audio_dataset.map(ProcessAudioData(tokenizer))
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


def greedy_ctc_collapse(pred_ids, blank_id):
    """Collapse CTC output: remove blanks and repeated tokens."""
    result = []
    prev = blank_id
    for p in pred_ids:
        p = int(p)
        if p != prev and p != blank_id:
            result.append(p)
        prev = p
    return result


def edit_distance(a, b):
    """Levenshtein distance between two integer sequences."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


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

        # Zero out infinite CTC losses (e.g. from impossible alignments)
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
    model_graphdef: nnx.GraphDef,
    model_state: nnx.State,
    padded_audios: jnp.ndarray,
    padded_labels: jnp.ndarray,
    frames: jnp.ndarray,
    label_lengths: jnp.ndarray,
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

    per_sample_loss = optax.ctc_loss(
        logits.astype(jnp.float32),
        logit_paddings,
        padded_labels,
        label_paddings,
        blank_id=tokenizer.blank_id,
    )
    per_sample_loss = jnp.where(jnp.isfinite(per_sample_loss), per_sample_loss, 0.0)

    return per_sample_loss.mean(), logits, real_times


# Initial Split
model_graphdef, model_state = nnx.split(model)
optimizer_graphdef, optimizer_state = nnx.split(optimizer)

print("Pre-compiling for all buckets...")

# Build abstract (shape-only) versions of the static pytrees once.
abstract_model_state = jax.eval_shape(lambda: model_state)
abstract_optimizer_state = jax.eval_shape(lambda: optimizer_state)

for b_frames, b_label in tqdm(args.bucket_sizes, desc="Compiling buckets"):
    # Dummy inputs — only shape & dtype matter, values are never used.
    d_audios = jnp.zeros((args.batch_size, b_frames), dtype=jnp.float32)
    d_labels = jnp.zeros((args.batch_size, b_label), dtype=jnp.int32)
    d_frames = jnp.full((args.batch_size,), b_frames, dtype=jnp.int32)
    d_label_lengths = jnp.full((args.batch_size,), b_label, dtype=jnp.int32)

    # Lower using abstract states — real buffers are never donated.
    train_step.lower(
        model_graphdef,
        abstract_model_state,
        optimizer_graphdef,
        abstract_optimizer_state,
        d_audios,
        d_labels,
        d_frames,
        d_label_lengths,
    ).compile()
    print(f"Compiled bucket: audio_frames={b_frames}, label_len={b_label}")

# Training Loop
global_step = 0
timers = Timers()

for epoch in range(args.num_epochs):
    train_loss_sum = 0.0
    train_steps = 0
    train_time_accum = 0.0
    data_time_accum = 0.0

    processed_train_dataset = (
        map_train_audio_dataset.shuffle(seed=42 + epoch)
        .to_iter_dataset(read_options=read_options)
        .map(ProcessAudioData(tokenizer))
        .random_map(SpeedPerturb(sample_rate=args.sampling_rate), seed=42 + epoch)
        .random_map(AddNoise(), seed=1000 + epoch)
        .batch(
            batch_size=args.batch_size,
            batch_fn=functools.partial(
                batch_fn,
                bucket_sizes=args.bucket_sizes,
                pad_token_id=tokenizer.label_pad_token,
            ),
        )
    )

    pbar = tqdm(processed_train_dataset, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
    for element in pbar:
        with timers("data_loading"):
            padded_audios, frames, padded_labels, label_lengths = element
            padded_audios = jnp.array(padded_audios, dtype=jnp.float32)
            padded_labels = jnp.array(padded_labels, dtype=jnp.int32)
            frames = jnp.array(frames, dtype=jnp.int32)
            label_lengths = jnp.array(label_lengths, dtype=jnp.int32)

        with timers("train_step"):
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

        train_loss_sum += float(loss)
        train_steps += 1
        global_step += 1

        train_time_accum += timers("train_step").current_elapsed
        data_time_accum += timers("data_loading").current_elapsed

        with timers("checkpoint"):
            if global_step > 1 and mngr.should_save(global_step):
                mngr.save(
                    global_step,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(model_state),
                        optimizer=ocp.args.StandardSave(optimizer_state),
                    ),
                )

        if global_step % 5 == 0 or global_step < 10:
            avg_train_loss = train_loss_sum / train_steps
            avg_step_time = (train_time_accum / train_steps) * 1000
            avg_data_time = (data_time_accum / train_steps) * 1000
            pbar.set_postfix(
                {
                    "loss": f"{avg_train_loss:.4f}",
                    "finite": f"{float(finite_ratio):.2f}",
                    "step": global_step,
                    "time": f"{avg_step_time:.0f}ms",
                }
            )
            current_lr = float(lr_schedule(global_step // accum_steps))
            with tb_writer.as_default():
                tf.summary.scalar("train/loss", avg_train_loss, step=global_step)
                tf.summary.scalar(
                    "train/finite_ratio", float(finite_ratio), step=global_step
                )
                tf.summary.scalar("train/learning_rate", current_lr, step=global_step)
                tf.summary.scalar("train/step_time_ms", avg_step_time, step=global_step)
                tf.summary.scalar("train/data_time_ms", avg_data_time, step=global_step)

    timers.log(
        ["data_loading", "train_step", "checkpoint"],
        normalizer=train_steps,
        tb_writer=tb_writer,
        step=global_step,
    )
    train_time_accum = 0.0
    data_time_accum = 0.0
    timers.reset()

    # Validation after each epoch
    print(f"\nRunning validation after epoch {epoch + 1}...")
    val_loss_sum = 0.0
    val_steps = 0
    val_cer_dist = 0
    val_cer_len = 0

    val_timer = Timer("validation")
    with val_timer:
        for element in tqdm(processed_test_dataset, desc="Validation"):
            padded_audios, frames, padded_labels, label_lengths = element
            # Convert numpy arrays to JAX arrays
            padded_audios = jnp.array(padded_audios, dtype=jnp.float32)
            padded_labels = jnp.array(padded_labels, dtype=jnp.int32)
            frames = jnp.array(frames, dtype=jnp.int32)
            label_lengths = jnp.array(label_lengths, dtype=jnp.int32)

            val_loss, logits, out_lengths = eval_step(
                model_graphdef,
                model_state,
                padded_audios,
                padded_labels,
                frames,
                label_lengths,
            )
            val_loss_sum += float(val_loss)
            val_steps += 1

        # Greedy CTC decode for CER
        pred_ids = np.argmax(np.array(logits), axis=-1)  # (B, T)
        for i, (pred, length, ref, ref_len) in enumerate(
            zip(
                pred_ids,
                np.array(out_lengths),
                np.array(padded_labels),
                np.array(label_lengths),
            )
        ):
            decoded = greedy_ctc_collapse(pred[:length], tokenizer.blank_id)
            reference = ref[:ref_len].tolist()
            val_cer_dist += edit_distance(decoded, reference)
            val_cer_len += len(reference)

    avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
    val_cer = val_cer_dist / max(val_cer_len, 1)
    avg_epoch_train_loss = train_loss_sum / train_steps
    print(
        f"Epoch {epoch + 1} - Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val CER: {val_cer:.4f}\n"
    )
    with tb_writer.as_default():
        tf.summary.scalar("epoch/train_loss", avg_epoch_train_loss, step=global_step)
        tf.summary.scalar("epoch/val_loss", avg_val_loss, step=global_step)
        tf.summary.scalar("epoch/val_cer", val_cer, step=global_step)
        tf.summary.scalar(
            "timers/validation", val_timer.elapsed * 1000, step=global_step
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
