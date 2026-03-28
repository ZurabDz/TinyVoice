import os

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_command_buffer= "
    "--xla_gpu_strict_conv_algorithm_picker=false "
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
    "--xla_gpu_ftz=true "
    "--xla_gpu_enable_fast_min_max=true "
)

from conformer.config import TrainingArguments
from conformer.decode import greedy_ctc_decode
from conformer.factory import build_model
from conformer.metrics import edit_distance, to_jax
from conformer.prefetch import PrefetchIterator
from conformer.steps import train_step, eval_step
from conformer.tokenizer import Tokenizer
from conformer.dataset import (
    build_data_sources,
    build_train_loader,
    build_test_loader,
)
from flax import nnx
import optax
import jax
import numpy as np
import jax.numpy as jnp
import jax.profiler
from pathlib import Path


# Enable JAX compilation caching
cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))

from tqdm.auto import tqdm

DEV_MODE = os.environ.get("TINYVOICE_DEV", "0") == "1"
DEV_STEPS = int(os.environ.get("TINYVOICE_DEV_STEPS", "10000"))
PROFILE_START_STEP = int(os.environ.get("TINYVOICE_PROFILE_START", "0"))
PROFILE_STEPS = int(os.environ.get("TINYVOICE_PROFILE_STEPS", "0"))

args = TrainingArguments()
if DEV_MODE:
    # Use the largest bucket only — every real sample fits, so the shape never
    # changes after step 1. Lazy JIT fires exactly once on the first batch.
    args.bucket_sizes = [args.bucket_sizes[-1]]
    print(
        f"DEV MODE: single bucket {args.bucket_sizes}, stopping after {DEV_STEPS} steps"
    )
else:
    print(f"Using dtype: {args.dtype}")
    print(f"Bucket sizes (audio_frames, label_len): {args.bucket_sizes}")

tokenizer_path = Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
tokenizer = Tokenizer.load_tokenizer(tokenizer_path)

model = build_model(args, tokenizer)

accum_steps = args.grad_accumulation_steps

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

map_train_audio_dataset, map_test_audio_dataset, steps_per_epoch = build_data_sources(
    args.data_dir, args.sampling_rate, args.batch_size
)
total_steps = steps_per_epoch * args.num_epochs
print(
    f"Filtered train samples: {len(map_train_audio_dataset)}, steps/epoch: {steps_per_epoch}, total steps: {total_steps}"
)

# Auto-align lr_decay_steps to actual training length so the cosine decay reaches
# lr_end_value exactly at the final step rather than flatlining early.
args.lr_decay_steps = total_steps
print(f"lr_decay_steps auto-set to {args.lr_decay_steps}")

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


# For fp16 training, use tighter gradient clipping
grad_clip_value = 1.0
inner_optimizer = optax.chain(
    # Clip by value first to prevent extreme gradients in fp16
    optax.clip(grad_clip_value),
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

# Adaptive loss scale state (replicates DynamicScale defaults)
loss_scale = jnp.array(2**15, dtype=jnp.float32)
scale_fin_steps = jnp.array(0, dtype=jnp.int32)

processed_test_dataset = build_test_loader(map_test_audio_dataset, tokenizer, args)

# Initial Split
model_graphdef, model_state = nnx.split(model)
optimizer_graphdef, optimizer_state = nnx.split(optimizer)

if not DEV_MODE:
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
            loss_scale,
            scale_fin_steps,
            d_audios,
            d_labels,
            d_frames,
            d_label_lengths,
            jnp.int32(0),  # step
            jnp.int32(tokenizer.blank_id),
            jnp.float32(args.entropy_weight),
        ).compile()
        print(f"Compiled bucket: audio_frames={b_frames}, label_len={b_label}")

# Training Loop
processed_train_dataset = build_train_loader(
    map_train_audio_dataset, tokenizer, args, args.num_epochs
)
global_step = 0

# Wrap the original iterator with prefetching
prefetch_iter = PrefetchIterator(iter(processed_train_dataset), to_jax, buffer_size=8)

# Prime: fetch the first batch (this will block until the worker produces one)
try:
    cur_audios, cur_frames, cur_labels, cur_label_lengths = next(prefetch_iter)
except StopIteration:
    raise RuntimeError("Dataset is empty!")

for epoch in range(args.num_epochs):
    train_loss_sum = jnp.float32(0.0)
    train_steps = 0

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
    for _ in pbar:
        # Start profiler trace at configured step
        if PROFILE_STEPS > 0 and global_step == PROFILE_START_STEP:
            jax.profiler.start_trace("./runs/profile")

        # 1. Dispatch — non-blocking, GPU starts immediately.
        (
            loss,
            finite_ratio,
            grad_finite,
            model_state,
            optimizer_state,
            loss_scale,
            scale_fin_steps,
        ) = train_step(
            model_graphdef,
            model_state,
            optimizer_graphdef,
            optimizer_state,
            loss_scale,
            scale_fin_steps,
            cur_audios,
            cur_labels,
            cur_frames,
            cur_label_lengths,
            jnp.int32(global_step),
            jnp.int32(tokenizer.blank_id),
            jnp.float32(args.entropy_weight),
        )

        # Stop profiler trace after configured number of steps
        if PROFILE_STEPS > 0 and global_step == PROFILE_START_STEP + PROFILE_STEPS:
            loss.block_until_ready()
            jax.profiler.stop_trace()

        # 2. While GPU runs, fetch the next batch from the prefetch queue.
        #    The worker thread has likely already prepared this.
        try:
            cur_audios, cur_frames, cur_labels, cur_label_lengths = next(prefetch_iter)
        except StopIteration:
            # Should not happen with .repeat(), but safe to handle
            break

        train_loss_sum = train_loss_sum + loss  # accumulate on-device, no CPU sync
        train_steps += 1
        global_step += 1

        if DEV_MODE and global_step >= DEV_STEPS:
            break

        if global_step > 1 and mngr.should_save(global_step):
            mngr.save(
                global_step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_state),
                    optimizer=ocp.args.StandardSave(optimizer_state),
                ),
            )

        if global_step % args.log_steps == 0 or global_step < 10:
            avg_train_loss = float(train_loss_sum) / train_steps  # single CPU sync
            pbar.set_postfix(
                {
                    "loss": f"{avg_train_loss:.4f}",
                    "finite": f"{float(finite_ratio):.2f}",
                    "step": global_step,
                }
            )
            current_lr = float(lr_schedule(global_step // accum_steps))
            with tb_writer.as_default():
                tf.summary.scalar("train/loss", avg_train_loss, step=global_step)
                tf.summary.scalar(
                    "train/finite_ratio", float(finite_ratio), step=global_step
                )
                tf.summary.scalar("train/learning_rate", current_lr, step=global_step)
                tf.summary.scalar(
                    "train/loss_scale", float(loss_scale), step=global_step
                )

    # Validation after each epoch
    print(f"\nRunning validation after epoch {epoch + 1}...")
    val_loss_sum = 0.0
    val_finite_ratio_sum = 0.0
    val_steps = 0
    val_cer_dist = 0
    val_cer_len = 0

    for element in tqdm(processed_test_dataset, desc="Validation"):
        padded_audios, frames, padded_labels, label_lengths = to_jax(element)

        val_loss, val_finite_ratio, logits, out_lengths = eval_step(
            model_graphdef,
            model_state,
            padded_audios,
            padded_labels,
            frames,
            label_lengths,
            jnp.int32(tokenizer.blank_id),
        )
        val_loss_sum += float(val_loss)
        val_finite_ratio_sum += float(val_finite_ratio)
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
            decoded = greedy_ctc_decode(np.array(logits)[i], length, tokenizer.blank_id)
            reference = ref[:ref_len].tolist()
            val_cer_dist += edit_distance(decoded, reference)
            val_cer_len += len(reference)

    avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
    avg_val_finite_ratio = val_finite_ratio_sum / val_steps if val_steps > 0 else 0.0
    val_cer = val_cer_dist / max(val_cer_len, 1)
    avg_epoch_train_loss = train_loss_sum / train_steps
    print(
        f"Epoch {epoch + 1} - Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val CER: {val_cer:.4f}, Val Finite: {avg_val_finite_ratio:.4f}\n"
    )
    with tb_writer.as_default():
        tf.summary.scalar("epoch/train_loss", avg_epoch_train_loss, step=global_step)
        tf.summary.scalar("epoch/val_loss", avg_val_loss, step=global_step)
        tf.summary.scalar("epoch/val_cer", val_cer, step=global_step)
        tf.summary.scalar(
            "epoch/val_finite_ratio", avg_val_finite_ratio, step=global_step
        )

    if DEV_MODE and global_step >= DEV_STEPS:
        break

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
