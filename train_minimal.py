"""Minimal end-to-end CTC training for the FastConformer ASR encoder."""

import csv
import os
import queue
import threading
from pathlib import Path
from statistics import mean

import jax
import jax.numpy as jnp
import jiwer
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm.auto import tqdm

from conformer.config import TrainingArguments
from conformer.dataset import build_data_sources, build_test_loader, build_train_loader
from conformer.decode import decode_token_ids, greedy_ctc_decode_text
from conformer.factory import build_model
from conformer.tokenizer import Tokenizer

DEV_MODE = os.environ.get("TINYVOICE_DEV", "0") == "1"
DEV_STEPS = int(os.environ.get("TINYVOICE_DEV_STEPS", "10000"))


def configure_jax_cache():
    """Cache JIT compilations across runs."""
    cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))


def build_training_args() -> TrainingArguments:
    args = TrainingArguments()
    if DEV_MODE:
        args.bucket_sizes = [args.bucket_sizes[-1]]
        args.enable_speed_perturb = False
        args.enable_additive_noise = False
        args.enable_reverb = False
        print(
            "DEV MODE:"
            f" single bucket {args.bucket_sizes}, stopping at {DEV_STEPS} steps,"
            " waveform augments disabled"
        )
    else:
        print(f"dtype={args.dtype}  buckets={args.bucket_sizes}")
    return args


def load_tokenizer(args: TrainingArguments) -> Tokenizer:
    tokenizer_path = Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
    return Tokenizer.load_tokenizer(tokenizer_path)


def build_optimizer(args: TrainingArguments, model, total_steps: int):
    accum_steps = args.grad_accumulation_steps
    args.lr_decay_steps = total_steps
    schedule_steps = max(total_steps // accum_steps, 1)
    warmup_steps = max(args.lr_warmup_steps // accum_steps, 1)

    if schedule_steps <= 1:
        lr_schedule = optax.constant_schedule(args.lr_peak_value)
    elif warmup_steps >= schedule_steps:
        lr_schedule = optax.linear_schedule(
            init_value=args.lr_init_value,
            end_value=args.lr_peak_value,
            transition_steps=max(schedule_steps - 1, 1),
        )
    else:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=args.lr_init_value,
            peak_value=args.lr_peak_value,
            warmup_steps=warmup_steps,
            decay_steps=schedule_steps,
            end_value=args.lr_end_value,
        )

    inner_optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.98,
            eps=1e-7,
            weight_decay=args.weight_decay,
            mask=lambda params: jax.tree_util.tree_map(lambda value: value.ndim > 1, params),
        ),
    )
    tx = (
        optax.MultiSteps(inner_optimizer, every_k_schedule=accum_steps)
        if accum_steps > 1
        else inner_optimizer
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, lr_schedule


def create_checkpoint_manager(args: TrainingArguments):
    return ocp.CheckpointManager(
        os.path.abspath(args.checkpoint_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=args.save_total_limit,
            save_interval_steps=args.save_steps,
        ),
    )


def save_checkpoint(checkpoint_manager, model, optimizer, step: int):
    checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
        ),
    )


def to_jax_batch(batch):
    audios, labels, audio_lengths, label_lengths = batch
    return (
        jnp.asarray(audios, dtype=jnp.float32),
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray(audio_lengths, dtype=jnp.int32),
        jnp.asarray(label_lengths, dtype=jnp.int32),
    )


class PrefetchIterator:
    """Run host work and host-to-device copies in a background thread."""

    _END = object()

    def __init__(self, iterator, transform_fn, buffer_size: int):
        self.iterator = iter(iterator)
        self.transform_fn = transform_fn
        self.buffer_size = max(int(buffer_size), 1)
        self.queue = queue.Queue(maxsize=self.buffer_size)
        self.stop_event = threading.Event()
        self.worker_error = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                self.queue.put(self.transform_fn(item))
        except Exception as exc:  # pragma: no cover - surfaced in __next__
            self.worker_error = exc
        finally:
            self._put_end_marker()

    def _put_end_marker(self):
        while True:
            try:
                self.queue.put(self._END, timeout=0.1)
                return
            except queue.Full:
                if self.stop_event.is_set():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is self._END:
            if self.worker_error is not None:
                raise RuntimeError("Prefetch worker failed") from self.worker_error
            raise StopIteration
        return item

    def close(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=1.0)


def _start_async_host_copy(value):
    copy_fn = getattr(value, "copy_to_host_async", None)
    if copy_fn is not None:
        copy_fn()


def _flush_pending_losses(pending_losses, train_losses):
    if not pending_losses:
        return None
    loss_values = [float(loss) for loss in pending_losses]
    train_losses.extend(loss_values)
    pending_losses.clear()
    return loss_values[-1]


def ctc_loss(logits, output_lengths, labels, label_lengths, blank_id: int):
    logit_pad = (jnp.arange(logits.shape[1]) >= output_lengths[:, None]).astype(jnp.float32)
    label_pad = (jnp.arange(labels.shape[1]) >= label_lengths[:, None]).astype(jnp.float32)
    logits_f32 = logits.astype(jnp.float32)
    per_sample_loss = optax.ctc_loss(
        logits_f32,
        logit_pad,
        labels,
        label_pad,
        blank_id=blank_id,
    )
    return per_sample_loss.mean(), logits_f32, logit_pad


@jax.remat
def entropy_term(logits_f32, frame_mask):
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
    return (entropy * frame_mask).sum() / jnp.maximum(frame_mask.sum(), 1.0)


def build_step_functions(blank_id: int, entropy_weight: float):
    @nnx.jit
    def train_step(model, optimizer, audios, labels, audio_lengths, label_lengths, step):
        def loss_fn(model):
            logits, output_lengths = model(
                audios,
                training=True,
                inputs_lengths=audio_lengths,
                step=step,
            )
            ctc, logits_f32, logit_pad = ctc_loss(
                logits,
                output_lengths,
                labels,
                label_lengths,
                blank_id=blank_id,
            )
            entropy_bonus = entropy_term(logits_f32, 1.0 - logit_pad)
            return ctc - entropy_weight * entropy_bonus

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model=model, grads=grads)
        return loss

    @nnx.jit
    def eval_step(model, audios, labels, audio_lengths, label_lengths):
        logits, output_lengths = model(audios, training=False, inputs_lengths=audio_lengths)
        loss, _, _ = ctc_loss(logits, output_lengths, labels, label_lengths, blank_id=blank_id)
        return loss, logits, output_lengths

    return train_step, eval_step


class CsvLogger:
    """Tiny append-only CSV logger."""

    FIELDS = ["step", "epoch", "phase", "loss", "lr", "val_cer"]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._writer = None

    def __enter__(self):
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()
        return self

    def __exit__(self, *exc):
        self._fh.close()

    def log(self, **row):
        self._writer.writerow({field: row.get(field, "") for field in self.FIELDS})
        self._fh.flush()


def precompile_buckets(args, model, optimizer, train_step):
    if DEV_MODE:
        return

    print("Pre-compiling for all buckets...")
    for bucket_audio_length, bucket_label_length in tqdm(args.bucket_sizes, desc="compile"):
        dummy_audios = jnp.zeros((args.batch_size, bucket_audio_length), dtype=jnp.float32)
        dummy_labels = jnp.zeros((args.batch_size, bucket_label_length), dtype=jnp.int32)
        dummy_audio_lengths = jnp.full((args.batch_size,), bucket_audio_length, dtype=jnp.int32)
        dummy_label_lengths = jnp.full((args.batch_size,), bucket_label_length, dtype=jnp.int32)
        train_step.lower(
            model,
            optimizer,
            dummy_audios,
            dummy_labels,
            dummy_audio_lengths,
            dummy_label_lengths,
            jnp.int32(0),
        ).compile()
        print(
            "  compiled bucket:"
            f" audio={bucket_audio_length}  labels={bucket_label_length}"
        )


def run_validation(model, test_loader, tokenizer, eval_step):
    val_losses = []
    refs = []
    hyps = []

    for batch in tqdm(test_loader, desc="val"):
        audios, labels, audio_lengths, label_lengths = to_jax_batch(batch)
        loss, logits, output_lengths = eval_step(
            model,
            audios,
            labels,
            audio_lengths,
            label_lengths,
        )
        val_losses.append(float(loss))

        logits_np = np.asarray(logits)
        output_lengths_np = np.asarray(output_lengths)
        labels_np = np.asarray(labels)
        label_lengths_np = np.asarray(label_lengths)
        for index in range(logits_np.shape[0]):
            hyps.append(
                greedy_ctc_decode_text(
                    logits_np[index],
                    int(output_lengths_np[index]),
                    tokenizer,
                )
            )
            refs.append(
                decode_token_ids(
                    labels_np[index, : int(label_lengths_np[index])],
                    tokenizer,
                )
            )

    average_loss = mean(val_losses) if val_losses else 0.0
    average_cer = jiwer.cer(refs, hyps) if refs else 0.0
    return average_loss, average_cer


def main():
    configure_jax_cache()
    args = build_training_args()
    tokenizer = load_tokenizer(args)
    model = build_model(args, tokenizer)

    train_source, eval_source, steps_per_epoch = build_data_sources(
        args.data_dir,
        args.sampling_rate,
        args.batch_size,
    )
    total_steps = steps_per_epoch * args.num_epochs
    if total_steps <= 0:
        raise ValueError(
            "Training dataset produced zero optimization steps. "
            "Check the packed dataset, duration filters, and batch size."
        )
    print(
        f"train samples: {len(train_source)}  steps/epoch: {steps_per_epoch}  total: {total_steps}"
    )

    train_loader = build_train_loader(train_source, tokenizer, args, args.num_epochs)
    eval_loader = build_test_loader(eval_source, tokenizer, args)
    optimizer, lr_schedule = build_optimizer(args, model, total_steps)
    checkpoint_manager = create_checkpoint_manager(args)
    train_step, eval_step = build_step_functions(
        blank_id=int(tokenizer.blank_id),
        entropy_weight=float(args.entropy_weight),
    )

    precompile_buckets(args, model, optimizer, train_step)

    global_step = 0
    accum_steps = args.grad_accumulation_steps
    train_iter = PrefetchIterator(
        train_loader,
        to_jax_batch,
        buffer_size=args.device_prefetch_batches,
    )

    try:
        with CsvLogger("runs/train.csv") as logger:
            for epoch_index in range(args.num_epochs):
                train_losses = []
                pending_losses = []
                progress = tqdm(
                    range(steps_per_epoch),
                    desc=f"epoch {epoch_index + 1}/{args.num_epochs}",
                )

                for _ in progress:
                    audios, labels, audio_lengths, label_lengths = next(train_iter)
                    loss = train_step(
                        model,
                        optimizer,
                        audios,
                        labels,
                        audio_lengths,
                        label_lengths,
                        jnp.int32(global_step),
                    )
                    _start_async_host_copy(loss)
                    pending_losses.append(loss)
                    global_step += 1

                    should_log = global_step % args.log_steps == 0 or global_step < 10
                    should_save = global_step > 1 and checkpoint_manager.should_save(global_step)
                    should_sync = (
                        should_log
                        or should_save
                        or len(pending_losses) >= args.loss_sync_steps
                        or (DEV_MODE and global_step >= DEV_STEPS)
                    )
                    loss_value = None
                    if should_sync:
                        loss_value = _flush_pending_losses(pending_losses, train_losses)

                    if should_log:
                        if loss_value is None:
                            loss_value = _flush_pending_losses(pending_losses, train_losses)
                        lr_value = float(lr_schedule(global_step // accum_steps))
                        progress.set_postfix(
                            loss=f"{loss_value:.4f}",
                            lr=f"{lr_value:.2e}",
                            step=global_step,
                        )
                        logger.log(
                            step=global_step,
                            epoch=epoch_index + 1,
                            phase="train",
                            loss=loss_value,
                            lr=lr_value,
                        )

                    if should_save:
                        save_checkpoint(checkpoint_manager, model, optimizer, global_step)

                    if DEV_MODE and global_step >= DEV_STEPS:
                        break

                _flush_pending_losses(pending_losses, train_losses)

                print(f"\nValidation after epoch {epoch_index + 1}...")
                val_loss, val_cer = run_validation(model, eval_loader, tokenizer, eval_step)
                train_loss = mean(train_losses) if train_losses else 0.0
                print(
                    f"epoch {epoch_index + 1}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  val_cer={val_cer:.4f}\n"
                )
                logger.log(
                    step=global_step,
                    epoch=epoch_index + 1,
                    phase="val",
                    loss=val_loss,
                    val_cer=val_cer,
                )

                if DEV_MODE and global_step >= DEV_STEPS:
                    break
    finally:
        train_iter.close()

    if not checkpoint_manager.should_save(global_step):
        save_checkpoint(checkpoint_manager, model, optimizer, global_step)
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    print(f"Final checkpoint saved at step {global_step}")


if __name__ == "__main__":
    main()
