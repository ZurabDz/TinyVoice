"""Minimal end-to-end CTC training for the FastConformer ASR encoder."""

import argparse
import csv
import itertools
import os
from pathlib import Path
from statistics import mean

# These are OFF by default; they enable cuDNN-accelerated kernels on Ampere+ GPUs.
# Ignored on TPU/CPU (XLA only reads gpu flags on the GPU backend).
# os.environ.setdefault("XLA_FLAGS", (
#     "--xla_gpu_enable_cudnn_layer_norm=true "
#     "--xla_gpu_use_runtime_fusion=true "
#     "--xla_gpu_cudnn_gemm_fusion_level=2 "
#     "--xla_gpu_fused_attention_use_cudnn_rng=true"
# ))

import jax
import jax.numpy as jnp
import jiwer
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm.auto import tqdm

from conformer.config import TrainingArguments
from conformer.dataset import build_data_sources, build_eval_loader, build_train_loader
from conformer.decode import decode_token_ids, greedy_ctc_decode_text
from conformer.factory import build_model
from conformer.tokenizer import Tokenizer

DEV_MODE = os.environ.get("TINYVOICE_DEV", "0") == "1"
DEV_STEPS = int(os.environ.get("TINYVOICE_DEV_STEPS", "10000"))


def parse_args():
    parser = argparse.ArgumentParser(description="TinyVoice FastConformer CTC training")

    # Model
    m = parser.add_argument_group("model")
    m.add_argument("--d-model", type=int)
    m.add_argument("--num-encoder-layers", type=int)
    m.add_argument("--num-attention-heads", type=int)
    m.add_argument("--feed-forward-expansion-factor", type=int)
    m.add_argument("--dropout", type=float)
    m.add_argument("--conv-kernel-size", type=int)

    # Frontend
    f = parser.add_argument_group("frontend")
    f.add_argument("--sampling-rate", type=int)
    f.add_argument("--n-fft", type=int)
    f.add_argument("--win-length", type=int)
    f.add_argument("--hop-length", type=int)
    f.add_argument("--n-mels", type=int)

    # Optimizer
    o = parser.add_argument_group("optimizer")
    o.add_argument("--grad-clip", type=float)
    o.add_argument("--weight-decay", type=float)
    o.add_argument("--lr-init-value", type=float)
    o.add_argument("--lr-peak-value", type=float)
    o.add_argument("--lr-warmup-steps", type=int)
    o.add_argument("--lr-end-value", type=float)

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--dtype", type=str, choices=["bfloat16", "float32", "float16"])
    t.add_argument("--num-epochs", type=int)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--log-steps", type=int)
    t.add_argument("--save-steps", type=int)
    t.add_argument("--save-total-limit", type=int)
    t.add_argument("--checkpoint-dir", type=str)

    # Data
    d = parser.add_argument_group("data")
    d.add_argument("--data-dir", type=str)
    d.add_argument("--audio-frames-max", type=int)
    d.add_argument("--label-length-max", type=int)
    d.add_argument("--min-audio-seconds", type=float)
    d.add_argument("--max-audio-seconds", type=float)
    d.add_argument("--enable-speed-perturb", action=argparse.BooleanOptionalAction)
    d.add_argument("--enable-additive-noise", action=argparse.BooleanOptionalAction)
    d.add_argument("--enable-reverb", action=argparse.BooleanOptionalAction)
    d.add_argument("--mp-prefetch-workers", type=int)
    d.add_argument("--mp-prefetch-buffer", type=int)

    return parser.parse_args()


def build_training_args(cli) -> TrainingArguments:
    dtype_map = {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "float16": jnp.float16}
    overrides = {}
    for key, value in vars(cli).items():
        if value is not None:
            overrides[key] = dtype_map[value] if key == "dtype" else value
    return TrainingArguments(**overrides)


def configure_jax_cache():
    cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))


def build_optimizer(args: TrainingArguments, model, total_steps: int):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr_init_value,
        peak_value=args.lr_peak_value,
        warmup_steps=min(args.lr_warmup_steps, max(total_steps - 1, 1)),
        decay_steps=total_steps,
        end_value=args.lr_end_value,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.98,
            eps=1e-7,
            weight_decay=args.weight_decay,
            mask=lambda params: jax.tree_util.tree_map(lambda v: v.ndim > 1, params),
        ),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param), schedule


def ctc_loss(logits, output_lengths, labels, label_lengths, blank_id: int):
    logit_pad = (jnp.arange(logits.shape[1]) >= output_lengths[:, None]).astype(jnp.float32)
    label_pad = (jnp.arange(labels.shape[1]) >= label_lengths[:, None]).astype(jnp.float32)
    loss = optax.ctc_loss(
        logits.astype(jnp.float32), logit_pad, labels, label_pad, blank_id=blank_id
    )
    return jnp.where(label_lengths > 0, loss / label_lengths, 0.0).mean()

class Trainer(nnx.Module):
    def __init__(self, model, optimizer, ema_model, blank_id: int, schedule):
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.blank_id = blank_id
        self.schedule = schedule
        self.step = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    @nnx.jit
    def train_step(self, audios, labels, audio_lengths, label_lengths):
        def loss_fn(m):
            logits, output_lengths = m(audios, audio_lengths, training=True)
            return ctc_loss(logits, output_lengths, labels, label_lengths, self.blank_id)

        loss, grads = nnx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)

        self.step[...] += 1
        step = self.step[...]

        # EMA update
        decay = jnp.minimum(0.999, (1.0 + step) / (10.0 + step))
        ema_params = nnx.state(self.ema_model, nnx.Param)
        model_params = nnx.state(self.model, nnx.Param)
        new_ema_params = jax.tree_util.tree_map(
            lambda e, m: e * decay + m * (1.0 - decay),
            ema_params,
            model_params,
        )
        nnx.update(self.ema_model, new_ema_params)

        return loss


@nnx.jit(static_argnames="blank_id")
def eval_step(model, audios, labels, audio_lengths, label_lengths, blank_id: int):
    logits, output_lengths = model(audios, audio_lengths, training=False)
    loss = ctc_loss(logits, output_lengths, labels, label_lengths, blank_id)
    return loss, logits, output_lengths


def run_validation(model, eval_loader, tokenizer, blank_id: int, max_batches=None):
    losses, refs, hyps = [], [], []
    for index, (audios, labels, audio_lengths, label_lengths) in enumerate(
        tqdm(eval_loader, desc="val", leave=False)
    ):
        if max_batches is not None and index >= max_batches:
            break
        loss, logits, output_lengths = eval_step(
            model, audios, labels, audio_lengths, label_lengths, blank_id
        )
        losses.append(float(loss))
        logits_np = np.asarray(logits)
        out_lens = np.asarray(output_lengths)
        labels_np = np.asarray(labels)
        label_lens = np.asarray(label_lengths)
        for i in range(logits_np.shape[0]):
            hyps.append(greedy_ctc_decode_text(logits_np[i], int(out_lens[i]), tokenizer))
            refs.append(decode_token_ids(labels_np[i, : int(label_lens[i])], tokenizer))
    return (mean(losses) if losses else 0.0, jiwer.cer(refs, hyps) if refs else 0.0)


class CsvLogger:
    FIELDS = ("step", "epoch", "phase", "loss", "lr", "val_cer")

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
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDS})

    def flush(self):
        self._fh.flush()


def save_checkpoint(manager, trainer, step):
    manager.save(
        step,
        args=ocp.args.Composite(
            trainer=ocp.args.StandardSave(nnx.state(trainer)),
        ),
    )


def restore_checkpoint(manager, trainer):
    latest = manager.latest_step()
    if latest is not None:
        restored = manager.restore(
            latest,
            args=ocp.args.Composite(
                trainer=ocp.args.StandardRestore(nnx.state(trainer)),
            ),
        )
        nnx.update(trainer, restored.trainer)
        return latest
    return 0


def main():
    configure_jax_cache()
    args = build_training_args(parse_args())
    if DEV_MODE:
        args.enable_speed_perturb = False
        args.enable_additive_noise = False
        args.enable_reverb = False
        print(f"DEV MODE: stopping at {DEV_STEPS} steps, augments off")
    print(f"dtype={args.dtype}  shape=({args.audio_frames_max},{args.label_length_max})")

    tokenizer = Tokenizer.load_tokenizer(
        Path(args.data_dir) / "tokenizer.pkl"
    )
    model = build_model(args, tokenizer)
    ema_model = build_model(args, tokenizer)
    nnx.update(ema_model, nnx.state(model))

    train_source, eval_source, steps_per_epoch = build_data_sources(args)
    total_steps = steps_per_epoch * args.num_epochs
    if total_steps <= 0:
        raise ValueError("Training dataset produced zero steps. Check filters and batch size.")
    print(f"train: {len(train_source)} samples  steps/epoch: {steps_per_epoch}  total: {total_steps}")

    train_loader = build_train_loader(train_source, tokenizer, args, args.num_epochs)
    eval_loader = build_eval_loader(eval_source, tokenizer, args)
    optimizer, schedule = build_optimizer(args, model, total_steps)

    trainer = Trainer(model, optimizer, ema_model, int(tokenizer.blank_id), schedule)

    manager = ocp.CheckpointManager(
        os.path.abspath(args.checkpoint_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=args.save_total_limit,
            save_interval_steps=args.save_steps,
        ),
    )

    restored_step = restore_checkpoint(manager, trainer)
    if restored_step > 0:
        print(f"Restored from checkpoint at step {restored_step}")
    
    # Calculate starting epoch and remaining steps if needed
    # But with Grain repeated dataset, we can just skip if we wanted to.
    # For simplicity in this minimal script, we start from the current global step.
    
    train_loader_iter = iter(train_loader)
    # If we restored, we might want to skip batches, but Grain MapDataset is better for that.
    # Here we just continue.

    global_step = int(trainer.step[...])

    with CsvLogger("runs/train.csv") as logger:
        for epoch in range(global_step // steps_per_epoch, args.num_epochs):
            train_losses = []
            
            # Slice the iterator for the current epoch
            remaining_steps = steps_per_epoch - (global_step % steps_per_epoch)
            epoch_iter = itertools.islice(train_loader_iter, remaining_steps)
            
            progress = tqdm(
                epoch_iter, 
                total=steps_per_epoch, 
                desc=f"epoch {epoch + 1}/{args.num_epochs}",
                initial=global_step % steps_per_epoch
            )

            for audios, labels, audio_lengths, label_lengths in progress:
                loss = trainer.train_step(audios, labels, audio_lengths, label_lengths)
                global_step = int(trainer.step[...])
                
                if global_step % args.log_steps == 0 or global_step < 10:
                    loss_val = float(loss)
                    train_losses.append(loss_val)
                    lr_val = float(schedule(global_step))
                    progress.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}", step=global_step)
                    logger.log(step=global_step, epoch=epoch + 1, phase="train", loss=loss_val, lr=lr_val)

                if global_step > 0 and manager.should_save(global_step):
                    save_checkpoint(manager, trainer, global_step)
                    logger.flush()

                if DEV_MODE and global_step >= DEV_STEPS:
                    break

            # Run validation at the end of epoch
            print(f"\nValidation after epoch {epoch + 1}...")
            val_loss, val_cer = run_validation(
                trainer.ema_model, eval_loader, tokenizer, trainer.blank_id,
                max_batches=10 if DEV_MODE else None,
            )
            train_loss = mean(train_losses) if train_losses else 0.0
            print(f"epoch {epoch + 1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_cer={val_cer:.4f}\n")
            logger.log(step=global_step, epoch=epoch + 1, phase="val", loss=val_loss, val_cer=val_cer)
            logger.flush()

            if DEV_MODE and global_step >= DEV_STEPS:
                break

    if not manager.should_save(global_step):
        save_checkpoint(manager, trainer, global_step)
    manager.wait_until_finished()
    manager.close()
    print(f"Final checkpoint saved at step {global_step}")


if __name__ == "__main__":
    main()
