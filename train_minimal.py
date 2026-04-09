"""Minimal end-to-end CTC training for the FastConformer ASR encoder."""

import csv
import os
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
from conformer.dataset import build_data_sources, build_eval_loader, build_train_loader
from conformer.decode import decode_token_ids, greedy_ctc_decode_text
from conformer.factory import build_model
from conformer.tokenizer import Tokenizer

DEV_MODE = os.environ.get("TINYVOICE_DEV", "0") == "1"
DEV_STEPS = int(os.environ.get("TINYVOICE_DEV_STEPS", "10000"))


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
    return optax.ctc_loss(
        logits.astype(jnp.float32), logit_pad, labels, label_pad, blank_id=blank_id
    ).mean()


def make_train_step(blank_id: int):
    @nnx.jit(donate_argnames=("model", "optimizer"))
    def train_step(model, optimizer, audios, labels, audio_lengths, label_lengths):
        def loss_fn(m):
            logits, output_lengths = m(audios, audio_lengths, training=True)
            return ctc_loss(logits, output_lengths, labels, label_lengths, blank_id)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_eval_step(blank_id: int):
    @nnx.jit
    def eval_step(model, audios, labels, audio_lengths, label_lengths):
        logits, output_lengths = model(audios, audio_lengths, training=False)
        loss = ctc_loss(logits, output_lengths, labels, label_lengths, blank_id)
        return loss, logits, output_lengths

    return eval_step


def run_validation(model, eval_loader, tokenizer, eval_step, max_batches=None):
    losses, refs, hyps = [], [], []
    for index, (audios, labels, audio_lengths, label_lengths) in enumerate(
        tqdm(eval_loader, desc="val")
    ):
        if max_batches is not None and index >= max_batches:
            break
        loss, logits, output_lengths = eval_step(
            model, audios, labels, audio_lengths, label_lengths
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


def save_checkpoint(manager, model, optimizer, step):
    manager.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
        ),
    )


def main():
    configure_jax_cache()
    args = TrainingArguments()
    if DEV_MODE:
        args.enable_speed_perturb = False
        args.enable_additive_noise = False
        args.enable_reverb = False
        print(f"DEV MODE: stopping at {DEV_STEPS} steps, augments off")
    print(f"dtype={args.dtype}  shape=({args.audio_frames_max},{args.label_length_max})")

    tokenizer = Tokenizer.load_tokenizer(
        Path(args.data_dir) / "packed_dataset" / "tokenizer.pkl"
    )
    model = build_model(args, tokenizer)

    train_source, eval_source, steps_per_epoch = build_data_sources(args)
    total_steps = steps_per_epoch * args.num_epochs
    if total_steps <= 0:
        raise ValueError("Training dataset produced zero steps. Check filters and batch size.")
    print(f"train: {len(train_source)} samples  steps/epoch: {steps_per_epoch}  total: {total_steps}")

    train_loader = build_train_loader(train_source, tokenizer, args, args.num_epochs)
    eval_loader = build_eval_loader(eval_source, tokenizer, args)
    optimizer, schedule = build_optimizer(args, model, total_steps)

    train_step = make_train_step(int(tokenizer.blank_id))
    eval_step = make_eval_step(int(tokenizer.blank_id))

    manager = ocp.CheckpointManager(
        os.path.abspath(args.checkpoint_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=args.save_total_limit,
            save_interval_steps=args.save_steps,
        ),
    )

    global_step = 0
    last_loss = None

    with CsvLogger("runs/train.csv") as logger:
        for epoch in range(args.num_epochs):
            train_losses = []
            progress = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch + 1}/{args.num_epochs}")

            for audios, labels, audio_lengths, label_lengths in progress:
                last_loss = train_step(model, optimizer, audios, labels, audio_lengths, label_lengths)
                global_step += 1

                if global_step % args.log_steps == 0 or global_step < 10:
                    loss_val = float(last_loss)
                    train_losses.append(loss_val)
                    lr_val = float(schedule(global_step))
                    progress.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}", step=global_step)
                    logger.log(step=global_step, epoch=epoch + 1, phase="train", loss=loss_val, lr=lr_val)

                if global_step > 1 and manager.should_save(global_step):
                    save_checkpoint(manager, model, optimizer, global_step)
                    logger.flush()

                if DEV_MODE and global_step >= DEV_STEPS:
                    break

            print(f"\nValidation after epoch {epoch + 1}...")
            val_loss, val_cer = run_validation(
                model, eval_loader, tokenizer, eval_step,
                max_batches=10 if DEV_MODE else None,
            )
            train_loss = mean(train_losses) if train_losses else 0.0
            print(f"epoch {epoch + 1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_cer={val_cer:.4f}\n")
            logger.log(step=global_step, epoch=epoch + 1, phase="val", loss=val_loss, val_cer=val_cer)
            logger.flush()

            if DEV_MODE and global_step >= DEV_STEPS:
                break

    if not manager.should_save(global_step):
        save_checkpoint(manager, model, optimizer, global_step)
    manager.wait_until_finished()
    manager.close()
    print(f"Final checkpoint saved at step {global_step}")


if __name__ == "__main__":
    main()
