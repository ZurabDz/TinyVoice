import os

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from conformer.config import TrainingArguments
from conformer.model import FastConformerEncoder


def build_model(args: TrainingArguments, tokenizer) -> FastConformerEncoder:
    return FastConformerEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_encoder_layers,
        num_heads=args.num_attention_heads,
        expansion=args.feed_forward_expansion_factor,
        kernel=args.conv_kernel_size,
        dropout=args.dropout,
        n_mels=args.n_mels,
        sample_rate=args.sampling_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        dtype=args.dtype,
        rngs=nnx.Rngs(0),
    )


def load_checkpoint(model, checkpoint_dir: str, args: TrainingArguments = None, tokenizer=None):
    """Restore the latest checkpoint into `model`. Returns (model, step or None).

    Checkpoints saved by the current Trainer use a 'trainer' composite key
    containing model + ema_model + optimizer + step.  When *args* and
    *tokenizer* are provided we rebuild that structure so Orbax can match
    shapes, then copy the **EMA** weights into *model* (better for inference).
    """
    mngr = ocp.CheckpointManager(os.path.abspath(checkpoint_dir))
    latest = mngr.latest_step()
    if latest is None:
        mngr.close()
        return model, None

    if args is not None and tokenizer is not None:
        # Build a Trainer-shaped shell so Orbax has a matching state target.
        ema_model = build_model(args, tokenizer)

        # Optimizer — only the state *shapes* matter; schedule values are irrelevant.
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=args.lr_init_value,
            peak_value=args.lr_peak_value,
            warmup_steps=1,
            decay_steps=2,
            end_value=args.lr_end_value,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(
                learning_rate=schedule,
                b1=0.9, b2=0.98, eps=1e-7,
                weight_decay=args.weight_decay,
                mask=lambda params: jax.tree_util.tree_map(lambda v: v.ndim > 1, params),
            ),
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        class _TrainerShell(nnx.Module):
            """Matches the Trainer nnx-state layout (model, optimizer, ema_model, step)."""
            def __init__(self):
                self.model = model
                self.optimizer = optimizer
                self.ema_model = ema_model
                self.step = nnx.Variable(jnp.array(0, dtype=jnp.int32))

        shell = _TrainerShell()
        restored = mngr.restore(
            latest,
            args=ocp.args.Composite(
                trainer=ocp.args.StandardRestore(nnx.state(shell)),
            ),
        )
        nnx.update(shell, restored.trainer)
        # Copy EMA weights into the caller's model.
        nnx.update(model, nnx.state(shell.ema_model, nnx.Param))
    else:
        # Legacy format: checkpoint saved with a bare 'model' key.
        restored = mngr.restore(
            latest,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
        )
        nnx.update(model, restored.model)

    mngr.close()
    return model, latest
