import os

import jax
import orbax.checkpoint as ocp
from flax import nnx

from conformer.config import TrainingArguments
from conformer.model import FastConformerEncoder


def build_model(args: TrainingArguments, tokenizer) -> FastConformerEncoder:
    """Construct and weight-initialize a FastConformerEncoder from config."""
    token_count = (
        tokenizer.vocab_size
        if hasattr(tokenizer, "vocab_size")
        else len(tokenizer.id_to_char)
    )
    model = FastConformerEncoder(
        token_count=token_count,
        num_layers=args.num_encoder_layers,
        d_model=args.d_model,
        num_head=args.num_attention_heads,
        dropout=args.feed_forward_dropout_p,
        feed_forward_expansion_factor=args.feed_forward_expansion_factor,
        conv_kernel_size=args.conv_kernel_size,
        layer_drop_prob=args.layer_drop_prob,
        layer_drop_anneal_steps=args.layer_drop_anneal_steps,
        d_input=args.n_mels,
        sample_rate=args.sampling_rate,
        n_fft=args.n_fft,
        n_window_size=args.win_length,
        n_window_stride=args.hop_length,
        dtype=args.dtype,
        rngs=nnx.Rngs(0),
    )
    model.initialize_weights(jax.random.PRNGKey(0))
    return model


def load_checkpoint(model, checkpoint_dir: str):
    """Restore latest checkpoint into model. Returns (model, step) or (model, None)."""
    mngr = ocp.CheckpointManager(os.path.abspath(checkpoint_dir))
    latest_step = mngr.latest_step()
    if latest_step is None:
        mngr.close()
        return model, None
    restored = mngr.restore(
        latest_step,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
    )
    nnx.update(model, restored.model)
    mngr.close()
    return model, latest_step
