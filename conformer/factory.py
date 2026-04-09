import os

import orbax.checkpoint as ocp
from flax import nnx

from conformer.config import TrainingArguments
from conformer.model import FastConformerEncoder


def build_model(args: TrainingArguments, tokenizer) -> FastConformerEncoder:
    """Construct a FastConformerEncoder from config + tokenizer vocab size."""
    return FastConformerEncoder(
        token_count=tokenizer.vocab_size,
        num_layers=args.num_encoder_layers,
        d_model=args.d_model,
        num_heads=args.num_attention_heads,
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


def load_checkpoint(model, checkpoint_dir: str):
    """Restore the latest checkpoint into `model`. Returns (model, step or None)."""
    mngr = ocp.CheckpointManager(os.path.abspath(checkpoint_dir))
    latest = mngr.latest_step()
    if latest is None:
        mngr.close()
        return model, None
    restored = mngr.restore(
        latest,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
    )
    nnx.update(model, restored.model)
    mngr.close()
    return model, latest
