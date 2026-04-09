import os

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
