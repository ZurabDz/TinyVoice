from .conformer_block import ConformerEncoder
import jax.numpy as jnp
import optax
from flax import nnx


def create_padding_mask(lengths: jnp.ndarray, max_len: int) -> jnp.ndarray:
    batch_size = lengths.shape[0]
    indices = jnp.arange(max_len).reshape(1, -1)
    mask = indices >= lengths.reshape(-1, 1)
    return mask.astype(jnp.float32)


def create_learning_rate_fn(warmup_steps: int, base_lr: float):
    def lr_fn(step):
        # No warmup if warmup_steps is 0 - use constant learning rate
        if warmup_steps == 0:
            return base_lr
        # Otherwise, linear warmup then constant
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        return base_lr * warmup_factor

    return lr_fn


def loss_fn(model: ConformerEncoder, batch, training):
    log_probs, output_lengths = model(
        batch["inputs"], batch["input_lengths"], training=training
    )

    max_logit_len = log_probs.shape[1]
    max_label_len = batch["labels"].shape[1]
    logit_paddings = create_padding_mask(output_lengths, max_logit_len)
    label_paddings = create_padding_mask(batch["label_lengths"], max_label_len)

    loss = optax.ctc_loss(
        log_probs, logit_paddings, batch["labels"], label_paddings, blank_id=0
    ).mean()
    return loss


# @nnx.jit(donate_argnums=0)
# @nnx.jit
# def train_step(model: ConformerEncoder, optimizer: nnx.Optimizer, batch: dict):
#     loss, grads = nnx.value_and_grad(loss_fn)(model, batch, training=True)
#     optimizer.update(model=model, grads=grads)
#     return loss

@nnx.jit
def train_step(graphdef, state, batch):
    model, optimizer = nnx.merge(graphdef, state)
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, training=True)
    optimizer.update(model=model, grads=grads)
    state = nnx.state((model, optimizer))
    return loss, state

@nnx.jit
def eval_step(graphdef, state, batch):
    model, _ = nnx.merge(graphdef, state)
    return loss_fn(model, batch, training=False)
