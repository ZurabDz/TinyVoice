import jax
import jax.numpy as jnp
import optax
from flax import nnx

SCALE_GROWTH_INTERVAL = 2000


@jax.jit(static_argnums=(0, 2), donate_argnums=(1, 3))
def train_step(
    model_graphdef,
    model_state,
    optimizer_graphdef,
    optimizer_state,
    loss_scale,
    scale_fin_steps,
    padded_audios,
    padded_labels,
    frames,
    label_lengths,
    step,
    blank_id,
    entropy_weight,
):
    """Training step using Functional API"""
    model = nnx.merge(model_graphdef, model_state)
    optimizer = nnx.merge(optimizer_graphdef, optimizer_state)

    def loss_fn(model):
        logits, real_times = model(
            padded_audios, training=True, inputs_lengths=frames, step=step
        )

        logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(
            jnp.float32
        )
        label_paddings = (
            jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]
        ).astype(jnp.float32)

        logits_f32 = logits.astype(jnp.float32)
        per_sample_loss = optax.ctc_loss(
            logits_f32,
            logit_paddings,
            padded_labels,
            label_paddings,
            blank_id=blank_id,
        )
        is_finite = jnp.isfinite(per_sample_loss)
        finite_loss_ratio = is_finite.mean()

        # Zero out infinite CTC losses (e.g. from impossible alignments)
        per_sample_loss = jnp.where(is_finite, per_sample_loss, 0.0)
        ctc_loss = per_sample_loss.mean()

        # Entropy regularization: prevent over-confident peaky CTC distributions
        # Wrapped in remat to avoid storing softmax activations during backprop
        frame_mask = 1.0 - logit_paddings  # 1 for valid, 0 for padding

        @jax.remat
        def compute_entropy(logits, mask):
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            probs = jnp.exp(log_probs)
            ent = -jnp.sum(probs * log_probs, axis=-1)  # (B, T)
            return (ent * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        masked_entropy = compute_entropy(logits_f32, frame_mask)

        loss = ctc_loss - entropy_weight * masked_entropy
        return loss, finite_loss_ratio

    def scaled_loss_fn(model):
        loss, aux = loss_fn(model)
        return loss * loss_scale, aux

    (scaled_loss, finite_loss_ratio), grads = nnx.value_and_grad(
        scaled_loss_fn, has_aux=True
    )(model)
    loss = scaled_loss / loss_scale

    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    grad_finite = jnp.all(
        jnp.array([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)])
    )
    grads = jax.tree.map(lambda g: jnp.where(grad_finite, g, jnp.zeros_like(g)), grads)

    optimizer.update(model=model, grads=grads)

    _, new_model_state = nnx.split(model)
    _, new_optimizer_state = nnx.split(optimizer)

    # Restore old state if gradients were non-finite — prevents Adam moments
    # and weight decay from being applied on a zero-gradient step.
    final_model_state = jax.tree.map(
        lambda new, old: jnp.where(grad_finite, new, old),
        new_model_state,
        model_state,
    )
    final_optimizer_state = jax.tree.map(
        lambda new, old: jnp.where(grad_finite, new, old),
        new_optimizer_state,
        optimizer_state,
    )

    # Update loss scale inline (avoids separate JIT dispatch round-trip)
    def _on_finite(args):
        ls, sfs = args
        new_sfs = sfs + jnp.int32(1)
        new_ls, new_sfs = jax.lax.cond(
            new_sfs >= SCALE_GROWTH_INTERVAL,
            lambda: (
                jnp.minimum(ls * jnp.float32(2.0), jnp.float32(2**24)),
                jnp.int32(0),
            ),
            lambda: (ls, new_sfs),
        )
        return new_ls, new_sfs

    def _on_infinite(args):
        ls, _ = args
        return jnp.maximum(ls * jnp.float32(0.5), jnp.float32(1.0)), jnp.int32(0)

    new_loss_scale, new_scale_fin_steps = jax.lax.cond(
        grad_finite, _on_finite, _on_infinite, (loss_scale, scale_fin_steps)
    )

    return (
        loss,
        finite_loss_ratio,
        grad_finite,
        final_model_state,
        final_optimizer_state,
        new_loss_scale,
        new_scale_fin_steps,
    )


@jax.jit(static_argnums=(0,))
def eval_step(
    model_graphdef: nnx.GraphDef,
    model_state: nnx.State,
    padded_audios: jnp.ndarray,
    padded_labels: jnp.ndarray,
    frames: jnp.ndarray,
    label_lengths: jnp.ndarray,
    blank_id,
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

    logits_f32 = logits.astype(jnp.float32)
    per_sample_loss = optax.ctc_loss(
        logits_f32,
        logit_paddings,
        padded_labels,
        label_paddings,
        blank_id=blank_id,
    )
    is_finite = jnp.isfinite(per_sample_loss)
    finite_ratio = is_finite.mean()
    finite_loss = jnp.where(is_finite, per_sample_loss, 0.0).sum() / jnp.maximum(
        is_finite.sum(), 1
    )

    return finite_loss, finite_ratio, logits, real_times
