import optax
import jax.numpy as jnp
import jax

logits = jnp.zeros((1, 10, 5))
labels = jnp.array([[1, 2, 0, 0, 0]], dtype=jnp.int32)
logit_paddings = jnp.array([[False]*5 + [True]*5])
label_paddings = jnp.array([[False, False, True, True, True]])

loss = optax.ctc_loss(logits, logit_paddings, labels, label_paddings)
print(f"Loss with zero logits: {loss}")

log_probs = jax.nn.log_softmax(logits, axis=-1)
loss_log_probs = optax.ctc_loss(log_probs, logit_paddings, labels, label_paddings)
print(f"Loss with log_probs: {loss_log_probs}")
