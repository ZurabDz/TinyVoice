import flax.nnx as nnx
import jax.numpy as jnp
import jax


class Swish(nnx.Module):
    """Swish activation function"""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.nn.sigmoid(x)
