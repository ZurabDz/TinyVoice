import flax.nnx as nnx
from .activation import Swish
import jax.numpy as jnp


class FeedForwardModule(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        expansion_factor: int,
        dropout_p: float,
        *,
        dtype,
        rngs: nnx.Rngs,
    ):
        expanded_dim = in_dim * expansion_factor
        self.layer_norm = nnx.LayerNorm(in_dim, rngs=rngs, dtype=jnp.bfloat16)
        self.linear1 = nnx.Linear(in_dim, expanded_dim, dtype=dtype, rngs=rngs)
        self.swish = Swish()
        self.dropout1 = nnx.Dropout(dropout_p, rngs=rngs)
        self.linear2 = nnx.Linear(expanded_dim, in_dim, dtype=dtype, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, training: bool) -> jnp.ndarray:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x, deterministic=not training)
        x = self.linear2(x)
        x = self.dropout2(x, deterministic=not training)
        return x
