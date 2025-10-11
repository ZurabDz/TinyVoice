import math
from flax import nnx
import jax.numpy as jnp

INF_VAL = 10000.0


class PositionalEncoding(nnx.Module):
    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        dropout_rate_emb: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        if dropout_rate_emb > 0:
            self.dropout_emb = nnx.Dropout(rate=dropout_rate_emb, rngs=rngs)
        else:
            self.dropout_emb = None

        self.pe = nnx.Variable(jnp.zeros((1, 1, d_model)))
        positions = jnp.arange(0, self.max_len)[:, None]
        self.pe.value = self.create_pe(positions, jnp.bfloat16)

    def create_pe(self, positions: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
        pos_length = positions.shape[0]
        pe = jnp.zeros((pos_length, self.d_model))

        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2, dtype=jnp.float32)
            * -(math.log(INF_VAL) / self.d_model)
        )

        pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))

        pe = jnp.expand_dims(pe, 0).astype(dtype)
        return pe

    def extend_pe(self, length: int, dtype: jnp.dtype):
        if self.pe.value.shape[1] >= length:
            return

        positions = jnp.arange(0, length, dtype=jnp.float32)[:, None]
        self.pe.value = self.create_pe(positions=positions, dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        input_len = x.shape[1]

        # self.extend_pe(input_len, x.dtype)

        pos_emb = self.pe.value[:, :input_len]

        if self.dropout_emb is not None:
            pos_emb = self.dropout_emb(pos_emb)

        x = x + pos_emb
        return self.dropout(x), pos_emb
