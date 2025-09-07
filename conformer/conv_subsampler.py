import flax.nnx as nnx
import jax.numpy as jnp
import jax


class ConvolutionSubsampling(nnx.Module):
    def __init__(self, output_dim: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=1, out_features=output_dim, kernel_size=(3, 3),
                            strides=(2, 2), dtype=jnp.float16, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=output_dim, out_features=output_dim, kernel_size=(3, 3),
                            strides=(2, 2), dtype=jnp.float16, rngs=rngs)
        # D * F/4 (F = mel_bins 80, hence 20)
        self.linear = nnx.Linear(in_features=20 * output_dim, out_features=output_dim,
                                  dtype=jnp.float16, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool):
        x = jnp.expand_dims(x, axis=-1)
        x = nnx.relu(self.conv1(x)) # (B, T/2, F/2, D)
        x = nnx.relu(self.conv2(x)) # (B, T/4, F/4, D)

        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, -1) # (B, T/4, F/4 * D)

        x = self.linear(x)
        x = self.dropout(x, deterministic=not training)

        return x