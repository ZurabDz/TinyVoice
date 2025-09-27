import flax.nnx as nnx
from .activation import Swish
import jax.numpy as jnp


class ConvolutionModule(nnx.Module):
    """ Convolution module from the Conformer paper. """
    def __init__(self, in_dim: int, kernel_size: int, expansion_factor: int, dropout_p: float, *, dtype, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(in_dim, dtype=jnp.bfloat16, rngs=rngs)
        self.pointwise_conv1 = nnx.Conv(in_dim, in_dim * expansion_factor, kernel_size=(1,),
                                        dtype=dtype, rngs=rngs)
        self.depthwise_conv = nnx.Conv(in_dim, in_dim, kernel_size=(kernel_size,), padding='SAME',
                                        feature_group_count=in_dim, dtype=dtype, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(in_dim, dtype=jnp.bfloat16, momentum=0.9, epsilon=1e-5, rngs=rngs)
        self.swish = Swish()
        self.pointwise_conv2 = nnx.Conv(in_dim, in_dim, kernel_size=(1,), dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_p, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, training: bool) -> jnp.ndarray:
        x_norm = self.layer_norm(x)
        x_conv = self.pointwise_conv1(x_norm)
        x_glu = nnx.glu(x_conv, axis=-1)
        x_depthwise = self.depthwise_conv(x_glu)
        x_bn = self.batch_norm(x_depthwise, use_running_average=not training)
        x_swish = self.swish(x_bn)
        x_pointwise = self.pointwise_conv2(x_swish)
        x_drop = self.dropout(x_pointwise, deterministic=not training)
        return x + x_drop