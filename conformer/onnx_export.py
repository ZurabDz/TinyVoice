"""Pure-JAX inference helpers used when exporting the NNX model to ONNX.

``FastConformerEncoder`` stores its encoder layers in a scanned NNX module.
That is efficient during training, but NNX's scan transform cannot currently
be traced by :mod:`jax2onnx`.  The functions here perform the same inference
math while reading the already-loaded NNX parameters, so they are safe to
trace and do not change the training/inference model implementation.
"""

import math

import jax
import jax.numpy as jnp

from .model import _apply_rope, _rope_table


def _rms_norm(x, scale, epsilon: float = 1e-6):
    """Match ``nnx.RMSNorm`` for this model's configuration."""
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + epsilon) * scale


def _linear(x, kernel):
    """Match the bias-free ``nnx.Linear`` layers used by the encoder blocks."""
    return jnp.matmul(x, kernel)


def _depthwise_conv_1d(x, kernel):
    """Match NNX's ``NWC`` depthwise convolution with ``SAME`` padding."""
    channels = x.shape[-1]
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=channels,
    )


def _frontend_for_onnx(frontend, audio, audio_lengths):
    """Real-valued implementation of the log-mel frontend.

    ``jax.scipy.signal.stft`` traces through complex-valued FFT operations,
    which are not interoperable between current jax2onnx and ONNX Runtime.
    This computes the same one-sided STFT with real DFT matrices instead.
    """
    window_size = frontend.n_window_size
    hop_size = frontend.n_window_stride
    n_fft = frontend.n_fft
    padding = window_size // 2
    # This is the right-padding used by scipy/JAX STFT after boundary padding.
    end_padding = (-(audio.shape[1] % hop_size)) % window_size
    padded_audio = jnp.pad(audio, ((0, 0), (padding, padding + end_padding)))

    num_frames = 1 + (padded_audio.shape[1] - window_size) // hop_size
    starts = jnp.arange(num_frames)[:, None] * hop_size
    samples = starts + jnp.arange(window_size)[None, :]
    frames = padded_audio[:, samples]

    # scipy.signal.get_window("hann", N, fftbins=True), used by jax.scipy.stft.
    window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * jnp.arange(window_size) / window_size)
    frames = frames * window[None, None, :]

    frequencies = jnp.arange(n_fft // 2 + 1)
    positions = jnp.arange(window_size)
    angle = 2.0 * jnp.pi * positions[:, None] * frequencies[None, :] / n_fft
    real = jnp.einsum("bts,sf->btf", frames, jnp.cos(angle))
    imag = jnp.einsum("bts,sf->btf", frames, -jnp.sin(angle))
    power = (jnp.square(real) + jnp.square(imag)) / jnp.square(jnp.sum(window))

    filterbank = frontend.filterbank.value
    mel = jnp.matmul(power, filterbank.T).transpose(0, 2, 1)
    mel = jnp.log(mel + 2.0**-24)

    spec_lengths = frontend.output_length(audio_lengths)
    valid = (jnp.arange(mel.shape[-1]) < spec_lengths[:, None])[:, None, :]
    count = spec_lengths.astype(mel.dtype)[:, None, None]
    masked = jnp.where(valid, mel, 0.0)
    mean = jnp.sum(masked, axis=-1, keepdims=True) / count
    variance = jnp.sum(jnp.where(valid, jnp.square(mel - mean), 0.0), axis=-1, keepdims=True) / count
    mel = (mel - mean) / (jnp.sqrt(variance) + 1e-5)
    mel = jnp.where(valid, mel, 0.0)
    return mel, spec_lengths


def _ffn(x, block, layer: int):
    h = _rms_norm(x, block.norm.scale.value[layer])
    h = jax.nn.silu(_linear(h, block.gate.kernel.value[layer])) * _linear(
        h, block.up.kernel.value[layer]
    )
    return _linear(h, block.down.kernel.value[layer])


def _attention(x, block, layer: int, cos, sin, lengths):
    h = _rms_norm(x, block.norm.scale.value[layer])
    batch, time, _ = h.shape
    qkv = _linear(h, block.qkv.kernel.value[layer]).reshape(
        batch, time, 3, block.num_heads, block.head_dim
    )
    q = _apply_rope(qkv[:, :, 0], cos, sin)
    k = _apply_rope(qkv[:, :, 1], cos, sin)
    v = qkv[:, :, 2]
    h = jax.nn.dot_product_attention(
        q,
        k,
        v,
        query_seq_lengths=lengths,
        key_value_seq_lengths=lengths,
        implementation=None,
    )
    return _linear(h.reshape(batch, time, -1), block.out.kernel.value[layer])


def _conv_module(x, block, layer: int, mask1d):
    h = _rms_norm(x, block.norm.scale.value[layer])
    h = _linear(h, block.pw1.kernel.value[layer])
    value, gate = jnp.split(h, 2, axis=-1)
    h = value * jax.nn.sigmoid(gate)
    h = h * mask1d
    h = _depthwise_conv_1d(h, block.dw.kernel.value[layer])
    h = jax.nn.silu(_rms_norm(h, block.act_norm.scale.value[layer]))
    return _linear(h, block.pw2.kernel.value[layer])


def forward_for_onnx(model, audio, audio_lengths):
    """Run ``model`` in inference mode without NNX scan transformations.

    The returned tensors have the same values and shapes as
    ``model(audio, audio_lengths, training=False)``.  The fixed number of
    encoder layers is unrolled while tracing, which makes the ONNX graph
    portable across ONNX Runtime providers.
    """
    mel, mel_lengths = _frontend_for_onnx(model.frontend, audio, audio_lengths)
    x = jnp.transpose(mel, (0, 2, 1))
    seq_len = model.subsampler.output_length(mel_lengths)
    x = model.subsampler(x)
    x = model.proj(x) * math.sqrt(model.d_model)

    time = x.shape[1]
    cos, sin = _rope_table(model.head_dim, time, model.dtype)
    mask1d = (jnp.arange(time)[None, :] < seq_len[:, None])[:, :, None].astype(x.dtype)

    for layer in range(model.blocks.ff1.norm.scale.value.shape[0]):
        x = x + 0.5 * _ffn(x, model.blocks.ff1, layer)
        x = x + _attention(x, model.blocks.attn, layer, cos, sin, seq_len)
        x = x + _conv_module(x, model.blocks.conv, layer, mask1d)
        x = x + 0.5 * _ffn(x, model.blocks.ff2, layer)

    x = _rms_norm(x, model.final_norm.scale.value)
    return model.head(x), seq_len
