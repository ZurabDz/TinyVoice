"""Build a fixed-shape ONNX encoder graph directly from the NNX parameters.

Why hand-build instead of tracing with ``jax2onnx``: the Allwinner/VeriSilicon
ACUITY importer accepts a specific, fairly narrow set of ONNX operators (no
``Einsum``, no fused attention, no dynamic shapes).  Tracing produces whatever
the tracer feels like emitting and then the conversion fails deep inside the
toolchain.  Emitting the graph by hand keeps the operator set pinned to ops the
importer definitely handles, and lets the RMS-norm pattern match ACUITY's
fusion rule exactly so it lowers to a native layer instead of seven.

The graph covers the encoder only -- log-mel in, CTC logits out.  See
``frontend_np`` for the CPU-side frontend and the reasoning behind the split.

Shapes are static, as the NPU requires.  Sequence length is fixed at export
time; a boolean frame mask is passed as a second input so shorter utterances
padded up to the window still produce the same logits as the JAX model.
"""

from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13


class GraphBuilder:
    """Minimal helper for emitting ONNX nodes with unique names."""

    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counts: dict[str, int] = {}

    def _name(self, prefix: str) -> str:
        index = self._counts.get(prefix, 0)
        self._counts[prefix] = index + 1
        return f"{prefix}_{index}"

    def op(self, op_type: str, inputs: list[str], *, out: str | None = None, **attrs) -> str:
        output = out or self._name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def const(self, array: np.ndarray, prefix: str = "const") -> str:
        """Emit a ``Constant`` node.

        ACUITY's fusion rules are written against ``Constant`` producers rather
        than graph initializers, so weights that participate in a fused pattern
        (RMS norm in particular) have to be materialised this way.
        """
        name = self._name(prefix)
        tensor = numpy_helper.from_array(np.ascontiguousarray(array), name + "_value")
        self.nodes.append(helper.make_node("Constant", [], [name], value=tensor))
        return name

    def init(self, array: np.ndarray, prefix: str = "w") -> str:
        """Register a graph initializer (used for large weight tensors)."""
        name = self._name(prefix)
        self.initializers.append(
            numpy_helper.from_array(np.ascontiguousarray(array), name)
        )
        return name


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def rms_norm(g: GraphBuilder, x: str, scale: np.ndarray, eps: float = 1e-6) -> str:
    """Emit RMS norm in the exact node order ACUITY's ``r_rmsnormalize`` matches.

    The rule is a literal subgraph match on
    ``Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Mul``, including which
    operand slot each constant lands in.  Deviating from it still produces a
    correct graph, just seven separate layers that quantise worse.
    """
    two = g.const(_f32(2.0), "rms_pow")
    eps_c = g.const(_f32(eps), "rms_eps")
    one = g.const(_f32(1.0), "rms_one")
    scale_c = g.const(_f32(scale), "rms_scale")

    squared = g.op("Pow", [x, two])
    mean = g.op("ReduceMean", [squared], axes=[-1], keepdims=1)
    shifted = g.op("Add", [mean, eps_c])
    root = g.op("Sqrt", [shifted])
    inv = g.op("Div", [one, root])
    scaled = g.op("Mul", [x, inv])
    return g.op("Mul", [scaled, scale_c])


def silu(g: GraphBuilder, x: str) -> str:
    """``x * sigmoid(x)``; ACUITY folds this into its native swish layer."""
    return g.op("Mul", [x, g.op("Sigmoid", [x])])


def linear(g: GraphBuilder, x: str, kernel: np.ndarray, bias: np.ndarray | None = None) -> str:
    """Bias-optional dense layer over the last axis (NNX kernels are ``(in, out)``)."""
    out = g.op("MatMul", [x, g.init(_f32(kernel), "kernel")])
    if bias is not None:
        out = g.op("Add", [out, g.init(_f32(bias), "bias")])
    return out


def slice_last(g: GraphBuilder, x: str, start: int, end: int, rank: int) -> str:
    starts = g.init(np.array([start], dtype=np.int64), "slice_start")
    ends = g.init(np.array([end], dtype=np.int64), "slice_end")
    axes = g.init(np.array([rank - 1], dtype=np.int64), "slice_axis")
    return g.op("Slice", [x, starts, ends, axes])


def reshape(g: GraphBuilder, x: str, shape: tuple[int, ...]) -> str:
    return g.op("Reshape", [x, g.init(np.array(shape, dtype=np.int64), "shape")])


def rope_tables(head_dim: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine/sine tables shaped ``(1, 1, T, head_dim // 2)`` for ``(B, H, T, D)`` input.

    Mirrors ``model._rope_table`` followed by the ``[..., :half]`` slice that
    ``model._apply_rope`` performs, so only the first half is ever needed.
    """
    inv = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = np.outer(np.arange(seq_len, dtype=np.float64), inv)
    cos = np.cos(freqs)[None, None].astype(np.float32)
    sin = np.sin(freqs)[None, None].astype(np.float32)
    return cos, sin


def apply_rope(g: GraphBuilder, x: str, cos: str, sin: str, head_dim: int) -> str:
    """Rotary embedding on a ``(B, H, T, D)`` tensor."""
    half = head_dim // 2
    x1 = slice_last(g, x, 0, half, rank=4)
    x2 = slice_last(g, x, half, head_dim, rank=4)

    left = g.op("Sub", [g.op("Mul", [x1, cos]), g.op("Mul", [x2, sin])])
    right = g.op("Add", [g.op("Mul", [x1, sin]), g.op("Mul", [x2, cos])])
    return g.op("Concat", [left, right], axis=-1)


def _ffn(g: GraphBuilder, x: str, block, layer: int) -> str:
    h = rms_norm(g, x, block.norm.scale.value[layer])
    gate = silu(g, linear(g, h, block.gate.kernel.value[layer]))
    up = linear(g, h, block.up.kernel.value[layer])
    return linear(g, g.op("Mul", [gate, up]), block.down.kernel.value[layer])


def _attention(
    g: GraphBuilder,
    x: str,
    block,
    layer: int,
    *,
    cos: str,
    sin: str,
    mask_scores: str,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> str:
    """Multi-head self-attention with RoPE and masked softmax.

    The mask is applied *after* the softmax and the result renormalised, rather
    than as an additive ``-inf`` bias before it.  Both are mathematically
    identical, but the additive form puts a large negative constant into the
    score tensor, which blows up that tensor's quantisation range and destroys
    the resolution of the real scores.  Post-softmax masking keeps every tensor
    inside ``[0, 1]``.
    """
    d_model = num_heads * head_dim
    h = rms_norm(g, x, block.norm.scale.value[layer])
    qkv = linear(g, h, block.qkv.kernel.value[layer])

    def head_split(index: int) -> str:
        part = slice_last(g, qkv, index * d_model, (index + 1) * d_model, rank=3)
        part = reshape(g, part, (1, seq_len, num_heads, head_dim))
        return g.op("Transpose", [part], perm=[0, 2, 1, 3])

    q = apply_rope(g, head_split(0), cos, sin, head_dim)
    k = apply_rope(g, head_split(1), cos, sin, head_dim)
    v = head_split(2)

    scores = g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 1, 3, 2])])
    scale = g.const(_f32(1.0 / math.sqrt(head_dim)), "attn_scale")
    weights = g.op("Softmax", [g.op("Mul", [scores, scale])], axis=-1)

    weights = g.op("Mul", [weights, mask_scores])
    # Opset 13 moved ReduceSum's axes from an attribute to an input (ReduceMean
    # kept the attribute until opset 18, hence the asymmetry with rms_norm).
    sum_axes = g.init(np.array([-1], dtype=np.int64), "sum_axes")
    total = g.op("ReduceSum", [weights, sum_axes], keepdims=1)
    weights = g.op("Div", [weights, g.op("Add", [total, g.const(_f32(1e-9), "attn_eps")])])

    out = g.op("MatMul", [weights, v])
    out = g.op("Transpose", [out], perm=[0, 2, 1, 3])
    out = reshape(g, out, (1, seq_len, d_model))
    return linear(g, out, block.out.kernel.value[layer])


def _conv_module(
    g: GraphBuilder,
    x: str,
    block,
    layer: int,
    *,
    mask_frames: str,
    seq_len: int,
    d_model: int,
) -> str:
    h = rms_norm(g, x, block.norm.scale.value[layer])
    h = linear(g, h, block.pw1.kernel.value[layer])

    value = slice_last(g, h, 0, d_model, rank=3)
    gate = slice_last(g, h, d_model, 2 * d_model, rank=3)
    h = g.op("Mul", [value, g.op("Sigmoid", [gate])])
    h = g.op("Mul", [h, mask_frames])

    # NNX depthwise kernels are (K, in/groups, out); ONNX wants (out, in/groups, K).
    kernel = np.asarray(block.dw.kernel.value[layer]).transpose(2, 1, 0)
    pad = (kernel.shape[-1] - 1) // 2

    h = g.op("Transpose", [h], perm=[0, 2, 1])
    h = g.op(
        "Conv",
        [h, g.init(_f32(kernel), "dw_kernel")],
        kernel_shape=[kernel.shape[-1]],
        strides=[1],
        pads=[pad, pad],
        group=d_model,
    )
    h = g.op("Transpose", [h], perm=[0, 2, 1])

    h = silu(g, rms_norm(g, h, block.act_norm.scale.value[layer]))
    return linear(g, h, block.pw2.kernel.value[layer])


def _subsample(g: GraphBuilder, mel: str, model, *, mel_frames: int, n_mels: int) -> str:
    """Two stride-2 2-D convs; ``(1, 1, T_mel, n_mels)`` NCHW in, ``(1, T, F, C)`` out."""
    x = mel

    for conv in (model.subsampler.conv1, model.subsampler.conv2):
        # NNX conv kernels are (KH, KW, in, out); ONNX wants (out, in, KH, KW).
        kernel = np.asarray(conv.kernel.value).transpose(3, 2, 0, 1)
        x = g.op(
            "Conv",
            [
                x,
                g.init(_f32(kernel), "sub_kernel"),
                g.init(_f32(conv.bias.value), "sub_bias"),
            ],
            kernel_shape=[kernel.shape[2], kernel.shape[3]],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
        )
        x = silu(g, x)

    # Back to (B, T, F, C) so the flatten order matches ConvSubsampler.__call__.
    x = g.op("Transpose", [x], perm=[0, 2, 3, 1])
    return x


def build_encoder_onnx(model, *, mel_frames: int, n_mels: int) -> tuple[onnx.ModelProto, dict]:
    """Convert a loaded ``FastConformerEncoder`` into a static ONNX graph.

    Returns the model and a metadata dict describing the tensor shapes the
    runtime needs to honour.
    """
    from .model import ConvSubsampler

    g = GraphBuilder()

    d_model = int(model.d_model)
    head_dim = int(model.head_dim)
    num_heads = d_model // head_dim
    num_layers = int(model.blocks.ff1.norm.scale.value.shape[0])

    seq_len = int(ConvSubsampler.output_length(np.int64(mel_frames)))
    freq_dim = int(ConvSubsampler.output_length(np.int64(n_mels)))
    if seq_len <= 0:
        raise ValueError(f"mel_frames={mel_frames} is too short to subsample")

    # Inputs are declared 4-D NCHW with a single channel.  ACUITY pads every
    # input to 4-D anyway and reads dimension 1 as the channel count when it
    # applies the per-channel mean/scale from the inputmeta -- a 3-D (1, T, F)
    # input makes it treat T as the channel axis and the calibration pass dies
    # trying to broadcast a scalar mean across it.
    mel_in = helper.make_tensor_value_info(
        "mel", TensorProto.FLOAT, [1, 1, mel_frames, n_mels]
    )
    mask_in = helper.make_tensor_value_info(
        "mask", TensorProto.FLOAT, [1, 1, seq_len, 1]
    )
    logits_out = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, seq_len, int(model.head.kernel.value.shape[-1])]
    )

    # (1, T, 1) drives the conv module; (1, 1, 1, T) masks attention keys.
    mask_frames = reshape(g, "mask", (1, seq_len, 1))
    mask_scores = reshape(g, "mask", (1, 1, 1, seq_len))

    x = _subsample(g, "mel", model, mel_frames=mel_frames, n_mels=n_mels)
    x = reshape(g, x, (1, seq_len, freq_dim * d_model))
    x = linear(g, x, model.proj.kernel.value, model.proj.bias.value)
    x = g.op("Mul", [x, g.const(_f32(math.sqrt(d_model)), "proj_scale")])

    cos_array, sin_array = rope_tables(head_dim, seq_len)
    cos = g.init(cos_array, "rope_cos")
    sin = g.init(sin_array, "rope_sin")

    for layer in range(num_layers):
        half = g.const(_f32(0.5), "macaron")
        x = g.op("Add", [x, g.op("Mul", [_ffn(g, x, model.blocks.ff1, layer), half])])
        x = g.op(
            "Add",
            [
                x,
                _attention(
                    g, x, model.blocks.attn, layer,
                    cos=cos, sin=sin, mask_scores=mask_scores,
                    seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
                ),
            ],
        )
        x = g.op(
            "Add",
            [
                x,
                _conv_module(
                    g, x, model.blocks.conv, layer,
                    mask_frames=mask_frames, seq_len=seq_len, d_model=d_model,
                ),
            ],
        )
        half = g.const(_f32(0.5), "macaron")
        x = g.op("Add", [x, g.op("Mul", [_ffn(g, x, model.blocks.ff2, layer), half])])

    x = rms_norm(g, x, model.final_norm.scale.value)
    linear(g, x, model.head.kernel.value, model.head.bias.value)
    # Rename the final Add so the graph output carries a stable name.
    g.nodes[-1].output[0] = "logits"

    graph = helper.make_graph(
        g.nodes,
        "fastconformer_encoder",
        [mel_in, mask_in],
        [logits_out],
        initializer=g.initializers,
    )
    onnx_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)]
    )
    onnx_model.ir_version = 8  # ONNX 1.12 in the SDK container tops out here.
    onnx.checker.check_model(onnx_model)

    meta = {
        "mel_frames": mel_frames,
        "n_mels": n_mels,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "vocab_size": int(model.head.kernel.value.shape[-1]),
    }
    return onnx_model, meta
