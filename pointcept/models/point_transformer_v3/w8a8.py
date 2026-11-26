"""Portable static W8A8 inference for PTv3 on NVIDIA Ampere and newer GPUs."""
import math
import sys
from pathlib import Path

import cumm.common

import torch
import triton
import triton.language as tl
from spconv.core import ConvAlgo
from spconv.pytorch import SubMConv3d
from spconv.pytorch.modules import SparseModule

FORMAT = "ptv3-w8a8-v1"


def target_modules(model):
    # The 6-channel, 5x5x5 stem uses the existing FP16 convolution.
    return {n: m for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) or (
                isinstance(m, SubMConv3d) and math.prod(m.kernel_size) <= 32
                and m.in_channels % 16 == 0 and m.out_channels % 16 == 0)}


@torch.inference_mode()
def export_w8a8(model, calibration_batches):
    layers = target_modules(model)
    maxima = {}

    def observer(name):
        def observe(module, inputs, output):
            sparse = isinstance(module, SubMConv3d)
            x = inputs[0].features if sparse else inputs[0]
            peaks = [x.detach().abs().amax().float()]
            if sparse:
                peaks.append(output.features.detach().abs().amax().float())
            peaks = torch.stack(peaks)
            maxima[name] = peaks if name not in maxima else torch.maximum(maxima[name], peaks)
        return observe

    handles = [m.register_forward_hook(observer(n)) for n, m in layers.items()]
    try:
        for batch in calibration_batches:
            with torch.autocast("cuda", dtype=torch.float16):
                model(batch)
    finally:
        for handle in handles:
            handle.remove()

    state = {n: t.detach().cpu() for n, t in model.state_dict().items()}
    specs = {}
    for name in layers:
        peaks = maxima[name].cpu()
        if not torch.isfinite(peaks).all():
            raise ValueError(f"Nonfinite calibration values: {name}")
        key = name + ".weight"
        w = state.pop(key).float()
        scale = (w.flatten(1).abs().amax(1) / 127).clamp_min(1e-10)
        w_int8 = (w / scale.view(-1, *([1] * (w.ndim - 1)))).round().clamp(-127, 127)
        specs[name] = dict(weight=w_int8.to(torch.int8), scales=scale,
                           activation=(peaks / 127).clamp_min(1e-10).tolist(),
                           bias=state.pop(name + ".bias", None))
    return dict(format=FORMAT, layers=specs, remaining=state)


@triton.jit
def _linear_kernel(X, W, S, B, Y, M, K: tl.constexpr, N: tl.constexpr,
                   INV, HAS_BIAS: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        x = tl.load(X + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        q = tl.extra.cuda.libdevice.rint(x.to(tl.float32) * INV)
        q = tl.minimum(127., tl.maximum(-127., q)).to(tl.int8)
        w = tl.load(W + cols[None, :] * K + kk[:, None],
                    (kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc += tl.dot(q, w, out_dtype=tl.int32)
    y = acc.to(tl.float32) * tl.load(S + cols, cols < N, other=0)[None, :]
    if HAS_BIAS:
        y += tl.load(B + cols, cols < N, other=0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], y,
             (rows[:, None] < M) & (cols[None, :] < N))


class Int8Linear(torch.nn.Module):
    def __init__(self, spec, device):
        super().__init__()
        sx, = spec["activation"]
        self.inv = 1.0 / sx
        self.register_buffer("weight", spec["weight"].to(device))
        self.register_buffer("scale", (spec["scales"] * sx).to(device))
        bias = spec["bias"]
        self.register_buffer("bias", None if bias is None else bias.to(device))

    def forward(self, x):
        x = x.contiguous()
        n, k = self.weight.shape
        m = x.numel() // k
        out = x.new_empty((*x.shape[:-1], n), dtype=torch.float16)
        if m:
            _linear_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
                x, self.weight, self.scale, self.bias if self.bias is not None else self.scale,
                out, m, k, n, self.inv, self.bias is not None, 32, 64, 32)
        return out


class Int8SubMConv(SparseModule):
    def __init__(self, conv, spec, device):
        super().__init__()
        self.conv = conv
        del self.conv.weight
        self.conv.register_parameter("bias", None)
        self.conv.algo = ConvAlgo.MaskImplicitGemm
        self.sx, self.sy = spec["activation"]
        s = spec["scales"]
        qw = torch._make_per_channel_quantized_tensor(
            spec["weight"], s.double(), torch.zeros_like(s, dtype=torch.long), 0)
        self.register_buffer("kernel", qw.to(device))
        self.register_buffer("multiplier", (s * (self.sx / self.sy)).to(device))
        bias = spec["bias"]
        bias = torch.zeros_like(s) if bias is None else bias.float()
        self.register_buffer("scaled_bias", (bias / self.sy).to(device))

    def forward(self, sparse):
        x = torch.quantize_per_tensor(sparse.features.float(), self.sx, 0, torch.qint8)
        y = self.conv._conv_forward(False, sparse.replace_feature(x), self.kernel,
                                    self.scaled_bias, channel_scale=self.multiplier,
                                    output_scale=self.sy)
        if y.features.dtype != torch.qint8:
            raise RuntimeError("spconv failed to produce quantized activations")
        return y.replace_feature(y.features.dequantize().half())


def load_w8a8(model, checkpoint):
    if checkpoint.get("format") != FORMAT:
        raise ValueError(f"Expected checkpoint format {FORMAT}")
    # cumm 0.7 only detects the targets/ layout of conda CUDA toolkits.
    # Prefer this environment's matching NVRTC headers over /usr/local/cuda.
    prefix = Path(sys.prefix)
    if (prefix / "include/cuda.h").is_file() and (prefix / "lib/libcudart.so").is_file():
        cumm.common._CACHED_CUDA_INCLUDE_LIB = ([prefix / "include"], prefix / "lib")
    layers = target_modules(model)
    for module in model.modules():
        if isinstance(module, SubMConv3d):
            module.algo = ConvAlgo.Native
    specs = checkpoint["layers"]
    if specs.keys() != layers.keys():
        raise ValueError("W8A8 module names do not match the configured model")
    removed = {name + "." + suffix for name, mod in layers.items()
               for suffix in ("weight", "bias") if getattr(mod, suffix) is not None}
    missing, extra = model.load_state_dict(checkpoint["remaining"], strict=False)
    if set(missing) != removed or extra:
        raise ValueError("Incomplete W8A8 checkpoint")
    model.half()
    for name, mod in layers.items():
        parent, _, key = name.rpartition(".")
        spec = specs[name]
        if spec["weight"].dtype != torch.int8 or spec["weight"].shape != mod.weight.shape:
            raise ValueError(f"Invalid INT8 weight: {name}")
        cls = Int8Linear if isinstance(mod, torch.nn.Linear) else Int8SubMConv
        device = mod.weight.device
        replacement = cls(spec, device) if cls is Int8Linear else cls(mod, spec, device)
        setattr(model.get_submodule(parent), key, replacement)
    return model.eval()
