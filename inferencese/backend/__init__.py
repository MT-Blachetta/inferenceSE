"""Compute backends.

The virtual machine does not implement tensor kernels itself; it drives an
existing array library through a small common surface:

* ``numpy``  - portable CPU backend, float32 compute.
* ``torch``  - PyTorch; with ``device: cuda`` the GPU's VRAM becomes tier 0 of
  the unified memory and weights are used in their storage dtype (BF16 stays
  BF16 on the GPU, packed 4-bit stays packed until the moment of use).

Activations are float32 on both backends; a weight's storage dtype only
matters inside ``linear``/``weight``.
"""
from __future__ import annotations

import numpy as np

from ..formats import dtypes as _dt
from ..uvm import HostDevice


class NumpyBackend:
    name = "numpy"
    is_gpu = False

    def __init__(self, **kw):
        self.device = HostDevice()
        self.threads = kw.get("threads")

    # ---- conversion
    def weight(self, raw, dtype, shape):
        """Storage bytes -> compute-ready array."""
        return _dt.decode(raw, dtype, shape)

    def raw(self, handle):
        """Uniform uint8 view of a cache handle for packed-format kernels."""
        return np.asarray(handle)

    def asarray(self, x, dtype="f32"):
        return np.asarray(x, dtype={"f32": np.float32, "i64": np.int64, "bool": bool,
                                    "u8": np.uint8, "f16": np.float16}[dtype])

    def to_numpy(self, x):
        return np.asarray(x)

    def from_numpy(self, x):
        return x

    def f32(self, x):
        return np.asarray(x, dtype=np.float32)

    # ---- math
    def linear(self, x, w):
        return x @ w.T

    def matmul(self, a, b):
        return a @ b

    def einsum(self, eq, *xs):
        return np.einsum(eq, *xs, optimize=True)

    def exp(self, x): return np.exp(x)
    def log(self, x): return np.log(x)
    def tanh(self, x): return np.tanh(x)
    def sqrt(self, x): return np.sqrt(x)
    def cos(self, x): return np.cos(x)
    def sin(self, x): return np.sin(x)
    def abs(self, x): return np.abs(x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def silu(self, x):
        return x / (1.0 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def gelu(self, x, tanh=False):
        if tanh:
            return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
        from math import sqrt
        return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))

    def softmax(self, x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def sum(self, x, axis=None, keepdims=False): return x.sum(axis=axis, keepdims=keepdims)
    def mean(self, x, axis=None, keepdims=False): return x.mean(axis=axis, keepdims=keepdims)
    def max(self, x, axis=None, keepdims=False): return x.max(axis=axis, keepdims=keepdims)
    def argmax(self, x, axis=-1): return x.argmax(axis=axis)
    def maximum(self, a, b): return np.maximum(a, b)
    def minimum(self, a, b): return np.minimum(a, b)
    def where(self, c, a, b): return np.where(c, a, b)
    def cat(self, xs, axis=0): return np.concatenate(xs, axis=axis)
    def stack(self, xs, axis=0): return np.stack(xs, axis=axis)
    def zeros(self, shape, dtype="f32"): return np.zeros(shape, dtype=self.asarray(0, dtype).dtype)
    def ones(self, shape, dtype="f32"): return np.ones(shape, dtype=self.asarray(0, dtype).dtype)
    def arange(self, n, dtype="i64"): return np.arange(n, dtype=self.asarray(0, dtype).dtype)
    def permute(self, x, axes): return x.transpose(axes)
    def take(self, x, idx, axis=0): return np.take(x, idx, axis=axis)
    def take_along(self, x, idx, axis=-1): return np.take_along_axis(x, idx, axis=axis)
    def copy(self, x): return x.copy()
    def tril_mask(self, t, s, offset=0):
        """Boolean [t, s] mask, True where key j is visible to query i (j <= i + offset)."""
        return np.arange(s)[None, :] <= (np.arange(t)[:, None] + offset)

    def topk(self, x, k, axis=-1):
        """Largest ``k`` along ``axis`` (unordered); returns (values, indices)."""
        n = x.shape[axis]
        if k >= n:
            idx = np.broadcast_to(np.arange(n).reshape([-1 if a == (axis % x.ndim) else 1
                                                         for a in range(x.ndim)]), x.shape)
            return x, idx
        idx = np.argpartition(x, n - k, axis=axis)
        idx = np.take(idx, np.arange(n - k, n), axis=axis)
        return np.take_along_axis(x, idx, axis=axis), idx

    def set_threads(self, n):
        pass


def _erf(x):
    # Abramowitz-Stegun 7.1.26; |error| < 1.5e-7, adequate for GELU in float32.
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return s * y


def make_backend(name: str = "auto", device: str = "auto", **kw):
    name = (name or "auto").lower()
    if name in ("auto", "torch"):
        try:
            from .torch_backend import TorchBackend
            import torch
            if name == "torch" or (device != "cpu" and torch.cuda.is_available()):
                return TorchBackend(device=device, **kw)
        except ImportError:
            if name == "torch":
                raise
    if name in ("auto", "numpy"):
        return NumpyBackend(**kw)
    raise ValueError(f"unknown backend {name!r} (numpy, torch, auto)")
