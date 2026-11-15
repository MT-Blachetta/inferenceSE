"""PyTorch backend.  With ``device='cuda'`` VRAM is tier 0 of the unified memory."""
from __future__ import annotations

import numpy as np
import torch

from ..formats import dtypes as _dt

_TORCH_DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
             "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
             "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
_NAMED = {"f32": torch.float32, "i64": torch.int64, "bool": torch.bool,
          "u8": torch.uint8, "f16": torch.float16}


class TorchDevice:
    """Tier-0 allocator: raw byte blocks living in VRAM as uint8 tensors."""

    def __init__(self, device, pin: bool = False):
        self.dev = device
        self.name = str(device)
        self.pin = bool(pin) and device.type == "cuda"
        if self.pin:
            self.alloc_host = self._alloc_pinned

    def _alloc_pinned(self, nbytes: int):
        """Page-locked read buffer: SSD -> RAM lands directly in memory the
        GPU can DMA from, so RAM -> VRAM copies run at bus speed."""
        from ..uvm.io import ALIGN, aligned_buffer
        size = max(ALIGN, (nbytes + ALIGN - 1) // ALIGN * ALIGN)
        if self.pin:
            try:
                arr = torch.empty(size, dtype=torch.uint8, pin_memory=True).numpy()
                if arr.ctypes.data % ALIGN == 0:
                    return arr
            except RuntimeError:
                pass
            self.pin = False                    # locked-memory limit reached: fall back
        return aligned_buffer(nbytes)

    def upload(self, buf):
        if isinstance(buf, torch.Tensor):
            return buf.to(self.dev)
        return torch.from_numpy(np.asarray(buf)).to(self.dev)

    def download(self, handle) -> np.ndarray:
        return handle.detach().cpu().numpy()

    def free(self, handle) -> None:
        pass                    # released when the last reference is dropped


class TorchBackend:
    name = "torch"

    def __init__(self, device: str = "auto", threads: int | None = None,
                 matmul_dtype: str = "auto", pin_host_memory: bool = False, **kw):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)
        self.is_gpu = self.dev.type != "cpu"
        self.device = TorchDevice(self.dev, pin=pin_host_memory)
        # dtype used for the matmul itself: storage dtype on GPU (BF16 stays BF16,
        # no 2x blow-up in VRAM), float32 on CPU where BF16 kernels are slow.
        self.native = self.is_gpu if matmul_dtype == "auto" else matmul_dtype == "native"
        if threads:
            torch.set_num_threads(int(threads))
        torch.set_grad_enabled(False)

    def set_threads(self, n):
        torch.set_num_threads(int(n))

    def vram_bytes(self) -> int:
        if self.dev.type == "cuda":
            return torch.cuda.get_device_properties(self.dev).total_memory
        return 0

    # ---- conversion
    def raw(self, handle):
        if isinstance(handle, torch.Tensor):
            return handle
        return torch.from_numpy(np.asarray(handle)).to(self.dev)

    def weight(self, raw, dtype, shape):
        tdt = _TORCH_DT.get(dtype)
        if tdt is None:                      # exotic storage (GGML blocks): decode on host
            host = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else raw
            return torch.from_numpy(np.ascontiguousarray(_dt.decode(host, dtype, shape))).to(self.dev)
        t = self.raw(raw)
        w = t.view(tdt).view(tuple(shape)) if tdt != torch.uint8 else t.view(tuple(shape))
        if w.is_floating_point() and not self.native and w.dtype != torch.float32:
            w = w.float()
        return w

    def asarray(self, x, dtype="f32"):
        if isinstance(x, torch.Tensor):
            return x.to(self.dev, _NAMED[dtype])
        return torch.as_tensor(np.asarray(x), dtype=_NAMED[dtype], device=self.dev)

    def to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.dtype == torch.bfloat16:
                x = x.float()
            return x.cpu().numpy()
        return np.asarray(x)

    def from_numpy(self, x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.dev)

    def f32(self, x):
        return x.float() if isinstance(x, torch.Tensor) else self.asarray(x)

    # ---- math
    def linear(self, x, w):
        if w.dtype != x.dtype:
            if w.is_floating_point() and w.dtype != torch.float32:
                return torch.nn.functional.linear(x.to(w.dtype), w).float()
            w = w.to(x.dtype)
        return torch.nn.functional.linear(x, w)

    def matmul(self, a, b):
        return a @ b

    def einsum(self, eq, *xs):
        return torch.einsum(eq, *xs)

    def exp(self, x): return torch.exp(x)
    def log(self, x): return torch.log(x)
    def tanh(self, x): return torch.tanh(x)
    def sqrt(self, x): return torch.sqrt(x)
    def cos(self, x): return torch.cos(x)
    def sin(self, x): return torch.sin(x)
    def abs(self, x): return torch.abs(x)
    def sigmoid(self, x): return torch.sigmoid(x)
    def silu(self, x): return torch.nn.functional.silu(x)
    def relu(self, x): return torch.relu(x)

    def gelu(self, x, tanh=False):
        return torch.nn.functional.gelu(x, approximate="tanh" if tanh else "none")

    def softmax(self, x, axis=-1): return torch.softmax(x, dim=axis)

    def sum(self, x, axis=None, keepdims=False):
        return x.sum() if axis is None else x.sum(dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return x.mean() if axis is None else x.mean(dim=axis, keepdim=keepdims)

    def max(self, x, axis=None, keepdims=False):
        return x.max() if axis is None else x.amax(dim=axis, keepdim=keepdims)

    def argmax(self, x, axis=-1): return x.argmax(dim=axis)
    def maximum(self, a, b): return torch.maximum(a, torch.as_tensor(b, dtype=a.dtype, device=a.device))
    def minimum(self, a, b): return torch.minimum(a, torch.as_tensor(b, dtype=a.dtype, device=a.device))
    def where(self, c, a, b): return torch.where(c, a, b)
    def cat(self, xs, axis=0): return torch.cat(list(xs), dim=axis)
    def stack(self, xs, axis=0): return torch.stack(list(xs), dim=axis)
    def zeros(self, shape, dtype="f32"): return torch.zeros(shape, dtype=_NAMED[dtype], device=self.dev)
    def ones(self, shape, dtype="f32"): return torch.ones(shape, dtype=_NAMED[dtype], device=self.dev)
    def arange(self, n, dtype="i64"): return torch.arange(n, dtype=_NAMED[dtype], device=self.dev)
    def permute(self, x, axes): return x.permute(*axes)

    def take(self, x, idx, axis=0):
        return torch.index_select(x, axis, self.asarray(idx, "i64").reshape(-1)).reshape(
            x.shape[:axis] + tuple(np.shape(idx) if not isinstance(idx, torch.Tensor) else idx.shape)
            + x.shape[axis + 1:])

    def take_along(self, x, idx, axis=-1): return torch.take_along_dim(x, idx, dim=axis)
    def copy(self, x): return x.clone()

    def tril_mask(self, t, s, offset=0):
        return (torch.arange(s, device=self.dev)[None, :]
                <= torch.arange(t, device=self.dev)[:, None] + offset)

    def topk(self, x, k, axis=-1):
        k = min(k, x.shape[axis])
        v, i = torch.topk(x, k, dim=axis, sorted=False)
        return v, i
