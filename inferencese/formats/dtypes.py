"""Storage dtypes and their decoding to numpy arrays.

Weights stay in their *storage* encoding (BF16, packed 4-bit, GGML blocks...)
inside every cache tier and are decoded only at the point of use, so a cached
byte is always a byte of the original file.
"""
from __future__ import annotations

import numpy as np

# name -> (numpy dtype or None, bytes per element)
PLAIN = {
    "F64": ("<f8", 8), "F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": (None, 2),
    "I64": ("<i8", 8), "I32": ("<i4", 4), "I16": ("<i2", 2), "I8": ("i1", 1),
    "U64": ("<u8", 8), "U32": ("<u4", 4), "U16": ("<u2", 2), "U8": ("u1", 1),
    "BOOL": ("?", 1),
}

# GGML block-quantised types: name -> (elements per block, bytes per block)
GGML_BLOCK = {
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36), "Q2_K": (256, 84), "Q3_K": (256, 110),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
}


def nbytes_of(dtype: str, shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    if dtype in PLAIN:
        return n * PLAIN[dtype][1]
    if dtype in GGML_BLOCK:
        per, size = GGML_BLOCK[dtype]
        return n // per * size
    raise ValueError(f"unknown storage dtype {dtype}")


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 is the upper half of an IEEE float32."""
    u = np.frombuffer(raw, dtype="<u2") if raw.dtype != np.uint16 else raw
    out = np.empty(u.shape, dtype=np.uint32)
    np.left_shift(u, 16, out=out, dtype=np.uint32, casting="unsafe")
    return out.view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 bit pattern (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = u + (0x7FFF + ((u >> 16) & 1))
    return (rounded >> 16).astype(np.uint16)


def decode(raw, dtype: str, shape, float_dtype=np.float32) -> np.ndarray:
    """Decode raw storage bytes into a numpy array (floats -> ``float_dtype``)."""
    raw = np.asarray(raw)
    if dtype == "BF16":
        return bf16_to_f32(raw).reshape(shape).astype(float_dtype, copy=False)
    if dtype in PLAIN:
        arr = np.frombuffer(raw, dtype=PLAIN[dtype][0]).reshape(shape)
        if arr.dtype.kind == "f" and arr.dtype != float_dtype:
            arr = arr.astype(float_dtype)
        return arr
    if dtype in GGML_BLOCK:
        return dequant_ggml(raw, dtype).reshape(shape).astype(float_dtype, copy=False)
    raise ValueError(f"unknown storage dtype {dtype}")


# --------------------------------------------------------------- GGML blocks
def dequant_ggml(raw: np.ndarray, dtype: str) -> np.ndarray:
    per, size = GGML_BLOCK[dtype]
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
    fn = _GGML.get(dtype)
    if fn is None:
        try:                                    # reference implementation, if installed
            import gguf
            return gguf.quants.dequantize(blk.reshape(-1), gguf.GGMLQuantizationType[dtype]).reshape(-1)
        except ImportError:
            raise NotImplementedError(
                f"GGML type {dtype} needs the 'gguf' package (pip install gguf)") from None
    return fn(blk).reshape(-1)


def _q8_0(b):
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    return d * b[:, 2:].view(np.int8).astype(np.float32)


def _q4_0(b):
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    qs = b[:, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int8) - 8
    return d * q.astype(np.float32)


def _q5_0(b):
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    qh = b[:, 2:6].copy().view("<u4")                               # one high bit per element
    bits = ((qh >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
    qs = b[:, 6:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1) | (bits << 4)
    return d * (q.astype(np.float32) - 16.0)


def _k4_scales(sc):
    """Unpack the 8 six-bit (scale, min) pairs of a K-quant block."""
    d = np.empty((sc.shape[0], 8), np.uint8)
    m = np.empty((sc.shape[0], 8), np.uint8)
    d[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    d[:, 4:] = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


def _q4_k(b):
    n = b.shape[0]
    d = b[:, 0:2].copy().view("<f2").astype(np.float32)
    dmin = b[:, 2:4].copy().view("<f2").astype(np.float32)
    sc, mn = _k4_scales(b[:, 4:16])
    qs = b[:, 16:144].reshape(n, 4, 32)
    q = np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(n, 8, 32).astype(np.float32)
    return ((d * sc)[:, :, None] * q - (dmin * mn)[:, :, None]).reshape(n, 256)


def _q6_k(b):
    n = b.shape[0]
    ql = b[:, 0:128].reshape(n, 2, 64)
    qh = b[:, 128:192].reshape(n, 2, 32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32).reshape(n, 2, 8)
    d = b[:, 208:210].copy().view("<f2").astype(np.float32)
    lo, hi = ql[:, :, :32], ql[:, :, 32:]
    q = np.stack([
        (lo & 0x0F) | ((qh & 3) << 4),
        (hi & 0x0F) | (((qh >> 2) & 3) << 4),
        (lo >> 4) | (((qh >> 4) & 3) << 4),
        (hi >> 4) | (((qh >> 6) & 3) << 4),
    ], axis=2).astype(np.int8) - 32                         # (n, 2, 4, 32)
    s = sc.reshape(n, 2, 4, 2).repeat(16, axis=3)           # scale index = l // 16 + 2 * group
    return (d[:, :, None, None] * s * q.astype(np.float32)).reshape(n, 256)


_GGML = {"Q8_0": _q8_0, "Q4_0": _q4_0, "Q5_0": _q5_0, "Q4_K": _q4_k, "Q6_K": _q6_k}
