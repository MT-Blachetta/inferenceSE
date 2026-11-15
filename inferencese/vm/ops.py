"""Built-in VM ops.

These cover ordinary decoder-only transformers.  Model-specific kernels are
supplied by an optimization profile's ``ops.py`` and registered the same way.
Every op receives the ``Context`` first; weight operands arrive as ``W`` names
and are materialised through ``ctx.ws`` at the moment of use, which is what
routes each access through the unified virtual memory.
"""
from __future__ import annotations

import math

import numpy as np

from ..runtime.store import W

OPS: dict = {}


def op(name: str, registry: dict | None = None):
    def deco(fn):
        (OPS if registry is None else registry)[name] = fn
        return fn
    return deco


def _w(ctx, w):
    return ctx.ws.get(w) if isinstance(w, W) else w


def rms_norm(B, x, weight, eps):
    x = B.f32(x)
    y = x * (1.0 / B.sqrt(B.mean(x * x, -1, True) + eps))
    return y if weight is None else y * B.f32(weight)


# ------------------------------------------------------------------ basics
@op("embed")
def embed(ctx, w, tokens, scale=1.0):
    x = ctx.ws.rows(w, tokens)
    return x * scale if scale != 1.0 else x


@op("linear")
def linear(ctx, x, w, bias=None):
    if isinstance(w, W) and ctx.ws.is_chunked(w):       # matrix streamed in row chunks
        y = ctx.B.cat([ctx.B.linear(x, c) for c in ctx.ws.iter_chunks(w)], -1)
    else:
        y = ctx.B.linear(x, _w(ctx, w))
    if bias is not None and (not isinstance(bias, W) or ctx.ws.has(bias)):
        y = y + ctx.B.f32(_w(ctx, bias))
    return y


@op("rmsnorm")
def rmsnorm(ctx, x, w=None, eps=1e-6):
    return rms_norm(ctx.B, x, _w(ctx, w) if w is not None else None, eps)


@op("layernorm")
def layernorm(ctx, x, w, b=None, eps=1e-5):
    B = ctx.B
    x = B.f32(x)
    mu = B.mean(x, -1, True)
    xc = x - mu
    y = xc / B.sqrt(B.mean(xc * xc, -1, True) + eps) * B.f32(_w(ctx, w))
    return y + B.f32(_w(ctx, b)) if b is not None else y


@op("add")
def add(ctx, a, b): return a + b


@op("sub")
def sub(ctx, a, b): return a - b


@op("mul")
def mul(ctx, a, b): return a * b


@op("scale")
def scale(ctx, a, s): return a * s


@op("silu")
def silu(ctx, x): return ctx.B.silu(x)


@op("gelu")
def gelu(ctx, x, tanh=False): return ctx.B.gelu(x, tanh)


@op("relu")
def relu(ctx, x): return ctx.B.relu(x)


@op("sigmoid")
def sigmoid(ctx, x): return ctx.B.sigmoid(x)


@op("tanh")
def tanh(ctx, x): return ctx.B.tanh(x)


@op("softmax")
def softmax(ctx, x, axis=-1): return ctx.B.softmax(x, axis)


@op("concat")
def concat(ctx, xs, axis=-1): return ctx.B.cat(xs, axis)


@op("last")
def last(ctx, x):
    """Keep only the final position (logits are needed for the last token)."""
    return x[-1:]


@op("weight")
def weight(ctx, w):
    return ctx.B.f32(ctx.ws.get(w))


@op("posembed")
def posembed(ctx, x, w):
    """Learned absolute position embeddings (GPT-2 style)."""
    pos = ctx.state.pos + np.arange(x.shape[0])
    return x + ctx.ws.rows(w, pos)


# ------------------------------------------------------------ memory hints
@op("release")
def release(ctx, *patterns):
    """Drop cached copies of weights that will not be used again this step."""
    for p in patterns:
        ctx.mem.release(ctx.ws.blocks(ctx.ws.match(str(p))))


@op("prefetch")
def prefetch(ctx, *patterns):
    """Start reading weights from the SSD before they are needed."""
    for p in patterns:
        ctx.mem.prefetch(ctx.ws.blocks(ctx.ws.match(str(p))))


@op("trace")
def trace(ctx, x, label="trace"):
    a = ctx.B.to_numpy(x)
    print(f"[{label}] shape={a.shape} mean={a.mean():.6g} std={a.std():.6g} "
          f"min={a.min():.6g} max={a.max():.6g}", flush=True)
    return x


# --------------------------------------------------------------- attention
def rope_tables(ctx, positions, dim, theta, style):
    B = ctx.B
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    return B.asarray(np.cos(ang)), B.asarray(np.sin(ang))


def apply_rope(B, x, cos, sin, style="half"):
    """x: [T, H, D]; cos/sin: [T, D/2]."""
    d = x.shape[-1] // 2
    c, s = cos[:, None, :], sin[:, None, :]
    if style == "half":                       # Hugging Face layout: [x1 | x2]
        x1, x2 = x[..., :d], x[..., d:]
        return B.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]       # GGUF/llama.cpp layout: interleaved pairs
    return B.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(x.shape)


@op("attention")
def attention(ctx, x, q, k, v, o, heads, kv_heads=None, key="attn", rope_theta=None,
              rope_style="half", q_bias=None, k_bias=None, v_bias=None, o_bias=None,
              head_dim=None):
    """Causal multi-head / grouped-query self-attention with a paged KV cache."""
    B, ws, st = ctx.B, ctx.ws, ctx.state
    T = x.shape[0]
    kvh = kv_heads or heads
    qv = linear(ctx, x, q, q_bias)
    kx = linear(ctx, x, k, k_bias)
    vx = linear(ctx, x, v, v_bias)
    d = head_dim or qv.shape[-1] // heads
    qv, kx, vx = qv.reshape(T, heads, d), kx.reshape(T, kvh, d), vx.reshape(T, kvh, d)
    klog, vlog = st.log(f"{key}.k", kvh * d), st.log(f"{key}.v", kvh * d)
    past = len(klog)
    if rope_theta:
        cos, sin = rope_tables(ctx, past + np.arange(T), d, rope_theta, rope_style)
        qv, kx = apply_rope(B, qv, cos, sin, rope_style), apply_rope(B, kx, cos, sin, rope_style)
    klog.append(kx)
    vlog.append(vx)
    K = klog.read_all().reshape(past + T, kvh, d)
    V = vlog.read_all().reshape(past + T, kvh, d)
    g = heads // kvh
    qg = qv.reshape(T, kvh, g, d)
    att = B.einsum("tkgd,skd->kgts", qg, K) * (1.0 / math.sqrt(d))
    mask = B.tril_mask(T, past + T, offset=past)
    att = B.where(mask[None, None], att, B.asarray(-1e30))
    p = B.softmax(att, -1)
    out = B.einsum("kgts,skd->tkgd", p, V).reshape(T, heads * d)
    return linear(ctx, out, o, o_bias)
