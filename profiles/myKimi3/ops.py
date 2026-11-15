"""VM ops for Kimi K3.1 Pro (text path).

The mathematics follows the reference implementation in vLLM
(``vllm/models/kimi_k3``, Apache-2.0), reduced to plain array code:

* ``attn_res``   attention-residual mixing of block outputs (AttnRes)
* ``situ``       SiTU-GLU activation
* ``kda``        Kimi Delta Attention, a recurrent linear attention with O(1)
                 state per sequence
* ``dsa``        gated sparse multi-head latent attention with a top-k indexer
* ``moe``        latent mixture of experts; MXFP4 experts are streamed from the
                 SSD through the unified memory, 16 of 896 per token and layer

``op`` is injected by the profile loader.
"""
import math

import numpy as np

from inferencese.runtime.store import W
from inferencese.vm.ops import rms_norm

try:
    import torch
except ImportError:                                     # numpy-only installs
    torch = None

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


# ------------------------------------------------------------------ MXFP4
def mxfp4_decode(B, packed, scale, out_features, in_features):
    """OCP MXFP4 -> float32 [out, in].

    ``packed``: uint8 [out, in/2], two E2M1 values per byte, low nibble first.
    ``scale``:  uint8 [out, in/32], E8M0 exponent (2**(s-127)) per 32 values.
    """
    if torch is not None and isinstance(packed, torch.Tensor):
        lut = B.cache_lut if hasattr(B, "cache_lut") else None
        if lut is None:
            lut = B.cache_lut = torch.from_numpy(FP4).to(packed.device)
        p = packed.reshape(out_features, in_features // 2)
        w = torch.stack((lut[(p & 0x0F).long()], lut[(p >> 4).long()]), dim=-1)
        s = torch.pow(2.0, scale.reshape(out_features, -1).to(torch.float32) - 127.0)
        return (w.reshape(out_features, -1, 32) * s[:, :, None]).reshape(out_features, in_features)
    p = np.asarray(packed).reshape(out_features, in_features // 2)
    w = np.empty((out_features, in_features // 2, 2), dtype=np.float32)
    w[:, :, 0] = FP4[p & 0x0F]
    w[:, :, 1] = FP4[p >> 4]
    s = np.ldexp(np.float32(1.0), np.asarray(scale).reshape(out_features, -1).astype(np.int32) - 127)
    w = w.reshape(out_features, -1, 32)
    w *= s[:, :, None].astype(np.float32)
    return w.reshape(out_features, in_features)


# ---------------------------------------------------------------- AttnRes
@op("attn_res")
def attn_res(ctx, prefix, delta, blocks, norm, proj, out_norm=None, num_blocks=0,
             write=-1, eps=1e-5):
    """Mix earlier block outputs into the current stream.

    ``prefix`` is the running sum inside the current block of layers, ``blocks``
    the list of completed block outputs.  Sources are ``blocks[:num_blocks]``
    plus the prefix; each is scored by ``<rmsnorm(v) * norm, proj>`` and the
    result is their softmax-weighted sum.  Returns (mixed [normalised by
    ``out_norm``], updated prefix).
    """
    B, ws = ctx.B, ctx.ws
    if delta is not None:
        prefix = prefix + delta
    if write >= 0:
        while len(blocks) <= write:
            blocks.append(None)
        blocks[write] = prefix
    if num_blocks == 0:
        mixed = prefix
    else:
        qk = B.f32(ws.get(norm)) * B.f32(ws.get(proj)).reshape(-1)
        src = B.stack(list(blocks[:num_blocks]) + [prefix], 0)             # [S, T, D]
        rstd = 1.0 / B.sqrt(B.mean(src * src, -1) + eps)                   # [S, T]
        logits = B.einsum("std,d->st", src, qk) * rstd
        p = B.softmax(logits, 0)
        mixed = B.einsum("st,std->td", p, src)
    if out_norm is not None:
        mixed = rms_norm(B, mixed, ws.get(out_norm), eps)
    return mixed, prefix


@op("new_blocks")
def new_blocks(ctx):
    return []


# ------------------------------------------------------------------- SiTU
def situ_glu(B, gate, up, beta, linear_beta):
    g = beta * B.tanh(gate * (1.0 / beta)) * B.sigmoid(gate)
    if linear_beta:
        up = linear_beta * B.tanh(up * (1.0 / linear_beta))
    return g * up


@op("situ")
def situ(ctx, gate, up):
    c = ctx.cfg
    return situ_glu(ctx.B, gate, up, c.get("activation_situ_beta") or 1.0,
                    c.get("activation_situ_linear_beta"))


# -------------------------------------------------------------------- KDA
@op("kda")
def kda(ctx, x, layer, prefix):
    """Kimi Delta Attention (full-rank gate).  State per layer: the last
    ``kernel-1`` projected inputs of the short convolution and the recurrent
    matrix S[head, value, key]; neither grows with the context."""
    B, ws, st = ctx.B, ctx.ws, ctx.state
    lc = ctx.cfg.linear_attn_config
    H, D, ksz = lc.num_heads, lc.head_dim, lc.short_conv_kernel_size
    P, T = H * D, x.shape[0]
    lin = lambda name, inp=x: B.linear(inp, ws.get(W(f"{prefix}.{name}.weight")))

    qkv = B.cat([lin("q_proj"), lin("k_proj"), lin("v_proj")], -1)        # [T, 3P]
    g_out = lin("g_proj").reshape(T, H, D)                                # output gate
    g_raw = lin("f_b_proj", lin("f_a_proj")).reshape(T, H, D)             # decay gate
    beta = B.sigmoid(lin("b_proj"))                                       # [T, H]

    # short causal depthwise convolution + SiLU over q, k, v
    cw = B.cat([B.f32(ws.get(W(f"{prefix}.{n}_conv1d.weight"))).reshape(P, ksz)
                for n in "qkv"], 0)                                       # [3P, ksz]
    hist = st.vars.get(f"kda.{layer}.conv")
    if hist is None:
        hist = B.zeros((ksz - 1, 3 * P))
    seq = B.cat([hist, qkv], 0)                                           # [T+ksz-1, 3P]
    conv = seq[0:T] * cw[:, 0]
    for j in range(1, ksz):
        conv = conv + seq[j:j + T] * cw[:, j]
    st.vars[f"kda.{layer}.conv"] = B.copy(seq[T:])
    qkv = B.silu(conv)
    q, k, v = (qkv[:, i * P:(i + 1) * P].reshape(T, H, D) for i in range(3))
    q = q * (1.0 / B.sqrt(B.sum(q * q, -1, True) + 1e-6)) * (D ** -0.5)
    k = k * (1.0 / B.sqrt(B.sum(k * k, -1, True) + 1e-6))

    # decay gate: lower_bound * sigmoid(exp(A_log) * (g + dt_bias)), per key dim
    a_log = B.f32(ws.get(W(f"{prefix}.A_log")))
    if ctx.opts.get("kda_a_log_axis", "head") == "head":
        a = B.exp(a_log[:H]).reshape(1, H, 1)       # as indexed by the reference kernels
    else:
        a = B.exp(a_log[:D]).reshape(1, 1, D)
    dt_bias = B.f32(ws.get(W(f"{prefix}.dt_bias"))).reshape(1, H, D)
    lb = lc.get("gate_lower_bound")
    if lb is None:
        raise NotImplementedError("KDA without gate_lower_bound is not implemented")
    decay = B.exp(lb * B.sigmoid(a * (g_raw + dt_bias)))                  # [T, H, D]

    S = st.vars.get(f"kda.{layer}.S")
    if S is None:
        S = B.zeros((H, D, D))                                            # [head, value, key]
    outs = []
    for t in range(T):
        S = S * decay[t][:, None, :]
        err = (v[t] - B.einsum("hvk,hk->hv", S, k[t])) * beta[t][:, None]
        S = S + err[:, :, None] * k[t][:, None, :]
        outs.append(B.einsum("hvk,hk->hv", S, q[t]))
    st.vars[f"kda.{layer}.S"] = S
    o = B.stack(outs, 0)                                                  # [T, H, D]
    o = rms_norm(B, o, ws.get(W(f"{prefix}.o_norm.weight")), 1e-5) * B.sigmoid(g_out)
    return lin("o_proj", o.reshape(T, P))


# -------------------------------------------------------------------- DSA
@op("dsa")
def dsa(ctx, x, layer, prefix):
    """Gated DeepSeek-style sparse attention over a multi-head latent (MLA,
    no positional encoding).  Per token the cache keeps a 576-value latent and
    a 128-value index key; a lightweight indexer picks the top-k past tokens
    and only those take part in attention."""
    B, ws, st, c = ctx.B, ctx.ws, ctx.state, ctx.cfg
    H, T = c.num_attention_heads, x.shape[0]
    dn, dr, dv, r_kv = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim, c.kv_lora_rank
    dc = c.dsa_config
    ih, idim, topk = dc.index_num_heads, dc.index_head_dim, dc.index_topk
    eps = c.rms_norm_eps
    lin = lambda name, inp=x: B.linear(inp, ws.get(W(f"{prefix}.{name}.weight")))
    nw = lambda name: ws.get(W(f"{prefix}.{name}.weight"))

    q_c = rms_norm(B, lin("q_a_proj"), nw("q_a_layernorm"), eps)
    kv = lin("kv_a_proj_with_mqa")
    kv_c = rms_norm(B, kv[:, :r_kv], nw("kv_a_layernorm"), eps)
    k_pe = kv[:, r_kv:]
    q = lin("q_b_proj", q_c).reshape(T, H, dn + dr)
    q_nope, q_pe = q[..., :dn], q[..., dn:]

    # indexer projections
    iq = lin("indexer.wq_b", q_c).reshape(T, ih, idim)
    ik = lin("indexer.wk")
    ik = ik - B.mean(ik, -1, True)
    ik = ik / B.sqrt(B.mean(ik * ik, -1, True) + eps)
    ik = ik * B.f32(nw("indexer.k_norm")) + B.f32(ws.get(W(f"{prefix}.indexer.k_norm.bias")))
    iw = lin("indexer.weights_proj")                                      # [T, ih]

    lat, idx = st.log(f"dsa.{layer}.kv", r_kv + dr), st.log(f"dsa.{layer}.ik", idim)
    past = len(lat)
    lat.append(B.cat([kv_c, k_pe], -1))
    idx.append(ik)
    S = past + T

    # absorb kv_b_proj: per head [k_nope ; v] rows over the kv latent
    wkv = B.f32(nw("kv_b_proj")).reshape(H, dn + dv, r_kv)
    w_uk, w_uv = wkv[:, :dn, :], wkv[:, dn:, :]
    ql = B.einsum("thn,hnl->thl", q_nope, w_uk)                           # query in latent space
    scale = (dn + dr) ** -0.5
    neg = B.asarray(-1e30)
    step = max(1, int(ctx.opts.get("dsa_query_batch", 64)))
    outs = []
    if S <= topk:                                                         # everything is selected
        cache = lat.read_all()
        kc, kp = cache[:, :r_kv], cache[:, r_kv:]
        for a in range(0, T, step):
            b = min(T, a + step)
            sc = (B.einsum("thl,sl->ths", ql[a:b], kc) + B.einsum("thr,sr->ths", q_pe[a:b], kp)) * scale
            mask = B.tril_mask(b - a, S, offset=past + a)
            p = B.softmax(B.where(mask[:, None, :], sc, neg), -1)
            outs.append(B.einsum("ths,sl->thl", p, kc))
    else:
        keys = idx.read_all()                                             # [S, idim]
        for a in range(0, T, step):
            b = min(T, a + step)
            rel = B.relu(B.einsum("tid,sd->tis", iq[a:b], keys))
            isc = B.einsum("ti,tis->ts", iw[a:b], rel)
            mask = B.tril_mask(b - a, S, offset=past + a)
            isc = B.where(mask, isc, neg)
            _, sel = B.topk(isc, topk, -1)                                # [t, topk]
            sel_np = np.asarray(B.to_numpy(sel), dtype=np.int64)
            uniq, inv = np.unique(sel_np, return_inverse=True)
            rows = lat.read_rows(uniq)                                    # only the selected tokens
            g = B.take(rows, inv.reshape(-1), 0).reshape(b - a, sel_np.shape[1], r_kv + dr)
            kc, kp = g[..., :r_kv], g[..., r_kv:]
            sc = (B.einsum("thl,tsl->ths", ql[a:b], kc) + B.einsum("thr,tsr->ths", q_pe[a:b], kp)) * scale
            ok = B.asarray(sel_np <= (past + a + np.arange(b - a))[:, None], "bool")
            p = B.softmax(B.where(ok[:, None, :], sc, neg), -1)
            outs.append(B.einsum("ths,tsl->thl", p, kc))
    o = B.einsum("thl,hvl->thv", outs[0] if len(outs) == 1 else B.cat(outs, 0), w_uv)
    o = o.reshape(T, H * dv) * B.sigmoid(lin("g_proj"))
    return lin("o_proj", o)


# -------------------------------------------------------------------- MoE
def route(B, logits, bias, topk, renormalize, scaling):
    """noaux_tc routing: choose by sigmoid score + correction bias, weight by
    the uncorrected score."""
    scores = B.sigmoid(B.f32(logits))
    _, ids = B.topk(scores + bias, topk, -1)
    w = B.take_along(scores, ids, -1)
    if renormalize:
        w = w / B.sum(w, -1, True)
    return (w * scaling if scaling != 1.0 else w), ids


@op("moe")
def moe(ctx, x, layer, prefix):
    """Latent MoE: router -> down-projection -> 16 routed experts (MXFP4, read
    from the SSD on demand) -> RMSNorm -> up-projection, plus shared experts."""
    B, ws, c, mem = ctx.B, ctx.ws, ctx.cfg, ctx.mem
    T = x.shape[0]
    lin = lambda name, inp=x: B.linear(inp, ws.get(W(f"{prefix}.{name}.weight")))
    beta, lbeta = c.get("activation_situ_beta") or 1.0, c.get("activation_situ_linear_beta")
    E, K = c.num_experts, c.num_experts_per_token
    lat, inter = c.routed_expert_hidden_size, c.moe_intermediate_size

    routes = ctx.cache.setdefault("routes", {})
    ahead = int(ctx.opts.get("expert_prefetch_layers", 0))
    if ahead and T <= int(ctx.opts.get("expert_prefetch_max_tokens", 4)):
        # speculative read-ahead: experts the previous token used in the next layers
        for nxt in range(layer + 1, layer + 1 + ahead):
            names = routes.get(nxt)
            if names:
                mem.prefetch(ws.blocks(names))

    bias = B.f32(ws.get(W(f"{prefix}.gate.e_score_correction_bias")))
    weights, ids = route(B, lin("gate"), bias, K, c.moe_renormalize, c.routed_scaling_factor)
    z = lin("routed_expert_down_proj")                                    # [T, lat]
    ids_np = np.asarray(B.to_numpy(ids), dtype=np.int64)
    used = np.unique(ids_np)
    first = lambda e: f"{prefix}.experts.{e}.w1.weight_packed"
    # visit experts in file order so a long prompt reads the shard sequentially
    used = sorted(used.tolist(), key=lambda e: ws.blocks([first(e)])[0].file_off)
    routes[layer] = [first(e) for e in used]

    acc = B.zeros((T, lat))
    group = max(1, int(ctx.opts.get("expert_batch", 16)))
    parts = ("w1.weight_packed", "w1.weight_scale", "w2.weight_packed", "w2.weight_scale",
             "w3.weight_packed", "w3.weight_scale")
    for a in range(0, len(used), group):
        chunk = used[a:a + group]
        if a + group < len(used):
            mem.prefetch(ws.blocks([first(e) for e in used[a + group:a + 2 * group]]))
        raws = ws.raw_many([W(f"{prefix}.experts.{e}.{p}") for e in chunk for p in parts])
        for j, e in enumerate(chunk):
            p1, s1, p2, s2, p3, s3 = (B.raw(r) for r in raws[6 * j:6 * j + 6])
            tok, slot = np.nonzero(ids_np == e)
            n_tok = len(tok)
            tok, slot = B.asarray(tok, "i64"), B.asarray(slot, "i64")
            zt = z[tok] if n_tok < T else z
            w1 = mxfp4_decode(B, p1, s1, inter, lat)
            w3 = mxfp4_decode(B, p3, s3, inter, lat)
            hmid = situ_glu(B, B.linear(zt, w1), B.linear(zt, w3), beta, lbeta)
            del w1, w3
            y = B.linear(hmid, mxfp4_decode(B, p2, s2, lat, inter))
            wt = weights[tok, slot][:, None]
            if n_tok < T:
                acc[tok] = acc[tok] + y * wt
            else:
                acc = acc + y * wt
    routed = lin("routed_expert_up_proj",
                 rms_norm(B, acc, ws.get(W(f"{prefix}.routed_expert_norm.weight")), c.rms_norm_eps)
                 if c.get("latent_moe_use_norm") else acc)
    shared = lin("shared_experts.down_proj",
                 situ_glu(B, lin("shared_experts.gate_proj"), lin("shared_experts.up_proj"), beta, lbeta))
    return routed + shared
