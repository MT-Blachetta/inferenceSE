"""Synthetic models for tests: same tensor names, dtypes and packing as the
real checkpoints, scaled down to kilobytes and filled with random weights."""
from __future__ import annotations

import json
import os

import numpy as np

from ..formats.catalog import write_safetensors
from ..formats.dtypes import f32_to_bf16

TINY_KIMI = {
    "vocab_size": 300, "hidden_size": 64, "num_hidden_layers": 6, "num_attention_heads": 4,
    "intermediate_size": 96, "hidden_act": "situ", "rms_norm_eps": 1e-5,
    "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
    "attn_res_block_size": 2, "first_k_dense_replace": 1,
    "num_experts": 12, "num_experts_per_token": 3, "num_shared_experts": 2,
    "moe_intermediate_size": 64, "routed_expert_hidden_size": 32, "latent_moe_use_norm": True,
    "moe_renormalize": True, "routed_scaling_factor": 1.0, "moe_router_activation_func": "sigmoid",
    "q_lora_rank": 24, "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
    "v_head_dim": 16, "mla_use_nope": True, "mla_use_output_gate": True,
    "linear_attn_config": {"kda_layers": [1, 2, 4], "full_attn_layers": [3, 5, 6],
                           "num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                           "gate_lower_bound": -5.0, "use_full_rank_gate": True},
    "dsa_config": {"dsa_layers": [3, 5, 6], "index_topk": 6, "index_head_dim": 8,
                   "index_num_heads": 2, "index_rope_dim": 0, "index_k_norm_type": "layernorm"},
    "bos_token_id": 1, "eos_token_id": 2,
}


def _bf16(rng, shape, std=None):
    std = std if std is not None else 1.0 / np.sqrt(shape[-1])
    return ("BF16", shape, f32_to_bf16(rng.normal(0, std, shape).astype(np.float32)).tobytes())


def _f32(rng, shape, std=0.5, mean=0.0):
    return ("F32", shape, rng.normal(mean, std, shape).astype(np.float32).tobytes())


def _mxfp4(rng, out, inp):
    """Random MXFP4 matrix: packed nibbles + E8M0 scales (exponent around 2**-3)."""
    packed = rng.integers(0, 256, (out, inp // 2), dtype=np.uint8)
    scale = rng.integers(122, 126, (out, inp // 32), dtype=np.uint8)
    return ("U8", (out, inp // 2), packed.tobytes()), ("U8", (out, inp // 32), scale.tobytes())


def make_tiny_kimi(out_dir: str, cfg: dict | None = None, seed: int = 0) -> dict:
    """Write a miniature Kimi-K3.1-shaped checkpoint (one shard per layer)."""
    c = dict(TINY_KIMI, **(cfg or {}))
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    D, H = c["hidden_size"], c["num_attention_heads"]
    lc, dc = c["linear_attn_config"], c["dsa_config"]
    kh, kd = lc["num_heads"], lc["head_dim"]
    P = kh * kd
    n_shards = c["num_hidden_layers"] + 1
    M = "language_model.model"
    for i in range(c["num_hidden_layers"]):
        t, L = {}, f"{M}.layers.{i}"
        for n in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            t[f"{L}.{n}.weight"] = _bf16(rng, (D,), 0.1)
            t[f"{L}.{n}.weight"] = ("BF16", (D,), f32_to_bf16(
                (1 + rng.normal(0, 0.1, D)).astype(np.float32)).tobytes())
        t[f"{L}.self_attention_res_proj.weight"] = _bf16(rng, (1, D), 0.5)
        t[f"{L}.mlp_res_proj.weight"] = _bf16(rng, (1, D), 0.5)
        A = f"{L}.self_attn"
        if (i + 1) in lc["kda_layers"]:
            for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
                t[f"{A}.{n}.weight"] = _bf16(rng, (P, D))
            t[f"{A}.f_a_proj.weight"] = _bf16(rng, (kd, D))
            t[f"{A}.f_b_proj.weight"] = _bf16(rng, (P, kd))
            t[f"{A}.b_proj.weight"] = _bf16(rng, (kh, D))
            t[f"{A}.o_proj.weight"] = _bf16(rng, (D, P))
            t[f"{A}.A_log"] = _f32(rng, (kd,), 0.3)
            t[f"{A}.dt_bias"] = _f32(rng, (P,), 0.5)
            t[f"{A}.o_norm.weight"] = _f32(rng, (kd,), 0.1, 1.0)
            for n in "qkv":
                t[f"{A}.{n}_conv1d.weight"] = _f32(rng, (P, 1, lc["short_conv_kernel_size"]), 0.5)
        else:
            qr, kr = c["q_lora_rank"], c["kv_lora_rank"]
            dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
            t[f"{A}.q_a_proj.weight"] = _bf16(rng, (qr, D))
            t[f"{A}.q_a_layernorm.weight"] = _bf16(rng, (qr,), 1.0)
            t[f"{A}.q_b_proj.weight"] = _bf16(rng, (H * (dn + dr), qr))
            t[f"{A}.kv_a_proj_with_mqa.weight"] = _bf16(rng, (kr + dr, D))
            t[f"{A}.kv_a_layernorm.weight"] = _bf16(rng, (kr,), 1.0)
            t[f"{A}.kv_b_proj.weight"] = _bf16(rng, (H * (dn + dv), kr))
            t[f"{A}.g_proj.weight"] = _bf16(rng, (H * dv, D))
            t[f"{A}.o_proj.weight"] = _bf16(rng, (D, H * dv))
            ih, idim = dc["index_num_heads"], dc["index_head_dim"]
            t[f"{A}.indexer.wq_b.weight"] = _bf16(rng, (ih * idim, qr))
            t[f"{A}.indexer.wk.weight"] = _bf16(rng, (idim, D))
            t[f"{A}.indexer.weights_proj.weight"] = _bf16(rng, (ih, D))
            t[f"{A}.indexer.k_norm.weight"] = _bf16(rng, (idim,), 1.0)
            t[f"{A}.indexer.k_norm.bias"] = _bf16(rng, (idim,), 0.2)
        if i < c["first_k_dense_replace"]:
            I = c["intermediate_size"]
            t[f"{L}.mlp.gate_proj.weight"] = _bf16(rng, (I, D))
            t[f"{L}.mlp.up_proj.weight"] = _bf16(rng, (I, D))
            t[f"{L}.mlp.down_proj.weight"] = _bf16(rng, (D, I))
        else:
            E, lat, inter = c["num_experts"], c["routed_expert_hidden_size"], c["moe_intermediate_size"]
            sh = inter * c["num_shared_experts"]
            X = f"{L}.block_sparse_moe"
            t[f"{X}.gate.e_score_correction_bias"] = _f32(rng, (E,), 0.2)
            t[f"{X}.gate.weight"] = _bf16(rng, (E, D), 0.5)
            t[f"{X}.routed_expert_down_proj.weight"] = _bf16(rng, (lat, D))
            t[f"{X}.routed_expert_norm.weight"] = _bf16(rng, (lat,), 1.0)
            t[f"{X}.routed_expert_up_proj.weight"] = _bf16(rng, (D, lat))
            t[f"{X}.shared_experts.gate_proj.weight"] = _bf16(rng, (sh, D))
            t[f"{X}.shared_experts.up_proj.weight"] = _bf16(rng, (sh, D))
            t[f"{X}.shared_experts.down_proj.weight"] = _bf16(rng, (D, sh))
            for e in sorted(range(E), key=str):          # checkpoint order is lexicographic
                for w, (o, n) in (("w1", (inter, lat)), ("w2", (lat, inter)), ("w3", (inter, lat))):
                    pk, sc = _mxfp4(rng, o, n)
                    t[f"{X}.experts.{e}.{w}.weight_packed"] = pk
                    t[f"{X}.experts.{e}.{w}.weight_scale"] = sc
        write_safetensors(os.path.join(out_dir, f"model-{i + 1:05d}-of-{n_shards:06d}.safetensors"), t)
    t = {
        "language_model.lm_head.weight": _bf16(rng, (c["vocab_size"], D)),
        f"{M}.embed_tokens.weight": _bf16(rng, (c["vocab_size"], D), 1.0),
        f"{M}.norm.weight": _bf16(rng, (D,), 1.0),
        f"{M}.output_attn_res_norm.weight": _bf16(rng, (D,), 1.0),
        f"{M}.output_attn_res_proj.weight": _bf16(rng, (1, D), 0.5),
    }
    write_safetensors(os.path.join(out_dir, f"model-{n_shards:05d}-of-{n_shards:06d}.safetensors"), t)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"model_type": "kimi_k31", "text_config": c}, f, indent=1)
    return c


TINY_LLAMA = {"vocab_size": 260, "hidden_size": 32, "num_hidden_layers": 3,
              "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 64,
              "rms_norm_eps": 1e-5, "rope_theta": 10000.0}


def make_tiny_llama(out_dir: str, cfg: dict | None = None, seed: int = 0, dtype: str = "F32") -> dict:
    """Random LLaMA-shaped checkpoint (Hugging Face tensor names)."""
    c = dict(TINY_LLAMA, **(cfg or {}))
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    D, I = c["hidden_size"], c["intermediate_size"]
    hd = D // c["num_attention_heads"]
    kv = hd * c["num_key_value_heads"]

    def t(shape, std=None, mean=0.0):
        a = rng.normal(mean, std if std is not None else 1.0 / np.sqrt(shape[-1]), shape).astype(np.float32)
        return ("BF16", shape, f32_to_bf16(a).tobytes()) if dtype == "BF16" else ("F32", shape, a.tobytes())

    w = {"model.embed_tokens.weight": t((c["vocab_size"], D), 1.0),
         "model.norm.weight": t((D,), 0.1, 1.0), "lm_head.weight": t((c["vocab_size"], D))}
    for i in range(c["num_hidden_layers"]):
        L = f"model.layers.{i}"
        w[f"{L}.input_layernorm.weight"] = t((D,), 0.1, 1.0)
        w[f"{L}.post_attention_layernorm.weight"] = t((D,), 0.1, 1.0)
        w[f"{L}.self_attn.q_proj.weight"] = t((D, D))
        w[f"{L}.self_attn.k_proj.weight"] = t((kv, D))
        w[f"{L}.self_attn.v_proj.weight"] = t((kv, D))
        w[f"{L}.self_attn.o_proj.weight"] = t((D, D))
        w[f"{L}.mlp.gate_proj.weight"] = t((I, D))
        w[f"{L}.mlp.up_proj.weight"] = t((I, D))
        w[f"{L}.mlp.down_proj.weight"] = t((D, I))
    write_safetensors(os.path.join(out_dir, "model.safetensors"), w)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(c, f, indent=1)
    return c
