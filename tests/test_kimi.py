"""myKimi3 profile: kernels against scalar re-derivations, profile integrity."""
import importlib.util
import os

import numpy as np
import pytest

from inferencese.backend import NumpyBackend
from inferencese.profile.loader import Profile, find_profile
from inferencese.vm import il

PROFILE = find_profile("myKimi3")


def _ops():
    reg = Profile("myKimi3").ops()
    name = "inferencese_profile_myKimi3_ops"
    import sys
    return reg, sys.modules[name]


def test_mxfp4_decode_matches_bitwise_definition():
    _, mod = _ops()
    rng = np.random.default_rng(0)
    out_f, in_f = 5, 64
    packed = rng.integers(0, 256, (out_f, in_f // 2), dtype=np.uint8)
    scale = rng.integers(110, 135, (out_f, in_f // 32), dtype=np.uint8)
    got = mod.mxfp4_decode(NumpyBackend(), packed, scale, out_f, in_f)
    e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]            # OCP MX E2M1 magnitudes
    for r in range(out_f):
        for c in range(in_f):
            byte = int(packed[r, c // 2])
            nib = (byte >> 4) if c % 2 else (byte & 0xF)        # low nibble holds the even element
            val = (-1.0 if nib & 8 else 1.0) * e2m1[nib & 7] * 2.0 ** (int(scale[r, c // 32]) - 127)
            assert got[r, c] == np.float32(val)


def test_mxfp4_decode_torch_equals_numpy():
    torch = pytest.importorskip("torch")
    from inferencese.backend.torch_backend import TorchBackend
    _, mod = _ops()
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, (6, 48), dtype=np.uint8)
    scale = rng.integers(120, 130, (6, 3), dtype=np.uint8)
    a = mod.mxfp4_decode(NumpyBackend(), packed, scale, 6, 96)
    b = mod.mxfp4_decode(TorchBackend(device="cpu"), torch.from_numpy(packed), torch.from_numpy(scale), 6, 96)
    assert np.array_equal(a, b.numpy())


def test_attn_res_matches_reference_kernel_semantics():
    """Transcription of the per-row loop of vLLM's _attn_res_kernel."""
    reg, _ = _ops()
    rng = np.random.default_rng(0)
    T, D, eps = 3, 16, 1e-5
    prefix = rng.normal(size=(T, D)).astype(np.float32)
    delta = rng.normal(size=(T, D)).astype(np.float32)
    blocks = [rng.normal(size=(T, D)).astype(np.float32) for _ in range(2)]
    norm_w, qk_w, out_w = (rng.normal(size=D).astype(np.float32) for _ in range(3))

    class WS:
        def get(self, n):
            return {"norm": norm_w, "proj": qk_w.reshape(1, D), "out": out_w}[n]

    class Ctx:
        B, ws = NumpyBackend(), WS()

    store = list(blocks) + [None]
    mixed, new_prefix = reg["attn_res"](Ctx, prefix, delta, store, "norm", "proj", out_norm="out",
                                        num_blocks=2, write=2, eps=eps)
    upd = prefix + delta
    assert np.allclose(new_prefix, upd) and np.allclose(store[2], upd)
    for t in range(T):
        srcs = [blocks[0][t], blocks[1][t], upd[t]]
        logits = [float((v * norm_w * qk_w).sum() / np.sqrt((v * v).mean() + eps)) for v in srcs]
        w = np.exp(np.array(logits) - max(logits))
        w /= w.sum()
        mix = sum(wi * v for wi, v in zip(w, srcs))
        ref = mix / np.sqrt((mix * mix).mean() + eps) * out_w
        assert np.allclose(mixed[t], ref, atol=1e-5)
    only, _ = reg["attn_res"](Ctx, prefix, None, [], "norm", "proj", num_blocks=0)
    assert np.array_equal(only, prefix)                         # single source: softmax weight is 1


def test_router_selects_by_corrected_score_and_weights_by_raw_score():
    _, mod = _ops()
    B = NumpyBackend()
    logits = np.array([[2.0, 1.0, 0.0, -1.0, 3.0]], np.float32)
    bias = np.array([0.0, 0.0, 5.0, 0.0, -5.0], np.float32)      # promotes expert 2, demotes 4
    w, ids = mod.route(B, logits, bias, 2, True, 1.0)
    assert set(ids[0].tolist()) == {2, 0}
    sig = 1 / (1 + np.exp(-logits[0]))
    expect = {i: sig[i] / (sig[0] + sig[2]) for i in (0, 2)}
    for i, wi in zip(ids[0].tolist(), w[0].tolist()):
        assert abs(wi - expect[i]) < 1e-6


def test_profile_is_complete_and_consistent():
    p = Profile("myKimi3")
    assert p.name == "myKimi3" and p.model["id"] == "moonshotai/Kimi-K3.1-Pro"
    for f in ("profile.yaml", "hardware.yaml", "memory.yaml", "model.seil", "ops.py",
              "tokenizer.py", "facts/config.json", "facts/catalog.json.xz"):
        assert os.path.exists(p.file(f)), f
    program, _ = p.program()
    ops = p.ops()
    used = {n.name for f in program.funcs.values() for n in _walk(f.body) if isinstance(n, il.Op)}
    assert used <= set(ops)
    cfg = p.config(None)
    assert cfg["num_hidden_layers"] == 93 and cfg["num_experts"] == 896
    assert len(cfg["linear_attn_config"]["kda_layers"]) == 69
    assert len(cfg["dsa_config"]["dsa_layers"]) == 24


def _walk(body):
    for n in body:
        yield n
        if isinstance(n, il.For):
            yield from _walk(n.body)
        elif isinstance(n, il.If):
            for _, b in n.arms:
                yield from _walk(b)


def test_real_tensor_layout_is_covered_by_the_memory_rules():
    """Plan the real 1.56 TB checkpoint from its tensor index (no weights)."""
    from inferencese.tools.plan import plan
    p = plan("myKimi3")
    by = {r["class"]: r for r in p["classes"]}
    assert p["tensors"] == 497_340 and p["model_bytes"] == 1_561_217_377_280
    assert by["expert"]["blocks"] == 92 * 896                    # one block per expert
    assert by["expert"]["bytes"] == 92 * 896 * 17_547_264
    dense = by["dense"]["bytes"] + by["lm_head"]["bytes"]
    assert p["vram_cache"] < dense < p["vram_cache"] + p["ram_cache"]   # spills into RAM, not SSD
    assert by["dense"]["on_ssd_only"] == 0 and by["lm_head"]["in_vram"] == by["lm_head"]["bytes"]
    assert by["vision"]["in_vram"] == by["vision"]["in_ram"] == 0
    assert p["vram_used"] <= p["vram_cache"] and p["ram_used"] <= p["ram_cache"]


def test_every_tensor_the_program_needs_exists_with_the_right_shape():
    """Names and shapes the Kimi ops derive from the config, checked against
    the real checkpoint index for all 93 layers."""
    from inferencese.formats.catalog import Catalog
    p = Profile("myKimi3")
    cat, c = Catalog.load_manifest(p.manifest()), p.config(None)
    D, H = c["hidden_size"], c["num_attention_heads"]
    lc, dc = c["linear_attn_config"], c["dsa_config"]
    P = lc["num_heads"] * lc["head_dim"]
    lat, inter = c["routed_expert_hidden_size"], c["moe_intermediate_size"]
    want = {"language_model.lm_head.weight": (c["vocab_size"], D),
            "language_model.model.embed_tokens.weight": (c["vocab_size"], D),
            "language_model.model.norm.weight": (D,),
            "language_model.model.output_attn_res_norm.weight": (D,),
            "language_model.model.output_attn_res_proj.weight": (1, D)}
    for i in range(c["num_hidden_layers"]):
        L = f"language_model.model.layers.{i}"
        for n in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            want[f"{L}.{n}.weight"] = (D,)
        want[f"{L}.self_attention_res_proj.weight"] = want[f"{L}.mlp_res_proj.weight"] = (1, D)
        A = f"{L}.self_attn"
        if (i + 1) in lc["kda_layers"]:
            for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
                want[f"{A}.{n}.weight"] = (P, D)
            want.update({f"{A}.f_a_proj.weight": (lc["head_dim"], D), f"{A}.f_b_proj.weight": (P, lc["head_dim"]),
                         f"{A}.b_proj.weight": (lc["num_heads"], D), f"{A}.o_proj.weight": (D, P),
                         f"{A}.dt_bias": (P,), f"{A}.o_norm.weight": (lc["head_dim"],)})
            for n in "qkv":
                want[f"{A}.{n}_conv1d.weight"] = (P, 1, lc["short_conv_kernel_size"])
            assert cat[f"{A}.A_log"].shape[0] >= lc["num_heads"]
        else:
            assert (i + 1) in dc["dsa_layers"]
            qr, kr = c["q_lora_rank"], c["kv_lora_rank"]
            dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
            want.update({f"{A}.q_a_proj.weight": (qr, D), f"{A}.q_a_layernorm.weight": (qr,),
                         f"{A}.q_b_proj.weight": (H * (dn + dr), qr),
                         f"{A}.kv_a_proj_with_mqa.weight": (kr + dr, D), f"{A}.kv_a_layernorm.weight": (kr,),
                         f"{A}.kv_b_proj.weight": (H * (dn + dv), kr), f"{A}.g_proj.weight": (H * dv, D),
                         f"{A}.o_proj.weight": (D, H * dv),
                         f"{A}.indexer.wq_b.weight": (dc["index_num_heads"] * dc["index_head_dim"], qr),
                         f"{A}.indexer.wk.weight": (dc["index_head_dim"], D),
                         f"{A}.indexer.weights_proj.weight": (dc["index_num_heads"], D),
                         f"{A}.indexer.k_norm.weight": (dc["index_head_dim"],),
                         f"{A}.indexer.k_norm.bias": (dc["index_head_dim"],)})
        if i < c["first_k_dense_replace"]:
            I = c["intermediate_size"]
            want.update({f"{L}.mlp.gate_proj.weight": (I, D), f"{L}.mlp.up_proj.weight": (I, D),
                         f"{L}.mlp.down_proj.weight": (D, I)})
        else:
            X, sh = f"{L}.block_sparse_moe", inter * c["num_shared_experts"]
            want.update({f"{X}.gate.weight": (c["num_experts"], D),
                         f"{X}.gate.e_score_correction_bias": (c["num_experts"],),
                         f"{X}.routed_expert_down_proj.weight": (lat, D),
                         f"{X}.routed_expert_norm.weight": (lat,),
                         f"{X}.routed_expert_up_proj.weight": (D, lat),
                         f"{X}.shared_experts.gate_proj.weight": (sh, D),
                         f"{X}.shared_experts.up_proj.weight": (sh, D),
                         f"{X}.shared_experts.down_proj.weight": (D, sh)})
            for e in (0, 1, 449, c["num_experts"] - 1):
                for w, (o, n) in (("w1", (inter, lat)), ("w2", (lat, inter)), ("w3", (inter, lat))):
                    want[f"{X}.experts.{e}.{w}.weight_packed"] = (o, n // 2)
                    want[f"{X}.experts.{e}.{w}.weight_scale"] = (o, n // 32)
    for name, shape in want.items():
        assert cat[name].shape == shape, name
    text = [n for n in cat.tensors if n.startswith("language_model.")]
    experts = [n for n in text if ".experts." in n]
    assert len(experts) == 92 * 896 * 6
    # nothing in the text model is left unused by the program
    assert len(text) - len(experts) == len([n for n in want if ".experts." not in n]) + 69


def test_kimi_tokenizer_chat_format(tmp_path):
    pytest.importorskip("tiktoken")
    import base64
    with open(tmp_path / "tiktoken.model", "wb") as f:           # minimal byte-level vocabulary
        for i in range(256):
            f.write(base64.b64encode(bytes([i])) + b" " + str(i).encode() + b"\n")
    import json
    names = {256: "<|end_of_msg|>", 257: "<|open|>", 258: "<|close|>", 259: "<|sep|>"}
    with open(tmp_path / "tokenizer_config.json", "w") as f:
        json.dump({"added_tokens_decoder": {str(k): {"content": v} for k, v in names.items()}}, f)
    spec = importlib.util.spec_from_file_location("ktok", os.path.join(PROFILE, "tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = mod.KimiTokenizer(str(tmp_path), {"thinking_effort": None})
    ids = tok.apply_chat([{"role": "user", "content": "hi <|sep|> there"}])
    text = tok.decode(ids)
    assert text == ('<|open|>message role="user"<|sep|>hi <|sep|> there<|close|>message<|sep|>'
                    '<|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>think<|sep|>')
    # control markers are single special tokens, user text can never forge them
    assert sum(1 for i in ids if i >= 256) == 5 + 4              # user msg (5) + generation prompt (4)
    assert tok.encode("<|sep|>") == list(b"<|sep|>")
    f = mod.ChannelFilter(thinking=True)
    stream = "plan<|close|>think<|sep|><|open|>response<|sep|>42<|close|>response<|sep|><|close|>message<|sep|>tail"
    got = [x for i in range(0, len(stream), 3) for x in f.push(stream[i:i + 3])]
    assert "".join(t for c, t in got if c == "reasoning") == "plan"
    assert "".join(t for c, t in got if c == "content") == "42"
    assert f.done
