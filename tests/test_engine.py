import numpy as np
import pytest

from conftest import has_torch
from inferencese.runtime.engine import Engine

BACKENDS = ["numpy"] + (["torch"] if has_torch() else [])


def engine(profile, model, backend="numpy", ram="64MiB", gpu=None, chunk=64, **extra):
    ov = {"runtime": {"backend": backend, "device": "cpu", "prefill_chunk": chunk, "max_context": 0},
          "memory": {"tiers": {"ram": {"capacity": ram},
                               "gpu": {"simulate": bool(gpu), "capacity": gpu or 0},
                               "ssd": {"direct_io": False}},
                     "warmup": []}}
    for k, v in extra.items():
        ov["memory"][k] = v
    return Engine(profile, model, overrides=ov, log=None)


def reference_llama(model_dir, tokens):
    """Straightforward dense numpy implementation, independent of the VM."""
    import json
    from inferencese.formats import decode, open_model
    cat = open_model(model_dir)
    cfg = json.load(open(model_dir + "/config.json"))
    raw = open(cat.files[0], "rb").read()
    W = lambda n: decode(np.frombuffer(raw[cat[n].offset:cat[n].offset + cat[n].nbytes], np.uint8),
                         cat[n].dtype, cat[n].shape).astype(np.float64)
    norm = lambda x, w: x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg["rms_norm_eps"]) * w
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    d, T = D // H, len(tokens)
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, d, 2) / d)
    ang = np.arange(T)[:, None] * inv[None]
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    rope = lambda v: np.concatenate([v[..., :d // 2] * cos - v[..., d // 2:] * sin,
                                     v[..., d // 2:] * cos + v[..., :d // 2] * sin], -1)
    x = W("model.embed_tokens.weight")[tokens]
    for i in range(cfg["num_hidden_layers"]):
        L = f"model.layers.{i}"
        h = norm(x, W(f"{L}.input_layernorm.weight"))
        q = rope((h @ W(f"{L}.self_attn.q_proj.weight").T).reshape(T, H, d))
        k = rope((h @ W(f"{L}.self_attn.k_proj.weight").T).reshape(T, KV, d))
        v = (h @ W(f"{L}.self_attn.v_proj.weight").T).reshape(T, KV, d)
        k, v = np.repeat(k, H // KV, 1), np.repeat(v, H // KV, 1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(d)
        att = np.where(np.tril(np.ones((T, T), bool))[None], att, -np.inf)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        x = x + np.einsum("hts,shd->thd", att, v).reshape(T, D) @ W(f"{L}.self_attn.o_proj.weight").T
        h = norm(x, W(f"{L}.post_attention_layernorm.weight"))
        g = h @ W(f"{L}.mlp.gate_proj.weight").T
        x = x + (g / (1 + np.exp(-g)) * (h @ W(f"{L}.mlp.up_proj.weight").T)) @ W(f"{L}.mlp.down_proj.weight").T
    return norm(x[-1], W("model.norm.weight")) @ W("lm_head.weight").T


TOKENS = list(np.random.default_rng(3).integers(0, 250, 37))


@pytest.mark.parametrize("backend", BACKENDS)
def test_llama_matches_independent_reference(tiny_llama, backend):
    e = engine("generic-llama", tiny_llama, backend)
    got = e.feed(TOKENS)
    ref = reference_llama(tiny_llama, TOKENS)
    assert np.abs(got - ref).max() < 2e-2 * np.abs(ref).max()     # float16 KV cache
    assert got.argmax() == ref.argmax()
    e.close()


@pytest.mark.parametrize("profile,fixture", [("generic-llama", "tiny_llama"), ("myKimi3", "tiny_kimi")])
@pytest.mark.parametrize("backend", BACKENDS)
def test_paging_and_chunking_do_not_change_results(profile, fixture, backend, request):
    """Same logits whether everything is cached or almost nothing is, whether
    the prompt is processed at once, in chunks, or token by token."""
    model = request.getfixturevalue(fixture)
    base = None
    for ram, gpu, chunk in [("64MiB", None, 64), ("120KiB", None, 64), ("90KiB", "60KiB", 5),
                            ("64MiB", None, 1), ("120KiB", None, 1)]:
        e = engine(profile, model, backend, ram, gpu, chunk)
        logits = e.feed(TOKENS)
        st = e.stats()["memory"]
        assert np.isfinite(logits).all()
        assert st["ram"]["used"] <= st["ram"]["capacity"] and st["gpu"]["used"] <= st["gpu"]["capacity"]
        if ram != "64MiB":
            assert st["totals"].get("evict_ram", 0) > 0           # paging really happened
        if base is None:
            base = logits
        assert np.abs(logits - base).max() < 5e-3 * max(1.0, np.abs(base).max())
        e.close()


@pytest.mark.skipif(not has_torch(), reason="torch not installed")
@pytest.mark.parametrize("profile,fixture", [("generic-llama", "tiny_llama"), ("myKimi3", "tiny_kimi")])
def test_backends_agree(profile, fixture, request):
    model = request.getfixturevalue(fixture)
    a, b = engine(profile, model, "numpy"), engine(profile, model, "torch")
    la, lb = a.feed(TOKENS), b.feed(TOKENS)
    assert np.abs(la - lb).max() < 5e-3 * max(1.0, np.abs(la).max())
    a.close(), b.close()


@pytest.mark.parametrize("profile,fixture", [("generic-llama", "tiny_llama"), ("myKimi3", "tiny_kimi")])
def test_prefix_reuse_and_rewind(profile, fixture, request):
    model = request.getfixturevalue(fixture)
    e = engine(profile, model, chunk=8)
    first, longer = TOKENS[:20], TOKENS[:30]
    branch = TOKENS[:20] + [7, 8, 9]
    e.feed(first)
    e.feed(longer)
    assert e.timing["prefill_tokens"] == 30                       # only the 10 new tokens
    got = e.feed(branch)                                          # diverges after 20 tokens
    fresh = engine(profile, model, chunk=8)
    assert np.abs(got - fresh.feed(branch)).max() < 5e-3
    assert e.state.tokens == branch
    again = e.feed(branch)                                        # identical prompt: no work
    assert np.array_equal(again, got)
    e.close(), fresh.close()


def test_generation_is_deterministic_and_stops(tiny_llama):
    e = engine("generic-llama", tiny_llama)
    a = list(e.generate(TOKENS, max_new_tokens=6, sampling={"temperature": 0}))
    e.reset()
    b = list(e.generate(TOKENS, max_new_tokens=6, sampling={"temperature": 0}))
    assert a == b and len(a) == 6
    e.reset()
    c = list(e.generate(TOKENS, max_new_tokens=6, sampling={"temperature": 0}, stop_ids=[a[2]]))
    assert c == a[:2]
    e.reset()
    s1 = list(e.generate(TOKENS, max_new_tokens=6, sampling={"temperature": 1.0, "seed": 5}))
    e.reset()
    s2 = list(e.generate(TOKENS, max_new_tokens=6, sampling={"temperature": 1.0, "seed": 5}))
    assert s1 == s2
    e.close()


def test_context_pages_out_to_swap(tiny_llama):
    """A context whose KV cache exceeds the RAM tier keeps working."""
    e = engine("generic-llama", tiny_llama, ram="100KiB", chunk=16,
               state={"tier": "ram", "rows_per_block": 8})
    toks = list(np.random.default_rng(0).integers(0, 250, 150))
    got = e.feed(toks)
    assert e.mem.swap is not None and e.mem.swap.bytes_written > 0
    assert e.stats()["memory"]["classes"]["state"]["swap_out_bytes"] > 0
    big = engine("generic-llama", tiny_llama, chunk=16)
    assert np.abs(got - big.feed(toks)).max() < 5e-3
    e.close(), big.close()


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        engine("generic-llama", str(tmp_path / "nope"))
