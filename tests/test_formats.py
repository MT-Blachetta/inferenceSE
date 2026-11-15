import struct

import numpy as np
import pytest

from inferencese.formats import decode, open_model
from inferencese.formats.catalog import Catalog, write_safetensors
from inferencese.formats.dtypes import bf16_to_f32, dequant_ggml, f32_to_bf16, nbytes_of


def test_bf16_roundtrip():
    x = np.random.default_rng(0).normal(0, 3, 1000).astype(np.float32)
    y = bf16_to_f32(f32_to_bf16(x))
    assert np.allclose(x, y, rtol=1 / 128)
    assert np.array_equal(bf16_to_f32(f32_to_bf16(y)), y)        # idempotent


def test_safetensors_catalog_and_manifest(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(5, dtype=np.int64)
    p = tmp_path / "m.safetensors"
    write_safetensors(str(p), {"a": ("F32", a.shape, a.tobytes()), "b": ("I64", b.shape, b.tobytes())})
    cat = open_model(str(tmp_path))
    assert len(cat) == 2 and cat["a"].shape == (3, 4)
    raw = open(p, "rb").read()
    e = cat["a"]
    assert np.array_equal(decode(np.frombuffer(raw[e.offset:e.offset + e.nbytes], np.uint8), e.dtype, e.shape), a)
    with pytest.raises(KeyError, match="not in the model"):
        cat["missing"]
    man = tmp_path / "cat.json.xz"
    cat.save_manifest(str(man))
    again = Catalog.load_manifest(str(man), str(tmp_path))
    assert again["b"].offset == cat["b"].offset and again["b"].path == cat["b"].path


def _gguf(path, tensors, meta):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, (t, v) in meta.items():
        out += s(k) + struct.pack("<I", t) + (s(v) if t == 8 else struct.pack("<I", v))
    off, blobs = 0, []
    for name, (gtype, dims, data) in tensors.items():
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", gtype, off)
        blobs.append((off, data))
        off += (len(data) + 31) // 32 * 32
    out += b"\0" * (-len(out) % 32)
    base = len(out)
    for o, data in blobs:
        out += b"\0" * (base + o - len(out)) + data
    open(path, "wb").write(out)


def test_gguf_reader(tmp_path):
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    q = np.zeros(34, np.uint8)
    q[:2] = np.frombuffer(np.float16(0.5).tobytes(), np.uint8)
    q[2:] = np.arange(32, dtype=np.int8).view(np.uint8)
    p = tmp_path / "m.gguf"
    _gguf(str(p), {"blk.0.w": (0, (3, 2), w.tobytes()), "blk.0.q": (8, (32,), q.tobytes())},
          {"general.architecture": (8, "llama"), "general.alignment": (4, 32)})
    cat = open_model(str(p))
    assert cat.metadata["gguf"]["general.architecture"] == "llama"
    assert cat["blk.0.w"].shape == (2, 3)                 # GGUF stores dims innermost-first
    raw = open(p, "rb").read()
    e = cat["blk.0.w"]
    assert np.array_equal(decode(np.frombuffer(raw[e.offset:e.offset + e.nbytes], np.uint8), "F32", e.shape), w)
    e = cat["blk.0.q"]
    assert e.dtype == "Q8_0" and e.nbytes == 34
    got = decode(np.frombuffer(raw[e.offset:e.offset + e.nbytes], np.uint8), "Q8_0", e.shape)
    assert np.allclose(got, 0.5 * np.arange(32))


@pytest.mark.parametrize("qtype", ["Q8_0", "Q4_0", "Q4_K", "Q6_K"])
def test_ggml_dequant_matches_reference(qtype):
    """Our block decoders against the reference decoder of the gguf package."""
    gguf = pytest.importorskip("gguf")
    from inferencese.formats.dtypes import GGML_BLOCK
    per, size = GGML_BLOCK[qtype]
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, (16, size), dtype=np.uint8)
    scale = rng.uniform(0.01, 0.2, (16, 2)).astype(np.float16)        # finite block scales
    if qtype in ("Q8_0", "Q4_0"):
        blocks[:, 0:2] = scale[:, :1].view(np.uint8)
    elif qtype == "Q4_K":
        blocks[:, 0:4] = scale.view(np.uint8)
    else:
        blocks[:, 208:210] = scale[:, :1].view(np.uint8)
    ref = gguf.quants.dequantize(blocks.reshape(-1), gguf.GGMLQuantizationType[qtype])
    mine = dequant_ggml(blocks.reshape(-1), qtype)
    assert mine.shape == ref.reshape(-1).shape == (16 * per,)
    assert np.allclose(mine, ref.reshape(-1), rtol=1e-5, atol=1e-6)
    assert nbytes_of(qtype, (16, per)) == blocks.size
