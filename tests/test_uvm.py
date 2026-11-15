import os

import numpy as np
import pytest

from inferencese.uvm import T_GPU, T_RAM, T_SSD, BlockClass, UnifiedMemory


@pytest.fixture
def blob(tmp_path):
    data = np.random.default_rng(0).integers(0, 256, 400_000, dtype=np.uint8)
    path = tmp_path / "weights.bin"
    data.tofile(path)
    return str(path), data


def make(tmp_path, **kw):
    kw.setdefault("swap_path", str(tmp_path / "swap.bin"))
    return UnifiedMemory(**kw)


def test_lookup_order_and_fault(tmp_path, blob):
    path, data = blob
    m = make(tmp_path, gpu_bytes=20_000, ram_bytes=20_000)
    k = m.add_class(BlockClass("dense", tier=T_GPU, planned=True))
    b = m.map_file(path, 123, 10_000, "a", k)
    assert b.tier == T_SSD                               # mapped, nothing read yet
    assert m.files.bytes_read == 0
    got = m.acquire(b, T_GPU)
    assert np.array_equal(got, data[123:10_123])
    assert b.tier == T_GPU and k.stats["fault"] == 1
    m.acquire(b, T_GPU)
    assert k.stats["hit_gpu"] == 1 and m.files.bytes_read == 10_000
    m.close()


def test_eviction_respects_capacity_and_content(tmp_path, blob):
    path, data = blob
    m = make(tmp_path, gpu_bytes=0, ram_bytes=35_000, policy="lru")
    blocks = [m.map_file(path, i * 10_000, 10_000, f"b{i}") for i in range(20)]
    rng = np.random.default_rng(1)
    for i in rng.integers(0, 20, 300):
        b = blocks[i]
        assert np.array_equal(m.acquire(b), data[b.file_off:b.file_off + b.nbytes])
        assert m.used[T_RAM] <= 35_000
    assert m.default_class.stats["evict_ram"] > 0
    m.close()


def test_gpu_spills_to_ram_then_ssd(tmp_path, blob):
    path, _ = blob
    m = make(tmp_path, gpu_bytes=20_000, ram_bytes=20_000)
    k = m.add_class(BlockClass("dense", tier=T_GPU, planned=True))
    blocks = [m.map_file(path, i * 10_000, 10_000, f"d{i}", k) for i in range(6)]
    for _ in range(3):
        for b in blocks:
            m.acquire(b, T_GPU)
        m.end_step()
    assert m.used[T_GPU] == 20_000 and m.used[T_RAM] == 20_000
    assert sorted(b.tier for b in blocks) == [0, 0, 1, 1, 2, 2]
    m.close()


def test_plan_policy_keeps_hot_experts(tmp_path, blob):
    path, _ = blob
    m = make(tmp_path, gpu_bytes=0, ram_bytes=50_000, policy="plan", policy_args={"alpha": 0.2})
    k = m.add_class(BlockClass("expert", tier=T_RAM, prior=0.05))
    blocks = [m.map_file(path, i * 10_000, 10_000, f"e{i}", k) for i in range(30)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        for i in (0, 1, 2):                               # always used
            m.acquire(blocks[i])
        m.acquire(blocks[int(rng.integers(3, 30))])       # one random cold block
        m.end_step()
    assert all(blocks[i].tier == T_RAM for i in (0, 1, 2))
    hits = k.stats["hit_ram"] / (k.stats["hit_ram"] + k.stats["fault"])
    assert hits > 0.74                                    # 3 of 4 accesses are the hot blocks
    m.close()


def test_writable_state_survives_swap(tmp_path, blob):
    path, _ = blob
    m = make(tmp_path, gpu_bytes=0, ram_bytes=30_000, policy="lru")
    a = m.map_anon(10_000, "state/a")
    b = m.map_anon(10_000, "state/b")
    assert m.read(a.vaddr, 16) == bytes(16)               # zero-filled without touching the SSD
    m.write(a.vaddr + 9_990, bytes(range(10)))
    m.write(b.vaddr, b"hello")
    for i in range(10):                                   # push both out of RAM
        m.acquire(m.map_file(path, i * 10_000, 10_000, f"f{i}"))
    assert a.tier == T_SSD and b.tier == T_SSD
    assert m.read(a.vaddr + 9_990, 10) == bytes(range(10))
    assert m.read(b.vaddr, 5) == b"hello"
    assert m.swap is not None and m.swap.end >= 20_000
    with pytest.raises(MemoryError):
        m.read(1, 1)                                      # unmapped address
    m.close()
    assert not os.path.exists(str(tmp_path / "swap.bin"))


def test_flat_read_spans_blocks(tmp_path, blob):
    path, data = blob
    m = make(tmp_path, ram_bytes=1 << 20)
    b1 = m.map_file(path, 0, 4096, "x")
    m.map_file(path, 4096, 4096, "y")
    assert m.read(b1.vaddr + 4000, 200) == data[4000:4200].tobytes()
    with pytest.raises(MemoryError):
        m.write(b1.vaddr, b"no")                          # model files are read-only
    m.close()


def test_dry_mode_accounts_without_data(tmp_path):
    m = make(tmp_path, gpu_bytes=100, ram_bytes=100, dry=True)
    k = m.add_class(BlockClass("dense", tier=T_GPU, planned=True))
    blocks = [m.map_file("/nonexistent/huge.bin", i << 40, 60, f"t{i}", k) for i in range(3)]
    for b in blocks:
        assert m.acquire(b, T_GPU) is None
    assert m.used == [60, 60] and k.stats["fault_bytes"] == 180
    m.close()


def test_prefetch_admits_in_background(tmp_path, blob):
    path, data = blob
    m = make(tmp_path, ram_bytes=1 << 20)
    b = m.map_file(path, 0, 50_000, "p")
    m.prefetch([b])
    for fut in list(m.inflight.values()):
        fut.result()
    import time
    for _ in range(100):
        if b.tier == T_RAM:
            break
        time.sleep(0.01)
    assert b.tier == T_RAM
    assert np.array_equal(m.acquire(b), data[:50_000])
    assert m.default_class.stats["hit_ram"] == 1
    m.close()
