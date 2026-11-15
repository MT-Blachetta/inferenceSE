"""Unified virtual memory (UVM).

One software-defined address space spans GPU VRAM, CPU RAM and the SSD:

    virtual address  -->  block  -->  { T0: VRAM | T1: RAM | T2: SSD extent }

* The **virtual memory index** (``AddressSpace``) maps every virtual address to
  a *block* and every block to the SSD extent that permanently holds its
  bytes: a ``(file, offset)`` inside the original model files for read-only
  data, or a slot in the swap file for writable data.  The SSD therefore
  always holds the whole address space; nothing is copied to build it.
* The **cache index** is the per-block residency record (``buf0``/``buf1``):
  a block is either *resident* in a fast tier or only a *reference* to its SSD
  extent.  ``acquire`` looks in T0, then T1, then faults the block in from the
  SSD, evicting whatever the policy says will be needed last.

Blocks are variable sized (a tensor, a slice of a huge tensor, or a group of
tensors that are always used together, e.g. one MoE expert), because that is
the granularity at which an inference program touches memory.
"""
from __future__ import annotations

import bisect
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .io import ALIGN, FileStore, SwapFile, aligned_buffer
from .policies import make_policy

T_GPU, T_RAM, T_SSD = 0, 1, 2
TIER_NAMES = {"gpu": T_GPU, "vram": T_GPU, "ram": T_RAM, "cpu": T_RAM,
              "ssd": T_SSD, "none": T_SSD, "disk": T_SSD}


class Block:
    __slots__ = ("id", "vaddr", "nbytes", "key", "klass", "file_id", "file_off",
                 "writable", "dirty", "zero", "pinned", "planned", "home",
                 "buf0", "buf1", "last_use", "uses", "p", "p_step", "used_step",
                 "plan_pos", "busy")

    def __init__(self, id, vaddr, nbytes, key, klass, file_id=-1, file_off=-1,
                 writable=False):
        self.id, self.vaddr, self.nbytes = id, vaddr, nbytes
        self.key, self.klass = key, klass
        self.file_id, self.file_off = file_id, file_off   # SSD extent (-1: swap, lazily allocated)
        self.writable = writable
        self.dirty = False
        self.zero = writable          # never written: reads as zeros, no SSD access
        self.pinned = False
        self.planned = klass.planned
        self.home = klass.tier        # best tier this block may live in
        self.buf0 = None              # VRAM handle
        self.buf1 = None              # RAM buffer (numpy uint8)
        self.last_use = 0
        self.uses = 0
        self.p = klass.prior
        self.p_step = 0
        self.used_step = -1
        self.plan_pos = -1
        self.busy = 0

    @property
    def tier(self) -> int:
        return T_GPU if self.buf0 is not None else T_RAM if self.buf1 is not None else T_SSD

    def __repr__(self):
        return f"<Block {self.id} {self.key} {self.nbytes}B {self.klass.name} T{self.tier}>"


@dataclass
class BlockClass:
    """Placement rule shared by a family of blocks (set by the profile)."""
    name: str
    tier: int = T_RAM          # highest tier the class may occupy
    planned: bool = False      # touched in a fixed order every step
    prior: float = 1.0         # prior probability of use per step (stochastic classes)
    pin: bool = False          # never evict once resident
    cache: bool = True         # False: stream through, never keep resident
    stats: dict = field(default_factory=lambda: defaultdict(int))


class AddressSpace:
    """Virtual memory index: virtual address -> block -> SSD extent."""

    def __init__(self, base: int = 1 << 32):
        self.blocks: list[Block] = []
        self.starts: list[int] = []
        self.by_key: dict[str, Block] = {}
        self.top = base

    def map(self, nbytes, key, klass, file_id=-1, file_off=-1, writable=False) -> Block:
        b = Block(len(self.blocks), self.top, nbytes, key, klass, file_id, file_off, writable)
        self.top += (nbytes + ALIGN - 1) // ALIGN * ALIGN
        self.blocks.append(b)
        self.starts.append(b.vaddr)
        self.by_key[key] = b
        return b

    def find(self, vaddr: int) -> Block:
        i = bisect.bisect_right(self.starts, vaddr) - 1
        if i < 0 or vaddr >= self.blocks[i].vaddr + self.blocks[i].nbytes:
            raise MemoryError(f"segmentation fault: unmapped virtual address {vaddr:#x}")
        return self.blocks[i]

    @property
    def size(self) -> int:
        return sum(b.nbytes for b in self.blocks)


class HostDevice:
    """Stand-in accelerator: 'VRAM' is ordinary host memory.  Used when no GPU
    backend is active so the three-tier logic stays testable everywhere."""
    name = "host"

    def upload(self, buf: np.ndarray):
        return buf.copy()

    def download(self, handle) -> np.ndarray:
        return np.asarray(handle)

    def free(self, handle) -> None:
        pass


class UnifiedMemory:
    def __init__(self, *, gpu_bytes: int = 0, ram_bytes: int = 1 << 30,
                 policy: str = "plan", policy_args: dict | None = None,
                 device=None, direct_io: bool = True, io_threads: int = 8,
                 swap_path: str | None = None, mirror: bool = False,
                 watermark: float = 0.03, dry: bool = False):
        self.space = AddressSpace()
        self.capacity = [int(gpu_bytes), int(ram_bytes)]
        self.used = [0, 0]
        self.resident: list[dict[int, Block]] = [{}, {}]
        self.policy = make_policy(policy, **(policy_args or {}))
        self.device = device or HostDevice()
        self.files = FileStore(direct=direct_io, io_threads=io_threads)
        self.swap_path = swap_path
        self.swap: SwapFile | None = None
        self.mirror = mirror
        self.watermark = watermark
        self.dry = dry                  # accounting only: no data is read or stored
        self.clock = 0
        self.lock = threading.RLock()
        self.inflight: dict[int, object] = {}
        self.classes: dict[str, BlockClass] = {}
        self.t_io = 0.0
        self.transient_gpu_bytes = 0
        self.default_class = self.add_class(BlockClass("default"))

    # ------------------------------------------------------------ mapping
    def add_class(self, klass: BlockClass) -> BlockClass:
        if klass.tier == T_GPU and self.capacity[T_GPU] <= 0:
            klass.tier = T_RAM
        self.classes[klass.name] = klass
        return klass

    def map_file(self, path: str, offset: int, nbytes: int, key: str,
                 klass: BlockClass | None = None) -> Block:
        """Map an extent of an existing file read-only into the address space.
        No bytes are read or copied."""
        return self.space.map(nbytes, key, klass or self.default_class,
                              self.files.add(path), offset)

    def map_anon(self, nbytes: int, key: str, klass: BlockClass | None = None) -> Block:
        """Map zero-filled writable memory (backed by the swap file on demand)."""
        return self.space.map(nbytes, key, klass or self.default_class, writable=True)

    # ------------------------------------------------------------- access
    def acquire(self, block: Block, device: int = T_RAM):
        """Return the block's bytes on ``device`` (T_GPU handle or T_RAM numpy
        uint8 array).  Lookup order: VRAM cache, RAM cache, SSD."""
        return self.acquire_many([block], device)[0]

    def acquire_many(self, blocks: list[Block], device: int = T_RAM) -> list:
        out: list = [None] * len(blocks)
        miss: list[tuple[int, Block, object]] = []
        with self.lock:
            for i, b in enumerate(blocks):
                self.clock += 1
                self.policy.on_access(b, self.clock)
                st = b.klass.stats
                b.busy += 1
                if b.buf0 is not None:
                    st["hit_gpu"] += 1
                elif b.buf1 is not None:
                    st["hit_ram"] += 1
                elif b.zero:
                    st["zero_fill"] += 1
                    self._install_ram(b, None if self.dry else aligned_buffer(b.nbytes)[:b.nbytes])
                else:
                    st["fault"] += 1
                    st["fault_bytes"] += b.nbytes
                    miss.append((i, b, self._start_read(b)))
        try:
            for i, b, fut in miss:                      # SSD reads run outside the lock
                t0 = time.perf_counter()
                data = fut.result() if fut is not None else None
                self.t_io += time.perf_counter() - t0
                with self.lock:
                    self.inflight.pop(b.id, None)
                    if b.buf0 is None and b.buf1 is None:
                        if not self._install_ram(b, data):
                            out[i] = data               # not admitted: hand out transiently
            with self.lock:
                for i, b in enumerate(blocks):
                    out[i] = self._view(b, device, out[i])
        finally:
            with self.lock:
                for b in blocks:
                    b.busy -= 1
        return out

    def _view(self, b: Block, device: int, transient):
        """Produce the handle for ``device``; promote into VRAM when allowed."""
        if device == T_GPU:
            if b.buf0 is not None:
                return b.buf0
            src = b.buf1 if b.buf1 is not None else transient
            if self.dry:
                if b.home == T_GPU and self._make_room(T_GPU, b):
                    self._set_gpu(b, True)
                else:
                    b.klass.stats["upload_bytes"] += b.nbytes
                return None
            if b.home == T_GPU and self._make_room(T_GPU, b):
                self._set_gpu(b, self.device.upload(src))
                return b.buf0
            b.klass.stats["upload_bytes"] += b.nbytes    # transient host->device copy
            return self.device.upload(src)
        if b.buf1 is not None:
            return b.buf1
        if b.buf0 is not None:
            return None if self.dry else self.device.download(b.buf0)
        return transient

    def _start_read(self, b: Block):
        if self.dry:
            return None
        fut = self.inflight.get(b.id)
        if fut is None:
            if b.file_id >= 0:
                fut = self.files.submit(b.file_id, b.file_off, b.nbytes)
            else:
                fut = self.files.pool.submit(self.swap.read, b.file_off, b.nbytes)
            self.inflight[b.id] = fut
        return fut

    def prefetch(self, blocks: list[Block]) -> None:
        """Start asynchronous SSD reads for blocks that are likely to be needed
        soon.  A prefetched block is only admitted if the policy rates it above
        what it would displace, so mispredictions cannot wreck the cache."""
        if self.dry:
            return
        with self.lock:
            for b in blocks:
                if b.buf0 is None and b.buf1 is None and not b.zero and b.id not in self.inflight:
                    b.klass.stats["prefetch"] += 1
                    b.klass.stats["fault_bytes"] += b.nbytes
                    fut = self._start_read(b)
                    fut.add_done_callback(lambda f, b=b: self._prefetched(b, f))

    def _prefetched(self, b: Block, fut) -> None:
        if fut.cancelled() or fut.exception() is not None:
            with self.lock:
                self.inflight.pop(b.id, None)
            return
        with self.lock:
            if self.inflight.pop(b.id, None) is not None and b.buf0 is None and b.buf1 is None:
                self._install_ram(b, fut.result())

    # ---------------------------------------------------------- residency
    def _install_ram(self, b: Block, data) -> bool:
        if not b.klass.cache and not b.writable:
            return False
        if b.home == T_SSD and not b.writable:
            return False
        if not self._make_room(T_RAM, b):
            if not b.writable:
                b.klass.stats["not_admitted"] += 1
                return False
            self._make_room(T_RAM, b, force=True)
        b.buf1 = data if not self.dry else True
        self.used[T_RAM] += b.nbytes
        self.resident[T_RAM][b.id] = b
        if b.klass.pin:
            b.pinned = True
        return True

    def _set_gpu(self, b: Block, handle) -> None:
        b.buf0 = handle
        self.used[T_GPU] += b.nbytes
        self.resident[T_GPU][b.id] = b
        b.klass.stats["promote"] += 1
        if not self.mirror and b.buf1 is not None and not b.writable:
            self._drop_ram(b)           # exclusive tiers: RAM copy is released

    def _drop_ram(self, b: Block) -> None:
        b.buf1 = None
        self.used[T_RAM] -= b.nbytes
        self.resident[T_RAM].pop(b.id, None)

    def _drop_gpu(self, b: Block) -> None:
        if not self.dry:
            self.device.free(b.buf0)
        b.buf0 = None
        self.used[T_GPU] -= b.nbytes
        self.resident[T_GPU].pop(b.id, None)

    def _make_room(self, tier: int, cand: Block, force: bool = False) -> bool:
        """Free space in ``tier`` for ``cand``.  Returns False (and evicts
        nothing) if that would displace blocks the policy values more."""
        cap, need = self.capacity[tier], cand.nbytes
        if need > cap:
            return False
        if self.used[tier] + need <= cap:
            return True
        now = self.clock
        cand_pri = -1.0 if force else self.policy.priority(cand, now)
        target = need + int(cap * self.watermark)
        scored = [(self.policy.priority(b, now), b) for b in self.resident[tier].values()
                  if not b.pinned and not b.busy and b is not cand]
        scored.sort(key=lambda t: t[0], reverse=True)
        victims, freed = [], self.used[tier] + need - cap
        gain = 0
        for pri, b in scored:
            if gain >= freed + (target - need):
                break
            if pri < cand_pri:
                break                       # everything left is more valuable than the candidate
            victims.append(b)
            gain += b.nbytes
        if gain < freed:
            return False
        for b in victims:
            self._evict(tier, b)
        return True

    def _evict(self, tier: int, b: Block) -> None:
        b.klass.stats["evict_gpu" if tier == T_GPU else "evict_ram"] += 1
        if tier == T_GPU:
            if b.buf1 is None and (b.writable or self.used[T_RAM] + b.nbytes <= self.capacity[T_RAM]):
                # demote VRAM -> RAM (free when RAM has room; state must not be lost)
                data = True if self.dry else self.device.download(b.buf0)
                self._drop_gpu(b)
                self._install_ram(b, data)
            else:
                self._drop_gpu(b)           # clean copy still on SSD (or mirrored in RAM)
            return
        if b.dirty:
            self._write_back(b)
        self._drop_ram(b)

    def _write_back(self, b: Block) -> None:
        if self.dry:
            b.dirty = False
            return
        if self.swap is None:
            path = self.swap_path or os.path.join(os.getcwd(), ".inferencese.swap")
            self.swap = SwapFile(path)
        if b.file_off < 0:
            b.file_off = self.swap.alloc(b.nbytes)
        self.swap.write(b.file_off, b.buf1)
        b.klass.stats["swap_out_bytes"] += b.nbytes
        b.dirty = False

    def mark_dirty(self, b: Block) -> None:
        """Declare that the RAM copy of a writable block was modified."""
        with self.lock:
            b.dirty = True
            b.zero = False
            if b.buf0 is not None:
                self._drop_gpu(b)           # stale device copy

    def pin(self, b: Block, pinned: bool = True) -> None:
        b.pinned = pinned

    def release(self, blocks: list[Block]) -> None:
        """Hint that blocks will not be needed again (drops clean copies)."""
        with self.lock:
            for b in blocks:
                if b.pinned or b.busy or b.dirty:
                    continue
                if b.buf0 is not None:
                    self._drop_gpu(b)
                if b.buf1 is not None:
                    self._drop_ram(b)

    # ------------------------------------------------- flat address access
    def read(self, vaddr: int, nbytes: int) -> bytes:
        """Byte-level read through the unified address space: the caller sees
        one flat RAM regardless of where the bytes physically are."""
        out = bytearray()
        while nbytes > 0:
            b = self.space.find(vaddr)
            off = vaddr - b.vaddr
            n = min(nbytes, b.nbytes - off)
            buf = self.acquire(b, T_RAM)
            out += bytes(memoryview(buf)[off:off + n])
            vaddr += n
            nbytes -= n
            if nbytes > 0:                   # skip alignment padding between blocks
                nxt = bisect.bisect_right(self.space.starts, vaddr - 1)
                if nxt < len(self.space.starts) and self.space.starts[nxt] > vaddr:
                    vaddr = self.space.starts[nxt]
        return bytes(out)

    def write(self, vaddr: int, data: bytes) -> None:
        pos = 0
        while pos < len(data):
            b = self.space.find(vaddr + pos)
            if not b.writable:
                raise MemoryError(f"write to read-only mapping {b.key}")
            off = vaddr + pos - b.vaddr
            n = min(len(data) - pos, b.nbytes - off)
            buf = self.acquire(b, T_RAM)
            buf[off:off + n] = np.frombuffer(data, dtype=np.uint8, count=n, offset=pos)
            self.mark_dirty(b)
            pos += n

    # ------------------------------------------------------------ control
    def end_step(self) -> None:
        with self.lock:
            self.policy.end_step()

    def warm(self, blocks: list[Block], device: int = T_RAM, chunk: int = 64) -> int:
        """Initial cache fill ("smart init"): load blocks in the given priority
        order until the tiers refuse further admissions."""
        loaded = 0
        for i in range(0, len(blocks), chunk):
            part = [b for b in blocks[i:i + chunk] if b.tier == T_SSD]
            if not part:
                continue
            free = sum(self.capacity) - sum(self.used)
            if free < part[0].nbytes:
                break
            self.acquire_many(part, device)
            loaded += sum(1 for b in part if b.tier != T_SSD)
        return loaded

    def save_heat(self, path: str) -> None:
        heat = self.policy.export_heat(self.space.blocks)
        if heat:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(heat, f)
            os.replace(tmp, path)

    def load_heat(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            heat = json.load(f)
        self.policy.import_heat(self.space.by_key, heat)
        return len(heat)

    def stats(self) -> dict:
        per = {}
        tot = defaultdict(int)
        for name, k in self.classes.items():
            if k.stats:
                per[name] = dict(k.stats)
                for a, v in k.stats.items():
                    tot[a] += v
        acc = tot["hit_gpu"] + tot["hit_ram"] + tot["fault"]
        return {
            "virtual_bytes": self.space.size,
            "blocks": len(self.space.blocks),
            "gpu": {"capacity": self.capacity[0], "used": self.used[0],
                    "blocks": len(self.resident[0])},
            "ram": {"capacity": self.capacity[1], "used": self.used[1],
                    "blocks": len(self.resident[1])},
            "ssd": {"bytes_read": self.files.bytes_read, "reads": self.files.read_calls,
                    "io_seconds": round(self.t_io, 3),
                    "swap_bytes": self.swap.end if self.swap else 0},
            "hit_rate": round((tot["hit_gpu"] + tot["hit_ram"]) / acc, 4) if acc else None,
            "totals": dict(tot),
            "classes": per,
            "policy": self.policy.name,
        }

    def close(self) -> None:
        self.files.close()
        if self.swap is not None:
            self.swap.close()
