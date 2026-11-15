"""Binding between a model's tensors and the unified virtual memory.

``WeightStore`` maps every catalogued tensor into the address space according
to the profile's memory rules (block classes, grouping, chunking) and hands
compute-ready weights to the VM.  ``SeqState`` keeps per-conversation state in
the same address space, so context that outgrows RAM pages out to the SSD.
"""
from __future__ import annotations

import fnmatch
import re
import time

import numpy as np

from ..formats.dtypes import PLAIN
from ..uvm import T_GPU, T_RAM, TIER_NAMES, BlockClass


class W(str):
    """A weight name (distinguishes weight operands from plain strings)."""
    __slots__ = ()


class Rule:
    def __init__(self, name: str, spec: dict):
        self.name = name
        pats = spec.get("match", [".*"])
        self.match = [re.compile(p) for p in ([pats] if isinstance(pats, str) else pats)]
        self.group = re.compile(spec["group"]) if spec.get("group") else None
        self.chunk_bytes = _size(spec.get("chunk_bytes", 0))
        self.klass = BlockClass(
            name,
            tier=TIER_NAMES[str(spec.get("tier", "ram")).lower()],
            planned=bool(spec.get("planned", False)),
            prior=float(spec.get("prior", 1.0)),
            pin=bool(spec.get("pin", False)),
            cache=bool(spec.get("cache", True)),
        )

    def matches(self, name: str) -> bool:
        return any(p.search(name) for p in self.match)


def _size(v) -> int:
    """'96GiB', '512MB', 4096 -> bytes."""
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmMgGtT]?)(i?)[bB]?\s*", str(v))
    if not m:
        raise ValueError(f"cannot parse size {v!r}")
    mult = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}[m.group(2).lower()]
    return int(float(m.group(1)) * 1024 ** mult)      # GB and GiB both mean 2**30 here


parse_size = _size


def _row_aligned(e) -> bool:
    """Block-quantised tensors can only be split between rows when a row is a
    whole number of quantisation blocks."""
    from ..formats.dtypes import GGML_BLOCK
    if e.dtype not in GGML_BLOCK:
        return True
    per_row = 1
    for d in e.shape[1:]:
        per_row *= d
    return per_row % GGML_BLOCK[e.dtype][0] == 0


class _Loc:
    __slots__ = ("entry", "block", "off", "chunks", "rows_per_chunk")

    def __init__(self, entry, block=None, off=0, chunks=None, rows_per_chunk=0):
        self.entry, self.block, self.off = entry, block, off
        self.chunks, self.rows_per_chunk = chunks, rows_per_chunk


class WeightStore:
    def __init__(self, catalog, mem, backend, rules: dict | None = None, aliases: dict | None = None):
        self.catalog, self.mem, self.B = catalog, mem, backend
        self.device = T_GPU if mem.capacity[T_GPU] > 0 else T_RAM
        self.aliases = aliases or {}
        self.rules = [Rule(n, s) for n, s in (rules or {}).items()]
        if not any(r.name == "default" for r in self.rules):
            self.rules.append(Rule("default", {"match": ".*", "tier": "gpu", "planned": True}))
        for r in self.rules:
            mem.add_class(r.klass)
        self.loc: dict[str, _Loc] = {}
        self.t_decode = 0.0
        self._map()

    # ---------------------------------------------------------------- map
    def _rule(self, name: str) -> Rule:
        for r in self.rules:
            if r.matches(name):
                return r
        return self.rules[-1]

    def _map(self) -> None:
        mem = self.mem
        cur = None               # open group: [key, rule, path, start, end, members]

        def flush():
            nonlocal cur
            if cur is None:
                return
            key, rule, path, start, end, members = cur
            blk = mem.map_file(path, start, end - start, key, rule.klass)
            for e in members:
                self.loc[e.name] = _Loc(e, blk, e.offset - start)
            cur = None

        entries = sorted(self.catalog.tensors.values(), key=lambda e: (e.path, e.offset))
        for e in entries:
            rule = self._rule(e.name)
            gkey = None
            if rule.group is not None:
                m = rule.group.search(e.name)
                if m:
                    gkey = m.group(1) if m.groups() else m.group(0)
            if gkey is not None:
                if (cur is not None and cur[0] == gkey and cur[2] == e.path
                        and 0 <= e.offset - cur[4] < 4096):
                    cur[4] = e.offset + e.nbytes
                    cur[5].append(e)
                    continue
                flush()
                cur = [gkey, rule, e.path, e.offset, e.offset + e.nbytes, [e]]
                continue
            flush()
            if (rule.chunk_bytes and e.nbytes > rule.chunk_bytes and len(e.shape) >= 2
                    and _row_aligned(e)):
                row_bytes = e.nbytes // e.shape[0]
                rpc = max(1, rule.chunk_bytes // row_bytes)
                chunks = [mem.map_file(e.path, e.offset + r * row_bytes,
                                       min(rpc, e.shape[0] - r) * row_bytes,
                                       f"{e.name}[{r}:]", rule.klass)
                          for r in range(0, e.shape[0], rpc)]
                self.loc[e.name] = _Loc(e, chunks=chunks, rows_per_chunk=rpc)
            else:
                self.loc[e.name] = _Loc(e, mem.map_file(e.path, e.offset, e.nbytes, e.name, rule.klass))
        flush()

    # ------------------------------------------------------------- lookup
    def _resolve(self, name: str) -> _Loc:
        loc = self.loc.get(name)
        if loc is None:
            alias = self.aliases.get(name)
            if alias is not None:
                loc = self.loc.get(alias)
            if loc is None:
                self.catalog[name]          # raises a helpful KeyError
        return loc

    def has(self, name: str) -> bool:
        return name in self.loc or self.aliases.get(name) in self.loc

    def shape(self, name: str) -> tuple:
        return self._resolve(name).entry.shape

    def dtype(self, name: str) -> str:
        return self._resolve(name).entry.dtype

    def blocks(self, names) -> list:
        out, seen = [], set()
        for n in names:
            loc = self._resolve(n)
            for b in (loc.chunks or [loc.block]):
                if b.id not in seen:
                    seen.add(b.id)
                    out.append(b)
        return out

    def match(self, pattern: str) -> list[str]:
        return fnmatch.filter(self.loc, pattern)

    # -------------------------------------------------------------- fetch
    def _slice(self, handle, loc: _Loc):
        e = loc.entry
        if loc.off == 0 and e.nbytes == loc.block.nbytes:
            return handle
        part = handle[loc.off:loc.off + e.nbytes]
        item = PLAIN.get(e.dtype, (None, 1))[1]
        if loc.off % item and hasattr(part, "clone"):
            part = part.clone()             # torch cannot re-type a misaligned view
        return part

    def raw_many(self, names: list[str]) -> list:
        """Storage-format bytes of several tensors (one batched acquire, so
        missing blocks are read from the SSD in parallel)."""
        locs = [self._resolve(n) for n in names]
        blocks, index = [], {}
        for loc in locs:
            if loc.chunks:
                raise ValueError(f"{loc.entry.name}: chunked tensors have no single raw view")
            if loc.block.id not in index:
                index[loc.block.id] = len(blocks)
                blocks.append(loc.block)
        handles = self.mem.acquire_many(blocks, self.device)
        return [self._slice(handles[index[loc.block.id]], loc) for loc in locs]

    def raw(self, name: str):
        return self.raw_many([name])[0]

    def get_many(self, names: list[str]) -> list:
        raws = self.raw_many(names)
        t0 = time.perf_counter()
        out = [self.B.weight(r, self._resolve(n).entry.dtype, self._resolve(n).entry.shape)
               for r, n in zip(raws, names)]
        self.t_decode += time.perf_counter() - t0
        return out

    def get(self, name: str):
        loc = self._resolve(name)
        if loc.chunks:
            e = loc.entry
            parts = self.mem.acquire_many(loc.chunks, self.device)
            rows = [b.nbytes // (e.nbytes // e.shape[0]) for b in loc.chunks]
            return self.B.cat([self.B.weight(p, e.dtype, (r,) + tuple(e.shape[1:]))
                               for p, r in zip(parts, rows)], 0)
        return self.get_many([name])[0]

    def is_chunked(self, name: str) -> bool:
        return bool(self._resolve(name).chunks)

    def iter_chunks(self, name: str):
        """Yield a chunked tensor piece by piece (rows), so a matrix larger than
        the caches can still be multiplied with bounded memory."""
        loc = self._resolve(name)
        e = loc.entry
        row_bytes = e.nbytes // e.shape[0]
        for blk in loc.chunks:
            h = self.mem.acquire(blk, self.device)
            yield self.B.weight(h, e.dtype, (blk.nbytes // row_bytes,) + tuple(e.shape[1:]))

    def rows(self, name: str, ids) -> object:
        """Gather rows of a (possibly chunked) 2-D tensor, touching only the
        chunks that contain them - an embedding lookup reads kilobytes."""
        loc = self._resolve(name)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        e = loc.entry
        if not loc.chunks:
            return self.B.take(self.B.f32(self.get(name)), ids, 0)
        rpc = loc.rows_per_chunk
        need = sorted(set((ids // rpc).tolist()))
        handles = dict(zip(need, self.mem.acquire_many([loc.chunks[c] for c in need], T_RAM)))
        row_bytes = e.nbytes // e.shape[0]
        out = np.empty((len(ids), row_bytes), dtype=np.uint8)
        for i, t in enumerate(ids.tolist()):
            r = t % rpc
            out[i] = np.asarray(handles[t // rpc])[r * row_bytes:(r + 1) * row_bytes]
        return self.B.f32(self.B.weight(out.reshape(-1), e.dtype, (len(ids),) + tuple(e.shape[1:])))


class PagedLog:
    """Append-only table of fixed-width rows stored in pageable virtual memory
    (float16 on the SSD/RAM, float32 when read)."""

    def __init__(self, mem, backend, klass, key: str, width: int, rows_per_block: int = 1024):
        self.mem, self.B, self.klass, self.key = mem, backend, klass, key
        self.width, self.rpb = width, rows_per_block
        self.blocks: list = []
        self.n = 0
        self.device = T_GPU if mem.capacity[T_GPU] > 0 and klass.tier == T_GPU else T_RAM

    def __len__(self):
        return self.n

    def reset(self):
        self.n = 0

    def append(self, rows) -> None:
        rows = np.asarray(self.B.to_numpy(rows), dtype=np.float16).reshape(-1, self.width)
        done = 0
        while done < len(rows):
            bi, r = divmod(self.n, self.rpb)
            if bi == len(self.blocks):
                self.blocks.append(self.mem.map_anon(self.rpb * self.width * 2,
                                                     f"{self.key}/{bi}", self.klass))
            take = min(self.rpb - r, len(rows) - done)
            buf = self.mem.acquire(self.blocks[bi], T_RAM)
            view = np.frombuffer(buf, dtype=np.float16).reshape(self.rpb, self.width)
            view[r:r + take] = rows[done:done + take]
            self.mem.mark_dirty(self.blocks[bi])
            done += take
            self.n += take

    def read_all(self):
        """All rows as a float32 array on the compute device."""
        if self.n == 0:
            return self.B.zeros((0, self.width))
        nb = (self.n + self.rpb - 1) // self.rpb
        handles = self.mem.acquire_many(self.blocks[:nb], self.device)
        parts = []
        for i, h in enumerate(handles):
            rows = min(self.rpb, self.n - i * self.rpb)
            parts.append(self.B.f32(self.B.weight(h, "F16", (self.rpb, self.width))[:rows]))
        return parts[0] if len(parts) == 1 else self.B.cat(parts, 0)

    def read_rows(self, idx) -> object:
        """Gather selected rows; only the blocks that hold them are touched."""
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        need = sorted(set((idx // self.rpb).tolist()))
        handles = dict(zip(need, self.mem.acquire_many([self.blocks[b] for b in need], T_RAM)))
        out = np.empty((len(idx), self.width), dtype=np.float16)
        for b in need:
            sel = np.nonzero(idx // self.rpb == b)[0]
            view = np.frombuffer(handles[b], dtype=np.float16).reshape(self.rpb, self.width)
            out[sel] = view[idx[sel] % self.rpb]
        return self.B.asarray(out.astype(np.float32))


class SeqState:
    """State of one token sequence: small tensors (recurrent states) and paged
    logs (per-token caches)."""

    def __init__(self, mem, backend, klass, rows_per_block: int = 1024):
        self.mem, self.B, self.klass, self.rpb = mem, backend, klass, rows_per_block
        self.vars: dict = {}
        self.logs: dict[str, PagedLog] = {}
        self.tokens: list[int] = []
        self.checkpoint = None
        self._uid = id(self)

    @property
    def pos(self) -> int:
        return len(self.tokens)

    def log(self, key: str, width: int) -> PagedLog:
        lg = self.logs.get(key)
        if lg is None:
            lg = self.logs[key] = PagedLog(self.mem, self.B, self.klass,
                                           f"state/{self._uid}/{key}", width, self.rpb)
        return lg

    def reset(self) -> None:
        self.vars.clear()
        for lg in self.logs.values():
            lg.reset()
        self.tokens = []
        self.checkpoint = None

    # A follow-up request usually shares a long prefix with the previous one.
    # Paged logs can simply be truncated; recurrent state cannot be unwound, so
    # one snapshot of it is kept at the end of the last prompt.
    def save_checkpoint(self, extra=None) -> None:
        self.checkpoint = (len(self.tokens), {k: self.B.copy(v) for k, v in self.vars.items()}, extra)

    def rewind(self, pos: int):
        """Truncate to the longest restorable prefix <= ``pos``.  Returns the
        ``extra`` payload stored with the checkpoint if it was hit exactly."""
        extra = None
        if self.vars or (self.checkpoint and self.checkpoint[1]):
            ck = self.checkpoint
            if ck is None or ck[0] > pos:
                self.reset()
                return None
            pos = ck[0]
            self.vars = {k: self.B.copy(v) for k, v in ck[1].items()}
            extra = ck[2]
        elif self.checkpoint and self.checkpoint[0] == pos:
            extra = self.checkpoint[2]
        for lg in self.logs.values():
            lg.n = min(lg.n, pos)
        self.tokens = self.tokens[:pos]
        return extra
