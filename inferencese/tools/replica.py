"""Sparse replica of a model's on-disk layout.

Given a profile's tensor manifest, recreate every model file with its true
size, a valid header and identical tensor offsets - but as a *sparse* file
whose data regions are holes.  A 1.5 TB checkpoint becomes a few megabytes on
disk, yet the engine sees the real file layout: same shards, same offsets,
same shapes.  That allows end-to-end runs of a profile (catalog, virtual
memory, paging, the SEIL program on full-size tensors) on a machine that does
not hold the weights.  Outputs are meaningless (weights read as zeros, apart
from small tensors that are filled with random values so routing varies).
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from collections import defaultdict

import numpy as np

from ..formats.catalog import Catalog
from ..profile.loader import Profile
from ..runtime.store import parse_size


def make_replica(profile: str, out_dir: str, fill_below="64KiB", seed: int = 0) -> dict:
    prof = Profile(profile)
    manifest = prof.manifest()
    if not manifest or not os.path.exists(manifest):
        raise FileNotFoundError(f"profile {prof.name} has no tensor manifest (model.manifest)")
    cat = Catalog.load_manifest(manifest, out_dir)
    sizes = cat.metadata.get("file_sizes") or [None] * len(cat.files)
    os.makedirs(out_dir, exist_ok=True)
    limit = parse_size(fill_below)
    rng = np.random.default_rng(seed)
    by_file = defaultdict(list)
    for e in cat.tensors.values():
        by_file[e.path].append(e)
    apparent = written = 0
    for path, size in zip(cat.files, sizes):
        entries = sorted(by_file[path], key=lambda e: e.offset)
        data_start = entries[0].offset
        header = {e.name: {"dtype": e.dtype, "shape": list(e.shape),
                           "data_offsets": [e.offset - data_start, e.offset - data_start + e.nbytes]}
                  for e in entries}
        blob = json.dumps(header, separators=(",", ":")).encode()
        room = data_start - 8
        if len(blob) > room:
            raise ValueError(f"{os.path.basename(path)}: rebuilt header ({len(blob)} B) does not "
                             f"fit the original header space ({room} B)")
        blob += b" " * (room - len(blob))
        end = max(e.offset + e.nbytes for e in entries)
        size = size or end
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", room))
            f.write(blob)
            written += 8 + room
            for e in entries:
                if e.nbytes <= limit:
                    f.seek(e.offset)
                    f.write(_random_values(rng, e))
                    written += e.nbytes
            f.truncate(size)                     # the rest stays a hole
        apparent += size
    for name in ("config.json",):
        src = prof.file(os.path.join("facts", name))
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    return {"files": len(cat.files), "apparent": apparent, "written": written}


def _random_values(rng, e) -> bytes:
    n = int(np.prod(e.shape)) if e.shape else 1
    if e.dtype == "F32":
        return rng.normal(0, 0.3, n).astype("<f4").tobytes()
    if e.dtype == "BF16":
        from ..formats.dtypes import f32_to_bf16
        return f32_to_bf16((1.0 + rng.normal(0, 0.1, n)).astype(np.float32)).tobytes()
    if e.dtype == "U8":
        return rng.integers(0, 256, e.nbytes, dtype=np.uint8).tobytes()
    return bytes(e.nbytes)
