"""Tensor catalog: where every tensor of a model lives on the SSD.

The catalog is format independent.  Readers for safetensors, GGUF and NumPy
archives only parse *headers*; no tensor data is read when a model is opened,
whatever its size.
"""
from __future__ import annotations

import glob
import json
import lzma
import os
import struct
from dataclasses import dataclass

import numpy as np

from .dtypes import GGML_BLOCK, PLAIN, nbytes_of


@dataclass(slots=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple
    path: str
    offset: int
    nbytes: int


class Catalog:
    def __init__(self):
        self.tensors: dict[str, TensorEntry] = {}
        self.metadata: dict = {}
        self.files: list[str] = []

    def add(self, e: TensorEntry) -> None:
        self.tensors[e.name] = e

    def __contains__(self, name):
        return name in self.tensors

    def __getitem__(self, name) -> TensorEntry:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"tensor {name!r} is not in the model "
                           f"({len(self.tensors)} tensors catalogued)") from None

    def __len__(self):
        return len(self.tensors)

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.tensors.values())

    def summary(self) -> dict:
        by = {}
        for e in self.tensors.values():
            d = by.setdefault(e.dtype, [0, 0])
            d[0] += 1
            d[1] += e.nbytes
        return {"tensors": len(self), "bytes": self.nbytes, "files": len(self.files),
                "dtypes": {k: {"tensors": v[0], "bytes": v[1]} for k, v in sorted(by.items())}}

    # ---- compact manifest (lets tools plan for a model that is not on this machine)
    def save_manifest(self, path: str) -> None:
        files = {p: i for i, p in enumerate(self.files)}
        rows = [[e.name, e.dtype, list(e.shape), files[e.path], e.offset, e.nbytes]
                for e in self.tensors.values()]
        doc = {"files": [os.path.basename(p) for p in self.files],
               "file_sizes": [os.path.getsize(p) if os.path.exists(p) else None for p in self.files],
               "metadata": self.metadata, "tensors": rows}
        with lzma.open(path, "wt", preset=9) as f:
            json.dump(doc, f, separators=(",", ":"))

    @classmethod
    def load_manifest(cls, path: str, root: str = "") -> "Catalog":
        with lzma.open(path, "rt") as f:
            doc = json.load(f)
        cat = cls()
        cat.metadata = doc.get("metadata", {})
        cat.metadata["file_sizes"] = doc.get("file_sizes")
        cat.files = [os.path.join(root, p) for p in doc["files"]]
        for name, dtype, shape, fi, off, nb in doc["tensors"]:
            cat.add(TensorEntry(name, dtype, tuple(shape), cat.files[fi], off, nb))
        return cat


# ------------------------------------------------------------- safetensors
def read_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 1 << 30:
            raise ValueError(f"{path}: implausible safetensors header size {n}")
        return json.loads(f.read(n)), 8 + n


def add_safetensors(cat: Catalog, path: str, header: dict | None = None,
                    data_start: int | None = None) -> None:
    if header is None:
        header, data_start = read_safetensors_header(path)
    cat.files.append(path)
    meta = header.get("__metadata__")
    if meta:
        cat.metadata.setdefault("safetensors", {}).update(meta)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        cat.add(TensorEntry(name, info["dtype"], tuple(info["shape"]), path,
                            data_start + a, b - a))


# -------------------------------------------------------------------- GGUF
_GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
               8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
               14: "Q6_K", 15: "Q8_K", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
               28: "F64", 30: "BF16"}
_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.unpack("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t in _GGUF_SCALAR:
            return self.unpack(_GGUF_SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            it, n = self.unpack("<I"), self.unpack("<Q")
            if it in _GGUF_SCALAR:
                fmt = _GGUF_SCALAR[it]
                return np.frombuffer(self.f.read(struct.calcsize(fmt) * n), dtype=fmt).tolist()
            return [self.value(it) for _ in range(n)]
        raise ValueError(f"GGUF: unknown metadata value type {t}")


def add_gguf(cat: Catalog, path: str) -> None:
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.unpack("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} is not supported")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.unpack("<I")
            dims = [r.unpack("<Q") for _ in range(nd)]
            t, off = r.unpack("<I"), r.unpack("<Q")
            infos.append((name, tuple(reversed(dims)), t, off))
        align = int(meta.get("general.alignment", 32))
        base = (f.tell() + align - 1) // align * align
    cat.files.append(path)
    cat.metadata.setdefault("gguf", {}).update(meta)
    for name, shape, t, off in infos:
        dtype = _GGUF_TYPES.get(t)
        if dtype is None:
            raise ValueError(f"{path}: tensor {name} has unsupported GGML type {t}")
        cat.add(TensorEntry(name, dtype, shape, path, base + off, nbytes_of(dtype, shape)))


# --------------------------------------------------------------- numpy npz
def add_npy(cat: Catalog, path: str, name: str | None = None) -> None:
    """Uncompressed .npy file (C-order) mapped in place."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dt = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                              else np.lib.format.read_array_header_2_0(f))
        if fortran:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        off = f.tell()
    codes = {np.dtype(v[0]).str: k for k, v in PLAIN.items() if v[0]}
    dtype = codes.get(np.dtype(dt).str)
    if dtype is None:
        raise ValueError(f"{path}: unsupported dtype {dt}")
    cat.files.append(path)
    cat.add(TensorEntry(name or os.path.splitext(os.path.basename(path))[0], dtype,
                        tuple(shape), path, off, nbytes_of(dtype, shape)))


READERS = {".safetensors": add_safetensors, ".gguf": add_gguf, ".npy": add_npy}


def open_model(path: str, manifest: str | None = None) -> Catalog:
    """Open a model directory or file and return its catalog.

    ``manifest`` may name a saved catalog (``*.json.xz``); it is used instead of
    scanning headers, with file paths resolved against ``path``."""
    if manifest and os.path.exists(manifest):
        return Catalog.load_manifest(manifest, path)
    cat = Catalog()
    if os.path.isdir(path):
        paths = sorted(p for ext in READERS for p in glob.glob(os.path.join(path, "*" + ext)))
        if not paths:
            raise FileNotFoundError(
                f"no model files ({', '.join(READERS)}) found in {path}")
    else:
        paths = [path]
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext not in READERS:
            raise ValueError(f"unsupported model file type: {p}")
        READERS[ext](cat, p)
    return cat


__all__ = ["Catalog", "TensorEntry", "open_model", "add_safetensors", "add_gguf",
           "add_npy", "read_safetensors_header", "GGML_BLOCK"]


def write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> None:
    """Minimal safetensors writer.  ``tensors``: name -> (dtype, shape, bytes)."""
    header, off = {}, 0
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    for name, (dtype, shape, data) in tensors.items():
        n = len(data)
        if n != nbytes_of(dtype, shape):
            raise ValueError(f"{name}: {n} bytes do not match {dtype}{list(shape)}")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, (_, _, data) in tensors.items():
            f.write(data)
