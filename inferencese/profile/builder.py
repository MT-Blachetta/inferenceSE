"""Create an optimization profile skeleton from a Hugging Face repository.

Everything is taken from the repository itself: the API record, the model
card, ``config.json``, the file listing with sizes, and the *headers* of the
safetensors shards (fetched with HTTP range requests, a few hundred kilobytes
per shard - the weights are not downloaded).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml

from ..formats.catalog import Catalog, add_safetensors
from .hardware import detect
from .loader import find_profile

HUB = "https://huggingface.co"
UA = {"User-Agent": "inferencese-profile-builder"}


def _get(url: str, rng: tuple | None = None, timeout: int = 120) -> bytes:
    headers = dict(UA)
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def _header(repo: str, rev: str, name: str) -> tuple[str, dict, int]:
    url = f"{HUB}/{repo}/resolve/{rev}/{name}"
    (n,) = struct.unpack("<Q", _get(url, (0, 7)))
    return name, json.loads(_get(url, (8, 8 + n - 1))), 8 + n


def scrape(repo: str, revision: str = "main") -> dict:
    api = json.loads(_get(f"{HUB}/api/models/{repo}/revision/{revision}"))
    tree = json.loads(_get(f"{HUB}/api/models/{repo}/tree/{revision}?recursive=true"))
    sizes = {e["path"]: (e.get("lfs") or {}).get("size", e.get("size", 0))
             for e in tree if e["type"] == "file"}
    out = {"api": api, "sizes": sizes, "config": {}, "card": ""}
    for key, name in (("config", "config.json"), ("card", "README.md")):
        if name in sizes:
            raw = _get(f"{HUB}/{repo}/resolve/{revision}/{name}")
            out[key] = json.loads(raw) if key == "config" else raw.decode("utf-8", "replace")
    shards = sorted(p for p in sizes if p.endswith(".safetensors"))
    cat = Catalog()
    with ThreadPoolExecutor(8) as ex:
        for name, header, start in ex.map(lambda s: _header(repo, revision, s), shards):
            add_safetensors(cat, name, header, start)
    cat.metadata["file_sizes_by_name"] = {s: sizes[s] for s in shards}
    out["catalog"] = cat
    return out


def _patterns(cat: Catalog) -> list:
    c = Counter()
    for e in cat.tensors.values():
        c[(re.sub(r"\d+", "N", e.name), e.dtype)] += 1
    return sorted(c.items())


def create_profile(repo: str, name: str, out_root: str = "profiles", revision: str = "main") -> str:
    info = scrape(repo, revision)
    cat, cfg, api = info["catalog"], info["config"], info["api"]
    dest = os.path.join(out_root, name)
    os.makedirs(os.path.join(dest, "facts"), exist_ok=True)
    with open(os.path.join(dest, "facts", "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(dest, "facts", "model_card.md"), "w") as f:
        f.write(info["card"])
    with open(os.path.join(dest, "facts", "files.json"), "w") as f:
        json.dump(info["sizes"], f, indent=1)
    with open(os.path.join(dest, "facts", "tensor_patterns.txt"), "w") as f:
        for (pat, dtype), n in _patterns(cat):
            f.write(f"{n:>8}  {dtype:<5} {pat}\n")
    if len(cat):
        import lzma
        doc_files = list(cat.files)
        rows = [[e.name, e.dtype, list(e.shape), doc_files.index(e.path), e.offset, e.nbytes]
                for e in cat.tensors.values()]
        with lzma.open(os.path.join(dest, "facts", "catalog.json.xz"), "wt", preset=9) as f:
            json.dump({"files": doc_files, "file_sizes": [info["sizes"][p] for p in doc_files],
                       "metadata": {}, "tensors": rows}, f, separators=(",", ":"))

    names = set(cat.tensors)
    llama_like = any(re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", n) for n in names)
    has_experts = any(re.search(r"\.experts\.\d+\.", n) for n in names)
    summary = cat.summary()
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    profile = {
        "name": name,
        "description": f"{repo} (generated skeleton - review before use)",
        "format_version": 1,
        "model": {
            "id": repo, "revision": api.get("sha"), "source": f"{HUB}/{repo}",
            "type": "llm", "pipeline": api.get("pipeline_tag"),
            "architecture": "mixture-of-experts" if has_experts else "dense-transformer",
            "architectures": cfg.get("architectures"),
            "file_types": sorted({os.path.splitext(p)[1].lstrip(".") for p in info["sizes"]
                                  if p.endswith((".safetensors", ".gguf", ".bin"))}),
            "tensor_types": sorted(summary["dtypes"]),
            "size_bytes": summary["bytes"], "files": summary["files"],
            "path": None, "config": "facts/config.json", "manifest": "facts/catalog.json.xz",
        },
        "program": "model.seil",
        "tokenizer": {"type": "auto"},
        "hardware": "hardware.yaml",
        "memory": "memory.yaml",
        "runtime": {"backend": "auto", "device": "auto", "prefill_chunk": 256},
        "sampling": {"temperature": 0.7, "top_p": 0.9},
    }
    if text_cfg is not None:
        profile["model"]["config_key"] = "text_config"
    if not cat.tensors.get("lm_head.weight") and "model.embed_tokens.weight" in names:
        profile["model"]["aliases"] = {"lm_head.weight": "model.embed_tokens.weight"}
    classes = {}
    if has_experts:
        k = (text_cfg or cfg).get("num_experts_per_token") or (text_cfg or cfg).get("num_experts_per_tok")
        e = (text_cfg or cfg).get("num_experts") or (text_cfg or cfg).get("num_local_experts")
        classes["expert"] = {"match": [r"\.experts\.\d+\."], "group": r"^(.*\.experts\.\d+)\.",
                             "tier": "ram", "prior": round(k / e, 4) if k and e else 0.05}
    classes["embedding"] = {"match": [r"embed_tokens\.weight$"], "tier": "ram",
                            "chunk_bytes": "4MiB", "prior": 0.05}
    classes["dense"] = {"match": [".*"], "tier": "gpu", "planned": True}
    memory = {"tiers": {"gpu": {"capacity": "auto", "reserve": "4GiB"},
                        "ram": {"capacity": "auto", "reserve": "8GiB"},
                        "ssd": {"direct_io": True, "io_threads": 16}},
              "policy": {"name": "plan"}, "classes": classes,
              "state": {"tier": "gpu", "rows_per_block": 1024},
              "warmup": [{"class": "dense", "device": "gpu"}]}
    hw = detect(".")
    for fname, doc in (("profile.yaml", profile), ("memory.yaml", memory),
                       ("hardware.yaml", {"detected_on_this_machine": hw})):
        with open(os.path.join(dest, fname), "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, width=100)
    seil = os.path.join(dest, "model.seil")
    if llama_like:
        shutil.copy(os.path.join(find_profile("generic-llama"), "model.seil"), seil)
    else:
        with open(seil, "w") as f:
            f.write("; TODO: describe the forward pass of this architecture.\n"
                    "; The tensor name patterns of the checkpoint are listed in\n"
                    "; facts/tensor_patterns.txt; see profiles/generic-llama/model.seil and\n"
                    "; profiles/myKimi3/model.seil for complete programs.\n"
                    "program model\n\nfunc main(%tokens) -> %logits\n"
                    "  %x = embed @\"model.embed_tokens.weight\", %tokens\n"
                    "  ret %x\nend\n")
    return dest
