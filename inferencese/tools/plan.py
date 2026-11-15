"""Planning tools: what will live where, and how fast can it be.

Both work from the tensor catalog alone (``dry`` unified memory: accounting
without data), so a profile can be planned on a machine that holds neither
the weights nor the target hardware.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from ..runtime.engine import Engine
from ..runtime.store import parse_size
from ..uvm import T_GPU, T_RAM

GiB = float(1 << 30)


def plan(profile: str, model: str | None = None, overrides: dict | None = None) -> dict:
    eng = Engine(profile, model, overrides, dry=True, log=None)
    mem = eng.mem
    classes = defaultdict(lambda: {"blocks": 0, "bytes": 0})
    for b in mem.space.blocks:
        c = classes[b.klass.name]
        c["blocks"] += 1
        c["bytes"] += b.nbytes
    eng.warm()
    res = defaultdict(lambda: [0, 0])
    for t in (T_GPU, T_RAM):
        for b in mem.resident[t].values():
            res[b.klass.name][t] += b.nbytes
    rows = []
    for name, c in classes.items():
        k = mem.classes[name]
        rows.append({"class": name, "blocks": c["blocks"], "bytes": c["bytes"],
                     "home": ["vram", "ram", "ssd"][k.tier], "planned": k.planned,
                     "in_vram": res[name][0], "in_ram": res[name][1],
                     "on_ssd_only": c["bytes"] - res[name][0] - res[name][1]})
    out = {"profile": eng.profile.name, "model_bytes": eng.catalog.nbytes,
           "tensors": len(eng.catalog), "blocks": len(mem.space.blocks),
           "vram_cache": mem.capacity[0], "ram_cache": mem.capacity[1],
           "vram_used": mem.used[0], "ram_used": mem.used[1], "classes": rows}
    eng.close()
    return out


def format_plan(p: dict) -> str:
    lines = [f"profile {p['profile']}: {p['tensors']:,} tensors in {p['blocks']:,} blocks, "
             f"{p['model_bytes'] / GiB:,.1f} GiB mapped from the SSD (no copy)",
             f"tier 0 VRAM cache {p['vram_cache'] / GiB:7.1f} GiB  (filled {p['vram_used'] / GiB:.1f})",
             f"tier 1 RAM  cache {p['ram_cache'] / GiB:7.1f} GiB  (filled {p['ram_used'] / GiB:.1f})",
             "", f"{'class':<11}{'blocks':>9}{'GiB':>10}  {'home':<5}{'in VRAM':>10}{'in RAM':>10}{'SSD only':>11}"]
    for r in sorted(p["classes"], key=lambda r: -r["bytes"]):
        lines.append(f"{r['class']:<11}{r['blocks']:>9,}{r['bytes'] / GiB:>10.2f}  {r['home']:<5}"
                     f"{r['in_vram'] / GiB:>10.2f}{r['in_ram'] / GiB:>10.2f}{r['on_ssd_only'] / GiB:>11.2f}")
    return "\n".join(lines)


def simulate(profile: str, model: str | None = None, overrides: dict | None = None,
             steps: int = 300, warm_steps: int = 150, zipf: float = 1.0, repeat: float = 0.0,
             seed: int = 0) -> dict:
    """Replay a synthetic decoding workload against the profile's memory plan.

    Every step touches all planned blocks once (the fixed layer schedule) and,
    for each stochastic rule, ``picks`` blocks per group drawn from a Zipf
    popularity law with exponent ``zipf`` (0 = uniform).  ``repeat`` is the
    probability that a pick repeats one made for the previous token.
    """
    eng = Engine(profile, model, overrides, dry=True, log=None)
    mem, rng = eng.mem, np.random.default_rng(seed)
    eng.warm()
    device = T_GPU if mem.capacity[T_GPU] > 0 else T_RAM
    planned = [b for b in mem.space.blocks if b.planned and b.klass.cache]
    groups = []
    for rule in (eng.profile.memory.get("simulate", {}) or {}).get("stochastic", []):
        pat = re.compile(rule["group_by"])
        by = defaultdict(list)
        for b in mem.space.blocks:
            if b.klass.name == rule["class"]:
                m = pat.search(b.key)
                by[m.group(1) if m else ""].append(b)
        for key, blocks in by.items():
            n = len(blocks)
            w = 1.0 / np.arange(1, n + 1) ** zipf if zipf > 0 else np.ones(n)
            order = rng.permutation(n)
            groups.append({"blocks": [blocks[i] for i in order], "p": w / w.sum(),
                           "picks": int(rule["picks"]), "prev": None,
                           "pos": blocks[0].file_id * (1 << 50) + blocks[0].file_off})
    # interleave: stochastic groups are visited between the planned blocks of their layer
    schedule = sorted([(b.file_id * (1 << 50) + b.file_off, 0, b) for b in planned]
                      + [(g["pos"], 1, g) for g in groups], key=lambda t: (t[0], t[1]))

    def snapshot():
        tot = defaultdict(int)
        for k in mem.classes.values():
            for a, v in k.stats.items():
                tot[a] += v
        return tot

    base = None
    for step in range(warm_steps + steps):
        if step == warm_steps:
            base = snapshot()
        for _, kind, item in schedule:
            if kind == 0:
                mem.acquire(item, device)
                continue
            g = item
            idx = rng.choice(len(g["blocks"]), size=g["picks"], replace=False, p=g["p"])
            if repeat > 0 and g["prev"] is not None:
                keep = rng.random(g["picks"]) < repeat
                idx = np.where(keep, g["prev"], idx)
                idx = np.unique(idx)
            g["prev"] = idx if len(idx) == g["picks"] else None
            mem.acquire_many([g["blocks"][i] for i in idx], device)
        mem.end_step()
    end = snapshot()
    d = {k: (end[k] - base[k]) / steps for k in set(end) | set(base)}
    tp = (eng.profile.hardware or {}).get("throughput", {})
    n_exp = sum(g["picks"] for g in groups)
    t_ssd = d.get("fault_bytes", 0) / GiB / float(tp.get("ssd_read", 5.0))
    t_h2d = d.get("upload_bytes", 0) / GiB / float(tp.get("host_to_gpu", 10.0))
    t_gpu = float(tp.get("gpu_dense_seconds_per_token", 0.0)) + n_exp * float(tp.get("gpu_expert_seconds_each", 0.0))
    total = t_ssd + t_h2d + t_gpu
    hits = d.get("hit_gpu", 0) + d.get("hit_ram", 0)
    out = {"zipf": zipf, "repeat": repeat, "steps": steps,
           "ssd_read_gib_per_token": d.get("fault_bytes", 0) / GiB,
           "host_to_gpu_gib_per_token": d.get("upload_bytes", 0) / GiB,
           "hit_rate": hits / max(1.0, hits + d.get("fault", 0)),
           "expert_hit_rate": None,
           "seconds_per_token": {"ssd": t_ssd, "host_to_gpu": t_h2d, "gpu": t_gpu, "total": total},
           "tokens_per_second": 1.0 / total if total else None,
           "vram_used": mem.used[0], "ram_used": mem.used[1]}
    if groups:
        k = mem.classes[groups[0]["blocks"][0].klass.name]
        out["expert_hit_rate"] = 1.0 - (d.get("fault", 0) / max(1, n_exp))
    eng.close()
    return out
