"""Host hardware detection (used for ``capacity: auto`` and sanity warnings)."""
from __future__ import annotations

import os
import shutil
import subprocess


def detect(path: str = ".") -> dict:
    hw: dict = {"cpu": {"logical_cores": os.cpu_count()}, "ram": {}, "gpus": [], "disk": {}}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    hw["cpu"]["model"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in f if ":" in l}
        hw["ram"] = {"total": mi.get("MemTotal"), "available": mi.get("MemAvailable")}
    except OSError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                hw["gpus"].append({"index": i, "name": p.name, "total": total, "free": free})
    except Exception:
        pass
    if not hw["gpus"] and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            for line in out.strip().splitlines():
                i, name, total, free = [x.strip() for x in line.split(",")]
                hw["gpus"].append({"index": int(i), "name": name,
                                   "total": int(total) << 20, "free": int(free) << 20})
        except Exception:
            pass
    try:
        du = shutil.disk_usage(path if os.path.exists(path) else ".")
        hw["disk"] = {"path": os.path.abspath(path), "total": du.total, "free": du.free}
    except OSError:
        pass
    return hw
