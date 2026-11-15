"""Optimization profiles.

A profile is a folder named after the profile.  It specialises the generic
engine for one model on one machine:

    <profile>/
      profile.yaml     identity, model facts, runtime options, file index
      hardware.yaml    the host the profile was tuned for
      memory.yaml      tiers, block classes, cache policy, warm-up, swap
      *.seil           the forward pass in the VM's intermediate language
      ops.py           model-specific VM ops            (optional)
      tokenizer.py     tokenizer / chat-template plugin (optional)
      facts/           data scraped from the model's repository

Everything is declarative except ``ops.py``/``tokenizer.py``, which are plain
Python executed inside the engine process: only install profiles you trust.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from functools import partial

import yaml

from ..vm import il
from ..vm import ops as builtin_ops

PROFILE_DIRS = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiles")]


def find_profile(name_or_path: str) -> str:
    if os.path.isdir(name_or_path) and os.path.exists(os.path.join(name_or_path, "profile.yaml")):
        return os.path.abspath(name_or_path)
    extra = [p for p in os.environ.get("INFERENCESE_PROFILES", "").split(os.pathsep) if p]
    for root in extra + PROFILE_DIRS + [os.path.join(os.getcwd(), "profiles")]:
        cand = os.path.join(root, name_or_path)
        if os.path.exists(os.path.join(cand, "profile.yaml")):
            return cand
    raise FileNotFoundError(f"optimization profile {name_or_path!r} not found "
                            f"(searched {extra + PROFILE_DIRS})")


def list_profiles() -> list[str]:
    out = set()
    extra = [p for p in os.environ.get("INFERENCESE_PROFILES", "").split(os.pathsep) if p]
    for root in extra + PROFILE_DIRS:
        if os.path.isdir(root):
            out.update(d for d in os.listdir(root)
                       if os.path.exists(os.path.join(root, d, "profile.yaml")))
    return sorted(out)


def _load(path: str):
    with open(path) as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)


def deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class Profile:
    def __init__(self, path: str, overrides: dict | None = None):
        self.dir = find_profile(path)
        self.name = os.path.basename(self.dir)
        self.doc = _load(os.path.join(self.dir, "profile.yaml")) or {}
        if self.doc.get("name", self.name) != self.name:
            raise ValueError(f"profile folder {self.name!r} does not match its "
                             f"declared name {self.doc.get('name')!r}")
        for section in ("hardware", "memory"):
            ref = self.doc.get(section)
            if isinstance(ref, str):
                self.doc[section] = _load(os.path.join(self.dir, ref)) or {}
            elif ref is None and os.path.exists(os.path.join(self.dir, section + ".yaml")):
                self.doc[section] = _load(os.path.join(self.dir, section + ".yaml")) or {}
        if overrides:
            deep_merge(self.doc, copy.deepcopy(overrides))
        self.model = self.doc.setdefault("model", {})
        self.memory = self.doc.setdefault("memory", {})
        self.runtime = self.doc.setdefault("runtime", {})
        self.hardware = self.doc.setdefault("hardware", {})

    def file(self, rel: str) -> str:
        return os.path.join(self.dir, rel)

    # ---- model configuration exposed to the program as ``cfg``
    def config(self, model_dir: str | None) -> dict:
        src = self.model.get("config", "facts/config.json")
        cfg = None
        if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
            cfg = _load(os.path.join(model_dir, "config.json"))
        elif isinstance(src, dict):
            cfg = copy.deepcopy(src)
        elif src and os.path.exists(self.file(src)):
            cfg = _load(self.file(src))
        if cfg is None:
            cfg = {}
        key = self.model.get("config_key")
        if key:
            cfg = dict(cfg.get(key, {}), _root=cfg)
        deep_merge(cfg, copy.deepcopy(self.model.get("config_overrides", {})))
        return cfg

    # ---- program and ops
    def program(self) -> tuple[il.Program, str]:
        path = self.file(self.doc.get("program", "model.seil"))
        with open(path) as f:
            return il.parse(f.read(), path), path

    def ops(self) -> dict:
        registry = dict(builtin_ops.OPS)
        rel = self.doc.get("ops")
        if rel is None and os.path.exists(self.file("ops.py")):
            rel = "ops.py"
        for r in ([rel] if isinstance(rel, str) else rel or []):
            mod = self._module(r, {"op": partial(builtin_ops.op, registry=registry)})
            if hasattr(mod, "register"):
                mod.register(registry)
        return registry

    def tokenizer(self, model_dir: str | None):
        spec = self.doc.get("tokenizer") or {}
        if isinstance(spec, str):
            spec = {"plugin": spec}
        if "plugin" in spec:
            return self._module(spec["plugin"]).load(model_dir, self)
        from ..tokenizers import load_tokenizer
        return load_tokenizer(model_dir, spec, profile_dir=self.dir)

    def _module(self, rel: str, inject: dict | None = None):
        path = self.file(rel)
        name = f"inferencese_profile_{self.name}_{os.path.splitext(os.path.basename(rel))[0]}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__.update(inject or {})
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def model_path(self, override: str | None = None) -> str | None:
        return (override or os.environ.get("INFERENCESE_MODEL")
                or self.model.get("path") or None)

    def manifest(self) -> str | None:
        rel = self.model.get("manifest")
        return self.file(rel) if rel else None

    def state_dir(self) -> str:
        d = self.runtime.get("state_dir") or os.path.join(
            os.path.expanduser("~"), ".cache", "inferencese", self.name)
        os.makedirs(d, exist_ok=True)
        return d
