"""The inference engine: profile + model files -> tokens."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from ..backend import make_backend
from ..formats import open_model
from ..profile.hardware import detect
from ..profile.loader import Profile
from ..uvm import T_GPU, T_RAM, TIER_NAMES, BlockClass, UnifiedMemory
from ..vm.interp import VM, Context
from .sampler import Sampler
from .store import SeqState, WeightStore, parse_size


def _capacity(spec, available: int | None, reserve) -> int:
    """Resolve a tier capacity: absolute size, 'auto' or a fraction such as '85%'."""
    if spec in (None, "auto"):
        return max(0, (available or 0) - parse_size(reserve or 0))
    if isinstance(spec, str) and spec.strip().endswith("%"):
        return int((available or 0) * float(spec.strip()[:-1]) / 100.0)
    return parse_size(spec)


class Engine:
    def __init__(self, profile: str | Profile, model_path: str | None = None,
                 overrides: dict | None = None, dry: bool = False, log=print):
        self.log = log or (lambda *a, **k: None)
        self.profile = profile if isinstance(profile, Profile) else Profile(profile, overrides)
        p = self.profile
        rt, memspec = p.runtime, p.memory
        self.model_dir = p.model_path(model_path)
        self.lock = threading.Lock()
        t0 = time.perf_counter()

        # ---- catalog: headers only, no tensor data is read
        have_files = self.model_dir and os.path.exists(self.model_dir)
        if dry and not have_files:
            self.catalog = open_model(self.model_dir or "", p.manifest())
        elif have_files:
            self.catalog = open_model(self.model_dir)
        else:
            raise FileNotFoundError(
                f"model path {self.model_dir!r} does not exist; pass --model or set "
                f"model.path in {p.file('profile.yaml')}")
        cfg_dir = self.model_dir if self.model_dir and os.path.isdir(self.model_dir) else (
            os.path.dirname(self.model_dir) if self.model_dir else None)
        self.cfg = p.config(cfg_dir)

        # ---- backend and tiers
        self.B = make_backend(rt.get("backend", "auto"), rt.get("device", "auto"),
                              threads=rt.get("threads"))
        hw = self.hardware = detect(self.model_dir or ".")
        tiers = memspec.get("tiers", {})
        g, r, s = tiers.get("gpu", {}), tiers.get("ram", {}), tiers.get("ssd", {})
        if self.B.is_gpu:
            free = hw["gpus"][0]["free"] if hw["gpus"] else 0
            gpu_bytes = _capacity(g.get("capacity", "auto"), free, g.get("reserve", "6GiB"))
        elif g.get("simulate"):
            gpu_bytes = parse_size(g.get("capacity", 0))      # VRAM emulated in host memory
        else:
            gpu_bytes = 0
        ram_bytes = _capacity(r.get("capacity", "auto"), hw["ram"].get("available"),
                              r.get("reserve", "8GiB"))
        pol = dict(memspec.get("policy", {}) or {})
        swap = (s.get("swap") or {}).get("path") or os.path.join(p.state_dir(), "swap.bin")
        self.mem = UnifiedMemory(
            gpu_bytes=gpu_bytes, ram_bytes=ram_bytes, policy=pol.pop("name", "plan"),
            policy_args=pol, device=self.B.device, direct_io=bool(s.get("direct_io", True)),
            io_threads=int(s.get("io_threads", 8)), swap_path=swap,
            mirror=bool(memspec.get("mirror_gpu_in_ram", False)), dry=dry)

        # ---- weights, state, program
        self.ws = WeightStore(self.catalog, self.mem, self.B, memspec.get("classes"),
                              p.model.get("aliases"))
        st = memspec.get("state", {})
        self.state_class = self.mem.add_class(BlockClass(
            "state", tier=TIER_NAMES[str(st.get("tier", "ram")).lower()], prior=1.0))
        self.state = SeqState(self.mem, self.B, self.state_class, int(st.get("rows_per_block", 1024)))
        self.ctx = Context(self.B, self.ws, self.mem, self.cfg, rt)
        program, path = p.program()
        self.vm = VM(program, p.ops(), self.ctx, path)
        self.tokenizer = None if dry else p.tokenizer(cfg_dir)
        self.heat_path = os.path.join(p.state_dir(), "heat.json")
        self.last_logits = None
        self.timing = {"load_s": 0.0, "prefill_tokens": 0, "prefill_s": 0.0,
                       "decode_tokens": 0, "decode_s": 0.0}
        self.log(f"[inferenceSE] profile={p.name} backend={self.B.name}"
                 f"{'/' + self.B.device.name if self.B.is_gpu else ''} "
                 f"tensors={len(self.catalog)} model={self.catalog.nbytes / 2**30:.2f} GiB "
                 f"vram-cache={gpu_bytes / 2**30:.2f} GiB ram-cache={ram_bytes / 2**30:.2f} GiB "
                 f"policy={self.mem.policy.name}")
        if not dry:
            self.warm()
        self.timing["load_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------- warm-up
    def warm(self) -> None:
        """Initial cache fill.  The profile lists which block classes to load
        first; stochastic classes are ordered by the heat map learnt in earlier
        runs, so a restart begins with the experts that were actually used."""
        steps = self.profile.memory.get("warmup") or []
        if not steps:
            return
        t0 = time.perf_counter()
        n_heat = self.mem.load_heat(self.heat_path)
        for step in steps:
            name = step.get("class")
            blocks = [b for b in self.mem.space.blocks if b.klass.name == name]
            if step.get("heat"):
                blocks = [b for b in blocks if b.p > b.klass.prior]
                blocks.sort(key=lambda b: -b.p)
            limit = parse_size(step["limit"]) if step.get("limit") else None
            if limit is not None:
                acc, keep = 0, []
                for b in blocks:
                    if acc + b.nbytes > limit:
                        break
                    acc += b.nbytes
                    keep.append(b)
                blocks = keep
            dev = TIER_NAMES[str(step.get("device", "ram")).lower()]
            dev = T_GPU if dev == T_GPU and self.mem.capacity[T_GPU] > 0 else T_RAM
            n = self.mem.warm(blocks, dev)
            self.log(f"[inferenceSE] warm-up: class={name} loaded {n}/{len(blocks)} blocks")
        self.mem.end_step()
        for k in self.mem.classes.values():
            k.stats.clear()
        self.log(f"[inferenceSE] warm-up done in {time.perf_counter() - t0:.1f}s "
                 f"(heat entries: {n_heat}); VRAM {self.mem.used[0] / 2**30:.2f} GiB, "
                 f"RAM {self.mem.used[1] / 2**30:.2f} GiB resident")

    # ------------------------------------------------------------- forward
    def forward(self, tokens) -> np.ndarray:
        """Run the profile's program over new tokens of the current sequence;
        returns the logits of the last position."""
        tokens = [int(t) for t in tokens]
        self.ctx.state = self.state
        out = self.vm.run("main", np.asarray(tokens, dtype=np.int64))
        self.state.tokens.extend(tokens)
        self.mem.end_step()
        logits = self.B.to_numpy(out).astype(np.float32).reshape(-1)
        self.last_logits = logits
        return logits

    def feed(self, tokens, on_progress=None) -> np.ndarray:
        """Bring the sequence state up to ``tokens``, reusing the already
        processed prefix (a follow-up chat turn only pays for its new tokens)."""
        tokens = [int(t) for t in tokens]
        done = self.state.tokens
        if not (len(done) <= len(tokens) and tokens[:len(done)] == done) or (
                len(done) == len(tokens) and self.last_logits is None):
            self.state.reset()
            self.last_logits = None
            done = []
        todo = tokens[len(done):]
        max_ctx = int(self.profile.runtime.get("max_context", 0) or 0)
        if max_ctx and len(tokens) > max_ctx:
            raise ValueError(f"prompt has {len(tokens)} tokens; profile limit is {max_ctx}")
        chunk = max(1, int(self.profile.runtime.get("prefill_chunk", 256)))
        t0 = time.perf_counter()
        for i in range(0, len(todo), chunk):
            self.forward(todo[i:i + chunk])
            if on_progress:
                on_progress(min(i + chunk, len(todo)), len(todo))
        self.timing["prefill_tokens"] += len(todo)
        self.timing["prefill_s"] += time.perf_counter() - t0
        return self.last_logits

    def generate(self, prompt_tokens, max_new_tokens: int = 128, sampling: dict | None = None,
                 stop_ids=None, on_progress=None):
        """Yield generated token ids one at a time."""
        opts = dict(self.profile.doc.get("sampling", {}) or {})
        opts.update({k: v for k, v in (sampling or {}).items() if v is not None})
        sampler = Sampler(**opts)
        stop = set(stop_ids if stop_ids is not None else
                   getattr(self.tokenizer, "eos_ids", None) or [])
        with self.lock:
            logits = self.feed(prompt_tokens, on_progress)
            history = list(self.state.tokens)
            for _ in range(max_new_tokens):
                tok = sampler(logits, history)
                if tok in stop:
                    break
                yield tok
                history.append(tok)
                t0 = time.perf_counter()
                logits = self.forward([tok])
                self.timing["decode_tokens"] += 1
                self.timing["decode_s"] += time.perf_counter() - t0

    # ---------------------------------------------------------- text level
    def complete(self, prompt: str, **kw):
        ids = self.tokenizer.encode(prompt)
        if getattr(self.tokenizer, "bos_id", None) is not None and self.profile.runtime.get("add_bos", True):
            ids = [self.tokenizer.bos_id] + ids
        yield from self._text(ids, **kw)

    def chat(self, messages: list[dict], template_args: dict | None = None, **kw):
        ids = self.tokenizer.apply_chat(messages, add_generation_prompt=True,
                                        **(template_args or {}))
        yield from self._text(ids, **kw)

    def _text(self, ids, stop: list[str] | None = None, **kw):
        from ..tokenizers import StreamDecoder
        dec = StreamDecoder(self.tokenizer)
        emitted = ""
        for tok in self.generate(ids, **kw):
            piece = dec.push(tok)
            if not piece:
                continue
            emitted += piece
            if stop:
                cut = min((emitted.find(s) for s in stop if s in emitted), default=-1)
                if cut >= 0:
                    keep = len(piece) - (len(emitted) - cut)
                    if keep > 0:
                        yield piece[:keep]
                    return
            yield piece

    # ---------------------------------------------------------------- misc
    def reset(self) -> None:
        with self.lock:
            self.state.reset()
            self.last_logits = None

    def stats(self) -> dict:
        t = dict(self.timing)
        t["prefill_tok_per_s"] = round(t["prefill_tokens"] / t["prefill_s"], 4) if t["prefill_s"] else None
        t["decode_tok_per_s"] = round(t["decode_tokens"] / t["decode_s"], 4) if t["decode_s"] else None
        out = {"profile": self.profile.name, "timing": t, "memory": self.mem.stats(),
               "context_tokens": self.state.pos, "decode_s_weights": round(self.ws.t_decode, 3)}
        if self.ctx.profile:
            out["ops"] = {k: {"seconds": round(v, 4), "calls": self.ctx.counts[k]}
                          for k, v in sorted(self.ctx.times.items(), key=lambda kv: -kv[1])}
        return out

    def close(self) -> None:
        try:
            self.mem.save_heat(self.heat_path)
        except OSError:
            pass
        self.mem.close()
