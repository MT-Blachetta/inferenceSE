"""SEIL interpreter: executes a profile's program against the unified memory."""
from __future__ import annotations

import time
from collections import ChainMap, defaultdict

from ..runtime.store import W
from . import il


class VMError(Exception):
    pass


class _Return(Exception):
    def __init__(self, values):
        self.values = values


class Context:
    """Everything an op may touch."""

    def __init__(self, backend, weights, mem, cfg: dict, opts: dict | None = None):
        self.B = backend
        self.ws = weights
        self.mem = mem
        self.cfg = il.Cfg(cfg)
        self.opts = opts or {}
        self.state = None           # SeqState of the sequence being processed
        self.cache: dict = {}       # profile-private memo (derived constants etc.)
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.profile = bool(self.opts.get("profile_ops", False))


class VM:
    def __init__(self, program: il.Program, ops: dict, ctx: Context, filename: str = "<seil>"):
        self.program, self.ops, self.ctx, self.filename = program, ops, ctx, filename
        self.globals = {"cfg": ctx.cfg, "opts": il.Cfg(ctx.opts)}
        missing = sorted({n.name for f in program.funcs.values() for n in _walk(f.body)
                          if isinstance(n, il.Op) and n.name not in ops})
        if missing:
            raise VMError(f"{filename}: program uses undefined ops: {', '.join(missing)}")

    def run(self, func: str = "main", *args):
        f = self.program.funcs.get(func)
        if f is None:
            raise VMError(f"{self.filename}: no function {func!r}")
        return self._call(f, list(args))

    # ------------------------------------------------------------ internals
    def _call(self, f: il.Func, args: list):
        if len(args) != len(f.params):
            raise VMError(f"{self.filename}:{f.line}: {f.name} expects {len(f.params)} "
                          f"arguments, got {len(args)}")
        regs, scalars = {}, {}
        for (kind, name), val in zip(f.params, args):
            (regs if kind == "reg" else scalars)[name] = val
        env = ChainMap(scalars, self.globals)
        try:
            self._block(f.body, regs, scalars, env)
        except _Return as r:
            return r.values[0] if len(r.values) == 1 else tuple(r.values)
        return None

    def _val(self, o, regs, env):
        if isinstance(o, il.Reg):
            try:
                return regs[o.name]
            except KeyError:
                raise VMError(f"register %{o.name} used before assignment") from None
        if isinstance(o, il.WeightT):
            return W(o.render(env))
        if isinstance(o, il.ListT):
            return [self._val(i, regs, env) for i in o.items]
        return o.eval(env)

    def _block(self, body, regs, scalars, env):
        ctx = self.ctx
        for node in body:
            try:
                if isinstance(node, il.Op):
                    args = [self._val(a, regs, env) for a in node.args]
                    kwargs = {k: self._val(v, regs, env) for k, v in node.kwargs.items()}
                    if ctx.profile:
                        t0 = time.perf_counter()
                        out = self.ops[node.name](ctx, *args, **kwargs)
                        ctx.times[node.name] += time.perf_counter() - t0
                        ctx.counts[node.name] += 1
                    else:
                        out = self.ops[node.name](ctx, *args, **kwargs)
                    self._bind(node.outs, out, regs)
                elif isinstance(node, il.Call):
                    args = [self._val(a, regs, env) for a in node.args]
                    self._bind(node.outs, self._call(self.program.funcs[node.func], args), regs)
                elif isinstance(node, il.Let):
                    scalars[node.name] = node.expr.eval(env)
                elif isinstance(node, il.For):
                    for v in node.expr.eval(env):
                        scalars[node.var] = v
                        self._block(node.body, regs, scalars, env)
                elif isinstance(node, il.If):
                    for cond, arm in node.arms:
                        if cond is None or cond.eval(env):
                            self._block(arm, regs, scalars, env)
                            break
                elif isinstance(node, il.Ret):
                    raise _Return([self._val(v, regs, env) for v in node.values])
            except (_Return, VMError):
                raise
            except Exception as e:
                what = getattr(node, "name", None) or getattr(node, "func", type(node).__name__)
                raise VMError(f"{self.filename}:{node.line}: {what}: "
                              f"{type(e).__name__}: {e}") from e

    @staticmethod
    def _bind(outs, value, regs):
        if not outs:
            return
        if len(outs) == 1:
            regs[outs[0]] = value
            return
        if not isinstance(value, (tuple, list)) or len(value) != len(outs):
            raise VMError(f"op returned {type(value).__name__}, expected {len(outs)} values")
        for name, v in zip(outs, value):
            regs[name] = v


def _walk(body):
    for n in body:
        yield n
        if isinstance(n, il.For):
            yield from _walk(n.body)
        elif isinstance(n, il.If):
            for _, b in n.arms:
                yield from _walk(b)
