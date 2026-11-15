"""Cache replacement policies for the unified virtual memory.

A policy answers one question: *how soon will this block be needed again?*
``priority(block)`` returns the (expected) distance to the block's next use,
measured in block accesses.  The pager evicts the resident block with the
largest distance, and refuses to admit a candidate whose distance is larger
than that of every block it would have to evict (admission control).

``plan`` is the default and implements "evict what will not be needed for the
longest time" (Belady's rule) as far as that is knowable:

* Blocks of *planned* classes (dense weights) are touched in the same order on
  every decoding step, because the layer schedule is fixed.  The policy records
  the access sequence of one step and from then on knows the exact distance to
  each block's next use.
* Blocks of *stochastic* classes (MoE experts, context pages) are used
  data-dependently.  For those the policy keeps an exponentially weighted
  estimate ``p`` of "used in a given step"; the expected distance to the next
  use is ``step_length / p``.

Both kinds live on one scale, so a dense tensor needed again in 40 accesses
outranks an expert that is used in 2 % of steps, and a hot expert outranks a
cold one.
"""
from __future__ import annotations


class Policy:
    name = "base"

    def __init__(self, **kw):
        self.step = 0

    def on_access(self, block, now: int) -> None:
        block.last_use = now
        block.uses += 1

    def priority(self, block, now: int) -> float:  # larger = evict sooner
        raise NotImplementedError

    def end_step(self) -> None:
        self.step += 1

    def export_heat(self, blocks) -> dict:
        return {}

    def import_heat(self, blocks_by_key: dict, heat: dict) -> None:
        pass


class LRU(Policy):
    name = "lru"

    def priority(self, block, now):
        return float(now - block.last_use)


class LFU(Policy):
    """Least-frequently-used with per-step aging."""
    name = "lfu"

    def __init__(self, decay: float = 0.98, **kw):
        super().__init__()
        self.decay = decay

    def on_access(self, block, now):
        super().on_access(block, now)
        block.p = self._p(block) + 1.0
        block.p_step = self.step

    def _p(self, block):
        return block.p * self.decay ** (self.step - block.p_step)

    def priority(self, block, now):
        return 1.0 / (self._p(block) + 1e-9)


class Plan(Policy):
    """Belady-style policy: exact next-use distance for planned blocks,
    expected next-use distance for stochastic ones (see module docstring)."""
    name = "plan"

    def __init__(self, alpha: float = 0.05, min_p: float = 1e-4, **kw):
        super().__init__()
        self.alpha = alpha
        self.min_p = min_p
        self.pos = 0            # position inside the current step
        self.length = 0         # accesses per step, learnt from the last step
        self._touched: list = []

    def on_access(self, block, now):
        super().on_access(block, now)
        self.pos += 1
        if block.planned:
            block.plan_pos = self.pos
        elif block.used_step != self.step:
            block.used_step = self.step
            self._touched.append(block)

    def _p(self, block) -> float:
        age = self.step - block.p_step
        return block.p * (1.0 - self.alpha) ** age if age > 0 else block.p

    def priority(self, block, now):
        length = max(self.length, self.pos, 1)
        if block.planned:
            if block.plan_pos < 0:
                return float(length)            # not seen yet: assume a full cycle away
            d = block.plan_pos - self.pos
            return float(d if d > 0 else d + length)
        p = self._p(block)
        if block.used_step == self.step:        # in use right now: count this step
            p = p * (1.0 - self.alpha) + self.alpha
        return length / max(p, self.min_p)

    def end_step(self):
        a = self.alpha
        for b in self._touched:
            b.p = self._p(b) * (1.0 - a) + a if b.p_step < self.step else b.p + a * (1.0 - b.p)
            b.p_step = self.step + 1
        self._touched = []
        if self.pos:
            self.length = self.pos
        self.pos = 0
        self.step += 1

    def export_heat(self, blocks):
        return {b.key: round(self._p(b), 6) for b in blocks
                if not b.planned and self._p(b) > 2 * self.min_p}

    def import_heat(self, blocks_by_key, heat):
        for key, p in heat.items():
            b = blocks_by_key.get(key)
            if b is not None and not b.planned:
                b.p, b.p_step = float(p), self.step


POLICIES = {c.name: c for c in (LRU, LFU, Plan)}


def make_policy(name: str = "plan", **kw) -> Policy:
    try:
        return POLICIES[name](**kw)
    except KeyError:
        raise ValueError(f"unknown cache policy {name!r}; choose from {sorted(POLICIES)}")
