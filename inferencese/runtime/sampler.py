"""Token sampling."""
from __future__ import annotations

import numpy as np


class Sampler:
    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                 repetition_penalty: float = 1.0, seed: int | None = None, **_):
        self.temperature = float(temperature or 0.0)
        self.top_p = float(top_p if top_p is not None else 1.0)
        self.top_k = int(top_k or 0)
        self.repetition_penalty = float(repetition_penalty or 1.0)
        self.rng = np.random.default_rng(seed)

    def __call__(self, logits: np.ndarray, history=()) -> int:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
        if self.repetition_penalty != 1.0 and len(history):
            ids = np.unique(np.asarray(history[-256:], dtype=np.int64))
            ids = ids[ids < logits.size]
            vals = logits[ids]
            logits[ids] = np.where(vals > 0, vals / self.repetition_penalty,
                                   vals * self.repetition_penalty)
        if self.temperature <= 0.0:
            return int(np.argmax(logits))
        logits /= self.temperature
        if 0 < self.top_k < logits.size:
            kth = np.partition(logits, -self.top_k)[-self.top_k]
            logits[logits < kth] = -np.inf
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        if self.top_p < 1.0:
            order = np.argsort(-probs)
            cum = np.cumsum(probs[order])
            keep = order[: int(np.searchsorted(cum, self.top_p) + 1)]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask / mask.sum()
        return int(self.rng.choice(probs.size, p=probs))
