"""Tokenizers.

The engine does not reimplement tokenisation; it wraps the reference
libraries (`tokenizers`, `tiktoken`) behind one small interface.  A profile
can replace this entirely with a ``tokenizer.py`` plugin.
"""
from __future__ import annotations

import json
import os


class Tokenizer:
    eos_ids: set = set()
    bos_id: int | None = None

    def encode(self, text: str, special: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids) -> str:
        raise NotImplementedError

    def apply_chat(self, messages: list[dict], add_generation_prompt: bool = True, **kw) -> list[int]:
        raise NotImplementedError


class StreamDecoder:
    """Incremental detokeniser that never emits half a UTF-8 character."""

    def __init__(self, tok: Tokenizer):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        out, self.sent = text[self.sent:], len(text)
        return out


class ByteTokenizer(Tokenizer):
    """Fallback for models without a tokenizer file: one token per byte."""

    def __init__(self, vocab_size: int = 256, eos: int | None = None):
        self.eos_ids = {eos} if eos is not None else set()

    def encode(self, text, special=False):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", "replace")

    def apply_chat(self, messages, add_generation_prompt=True, **kw):
        text = "".join(f"<{m['role']}>{_text(m.get('content'))}\n" for m in messages)
        return self.encode(text + ("<assistant>" if add_generation_prompt else ""))


def _text(content) -> str:
    if isinstance(content, str) or content is None:
        return content or ""
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


class HFTokenizer(Tokenizer):
    """Hugging Face ``tokenizer.json`` (+ ``tokenizer_config.json`` chat template)."""

    def __init__(self, model_dir: str, spec: dict | None = None):
        from tokenizers import Tokenizer as _Tok
        spec = spec or {}
        self.tk = _Tok.from_file(os.path.join(model_dir, spec.get("file", "tokenizer.json")))
        cfg = {}
        p = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(p):
            with open(p) as f:
                cfg = json.load(f)
        self.cfg = cfg
        self.template = spec.get("chat_template") or cfg.get("chat_template")
        tpl = os.path.join(model_dir, "chat_template.jinja")
        if not self.template and os.path.exists(tpl):
            with open(tpl) as f:
                self.template = f.read()

        def tok_id(v):
            if isinstance(v, dict):
                v = v.get("content")
            return self.tk.token_to_id(v) if isinstance(v, str) else None

        self.special = {k: (v.get("content") if isinstance(v, dict) else v)
                        for k, v in cfg.items() if k.endswith("_token") and v}
        self.bos_id = tok_id(cfg.get("bos_token"))
        self.eos_ids = {i for i in [tok_id(cfg.get("eos_token"))] if i is not None}
        for extra in spec.get("stop_tokens", []):
            i = self.tk.token_to_id(extra)
            if i is not None:
                self.eos_ids.add(i)
        gen = os.path.join(model_dir, "generation_config.json")
        if os.path.exists(gen):
            with open(gen) as f:
                e = json.load(f).get("eos_token_id")
            self.eos_ids.update([e] if isinstance(e, int) else e or [])

    def encode(self, text, special=False):
        return self.tk.encode(text, add_special_tokens=special).ids

    def decode(self, ids):
        return self.tk.decode(list(ids), skip_special_tokens=True)

    def apply_chat(self, messages, add_generation_prompt=True, **kw):
        if self.template:

            from jinja2.sandbox import ImmutableSandboxedEnvironment
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                     extensions=["jinja2.ext.loopcontrols"])
            env.globals["raise_exception"] = _raise
            text = env.from_string(self.template).render(
                messages=messages, add_generation_prompt=add_generation_prompt,
                **self.special, **kw)
        else:                                   # ChatML is the most common convention
            text = "".join(f"<|im_start|>{m['role']}\n{_text(m.get('content'))}<|im_end|>\n"
                           for m in messages)
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
        return self.tk.encode(text, add_special_tokens=False).ids


def _raise(msg):
    raise ValueError(msg)


class GGUFTokenizer(HFTokenizer):
    """Tokenizer rebuilt from the vocabulary embedded in a GGUF file
    (byte-level BPE, ``tokenizer.ggml.model == "gpt2"``)."""

    def __init__(self, meta: dict, spec: dict | None = None):
        from tokenizers import AddedToken, Tokenizer as _Tok, decoders, models, pre_tokenizers
        kind = meta.get("tokenizer.ggml.model")
        if kind != "gpt2":
            raise NotImplementedError(
                f"GGUF tokenizer model {kind!r} is not supported; place the model's "
                f"tokenizer.json next to the .gguf file")
        tokens = meta["tokenizer.ggml.tokens"]
        types = meta.get("tokenizer.ggml.token_type") or [1] * len(tokens)
        vocab = {t: i for i, t in enumerate(tokens)}
        merges = [tuple(m.split(" ", 1)) for m in meta.get("tokenizer.ggml.merges", [])]
        tk = _Tok(models.BPE(vocab=vocab, merges=merges))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.add_special_tokens([AddedToken(t, special=True) for t, ty in zip(tokens, types) if ty in (3, 4)])
        self.tk = tk
        self.cfg = {}
        self.template = (spec or {}).get("chat_template") or meta.get("tokenizer.chat_template")
        bos, eos = meta.get("tokenizer.ggml.bos_token_id"), meta.get("tokenizer.ggml.eos_token_id")
        self.special = {"bos_token": tokens[bos] if bos is not None else "",
                        "eos_token": tokens[eos] if eos is not None else ""}
        self.bos_id = bos if meta.get("tokenizer.ggml.add_bos_token", True) else None
        self.eos_ids = {eos} if eos is not None else set()
        for name in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
            if name in vocab:
                self.eos_ids.add(vocab[name])


def load_tokenizer(model_dir: str | None, spec: dict | None = None, profile_dir: str | None = None,
                   gguf_meta: dict | None = None):
    spec = spec or {}
    kind = spec.get("type", "auto")
    if kind in ("auto", "hf") and model_dir and os.path.exists(
            os.path.join(model_dir, spec.get("file", "tokenizer.json"))):
        return HFTokenizer(model_dir, spec)
    if kind == "hf":
        raise FileNotFoundError(f"no tokenizer.json in {model_dir}")
    if kind in ("auto", "gguf") and gguf_meta and "tokenizer.ggml.tokens" in gguf_meta:
        return GGUFTokenizer(gguf_meta, spec)
    return ByteTokenizer(eos=spec.get("eos"))
