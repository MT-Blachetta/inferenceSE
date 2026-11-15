"""Tokenizer and chat format of Kimi K3.1 Pro.

Vocabulary: the model's own ``tiktoken.model`` through the ``tiktoken``
library, with the pre-tokenisation pattern and special-token table of the
upstream ``tokenization_kimi.py``.

Chat format ("XTML"): ``<|open|>tag attrs<|sep|> ... <|close|>tag<|sep|>``,
each message terminated by ``<|end_of_msg|>``.  Plain system/user/assistant
conversations are rendered here.  Tool declarations, tool results and
structured-output requests need the full upstream renderer; when the profile
sets ``tokenizer.trust_model_code: true`` the model's own ``encoding_k3.py``
is imported from the model directory for those (that executes code shipped
with the checkpoint, exactly like ``trust_remote_code`` elsewhere).
"""
import importlib.util
import json
import os
import re

OPEN, CLOSE, SEP, END = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"

PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
NUM_RESERVED = 256
THINKING_EFFORTS = ("low", "high", "max")


def _text(content) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for p in content:
        if p.get("type") in ("image", "image_url"):
            raise NotImplementedError("image input is not supported by the myKimi3 profile (text only)")
        parts.append(p.get("text", ""))
    return "".join(parts)


def _esc(v) -> str:
    return str(v).replace("&", "&amp;").replace('"', "&quot;")


class KimiTokenizer:
    def __init__(self, model_dir: str, spec: dict):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        self.model_dir = model_dir
        self.spec = spec
        ranks = load_tiktoken_bpe(os.path.join(model_dir, "tiktoken.model"))
        names = {}
        cfg_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                names = {int(k): v["content"] for k, v in
                         json.load(f).get("added_tokens_decoder", {}).items()}
        base = len(ranks)
        self.special = {names.get(i, f"<|reserved_token_{i}|>"): i
                        for i in range(base, base + NUM_RESERVED)}
        self.enc = tiktoken.Encoding(name="kimi-k3", pat_str=PAT_STR, mergeable_ranks=ranks,
                                     special_tokens=self.special)
        self.bos_id = None                       # the chat format carries no BOS token
        self.eos_ids = {self.special[t] for t in (END, "[EOS]") if t in self.special}
        self.thinking = bool(spec.get("thinking", True))
        self.thinking_effort = spec.get("thinking_effort", "max")
        self._upstream = None

    # ---- text <-> ids
    def encode(self, text: str, special: bool = False) -> list:
        if special:
            return self.enc.encode(text, allowed_special="all")
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids) -> str:
        return self.enc.decode(list(ids))

    # ---- chat rendering: list of (text, is_control)
    def _segments(self, messages, add_generation_prompt, thinking, thinking_effort):
        seg = []
        ctl = lambda t: seg.append((t, True))
        txt = lambda t: seg.append((str(t), False)) if str(t) else None

        def open_tag(tag, attrs=()):
            ctl(OPEN)
            txt(tag)
            for k, v in attrs:
                txt(f" {k}")
                txt('="')
                txt(_esc(v))
                txt('"')
            ctl(SEP)

        def close_tag(tag):
            ctl(CLOSE)
            txt(tag)
            ctl(SEP)

        if thinking and thinking_effort in THINKING_EFFORTS:
            open_tag("message", [("role", "system"), ("type", "thinking-effort")])
            txt("`thinking_effort` guides on how much to think in your thinking channel "
                "(not including the response channel), supported values include `low`, "
                "`medium`, `high`, and `max`.\n"
                f"Now the system is invoked with `thinking_effort={thinking_effort}`.")
            close_tag("message")
            ctl(END)
        for m in messages:
            role = m.get("role")
            attrs = [("role", role)] + ([("name", m["name"])] if m.get("name") else [])
            if role in ("user", "system"):
                open_tag("message", attrs)
                txt(_text(m.get("content")))
                close_tag("message")
            elif role == "assistant":
                open_tag("message", attrs)
                if thinking:
                    open_tag("think")
                    r = m.get("reasoning_content") or m.get("reasoning")
                    if r is not None and str(r) != "":
                        txt(r)
                    close_tag("think")
                open_tag("response")
                txt(_text(m.get("content")))
                close_tag("response")
                close_tag("message")
            else:
                raise ValueError(f"role {role!r} needs the upstream renderer")
            ctl(END)
        if add_generation_prompt:
            open_tag("message", [("role", "assistant")])
            open_tag("think" if thinking else "response")
        return seg

    def _needs_upstream(self, messages, kw) -> bool:
        return bool(kw.get("tools") or kw.get("tool_choice") or kw.get("response_format")
                    or any(m.get("role") == "tool" or m.get("tool_calls") or m.get("tools")
                           for m in messages))

    def _load_upstream(self):
        if self._upstream is None:
            if not self.spec.get("trust_model_code", False):
                raise NotImplementedError(
                    "tools / structured output need the model's encoding_k3.py; set "
                    "tokenizer.trust_model_code: true in the profile to allow importing it")
            path = os.path.join(self.model_dir, "encoding_k3.py")
            spec = importlib.util.spec_from_file_location("kimi_encoding_k3", path)
            mod = importlib.util.module_from_spec(spec)
            import sys
            sys.modules["kimi_encoding_k3"] = mod
            spec.loader.exec_module(mod)
            self._upstream = mod
        return self._upstream

    def apply_chat(self, messages, add_generation_prompt=True, **kw) -> list:
        thinking = bool(kw.pop("thinking", self.thinking))
        effort = kw.pop("thinking_effort", None) or kw.pop("reasoning_effort", None) or self.thinking_effort
        if self._needs_upstream(messages, kw):
            segs = self._load_upstream().build_chat_segments(
                messages, tools=kw.pop("tools", None), add_generation_prompt=add_generation_prompt,
                thinking=thinking, thinking_effort=effort, **kw)
            segs = [(s.text, s.allow_special) for s in segs]
        else:
            segs = self._segments(messages, add_generation_prompt, thinking, effort)
        ids = []
        for text, control in segs:
            ids.extend(self.encode(text, special=control))
        return ids

    def render(self, messages, **kw) -> str:
        return self.decode(self.apply_chat(messages, **kw))

    def output_filter(self, thinking=None):
        return ChannelFilter(self.thinking if thinking is None else thinking)


_MARK = re.compile(r"<\|(open|close)\|>([^<]*?)<\|sep\|>")


class ChannelFilter:
    """Splits the generated XTML stream into ("reasoning" | "content", text)."""

    def __init__(self, thinking: bool = True):
        self.channel = "reasoning" if thinking else "content"
        self.buf = ""
        self.done = False

    def push(self, text: str) -> list:
        self.buf += text
        out = []
        while self.buf and not self.done:
            m = _MARK.search(self.buf)
            cut = m.start() if m else self.buf.find("<|")
            if cut < 0:
                cut = len(self.buf)
                tail = next((n for n in (2, 1) if self.buf.endswith("<|"[:n])), 0)
                cut -= tail if self.buf.endswith("<") else 0
            if cut > 0:
                out.append((self.channel, self.buf[:cut]))
                self.buf = self.buf[cut:]
            if not m:
                break                              # possibly an incomplete marker: wait
            kind, tag = m.group(1), m.group(2).split(" ")[0]
            self.buf = self.buf[m.end() - m.start():]
            if kind == "open" and tag == "response":
                self.channel = "content"
            elif kind == "open" and tag == "think":
                self.channel = "reasoning"
            elif kind == "open" and tag == "tools":
                self.channel = "tool_calls"
            elif kind == "close" and tag == "message":
                self.done = True
        return [(c, t) for c, t in out if t]


def load(model_dir, profile):
    spec = profile.doc.get("tokenizer") or {}
    if not model_dir or not os.path.exists(os.path.join(model_dir, "tiktoken.model")):
        # synthetic test checkpoints carry no vocabulary
        from inferencese.tokenizers import load_tokenizer
        return load_tokenizer(model_dir, {})
    return KimiTokenizer(model_dir, spec if isinstance(spec, dict) else {})
