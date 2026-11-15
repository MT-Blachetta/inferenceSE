"""SEIL - the inferenceSE intermediate language.

A model's forward pass is not compiled into the engine; it is a small program
shipped inside an optimization profile and interpreted by the virtual machine.
Every weight the program names is a reference into the unified virtual memory,
never a pointer into host RAM.

Syntax (one statement per line, ``;`` or ``#`` starts a comment)::

    program llama

    func main(%tokens) -> %logits
      %x = embed @"model.embed_tokens.weight", %tokens
      for i in range(cfg.num_hidden_layers)
        %x = call block(i, %x)
      end
      %logits = linear %x, @"lm_head.weight"
      ret %logits
    end

* ``%name``         tensor register
* ``@"a.{i}.b"``    weight reference; ``{expr}`` is evaluated and substituted
* ``name=value``    keyword argument of an op
* ``f"kv.{i}"``     f-strings build state keys and other strings
* anything else     scalar expression over loop variables, ``let`` bindings and
                    ``cfg`` (the model configuration); a restricted Python
                    expression grammar, evaluated without builtins
* ``[%a, %b]``      list of values
* control flow      ``for v in expr`` / ``if expr`` / ``elif`` / ``else`` /
                    ``end``, ``let v = expr``, ``call f(...)``, ``ret ...``
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


class ILError(Exception):
    pass


# ------------------------------------------------------------ expressions
_ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Attribute,
            ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
            ast.IfExp, ast.Call, ast.List, ast.Tuple, ast.Dict, ast.keyword,
            ast.JoinedStr, ast.FormattedValue,
            ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
_FUNCS = {"range": range, "len": len, "min": min, "max": max, "int": int,
          "float": float, "abs": abs, "bool": bool, "str": str, "list": list,
          "sum": sum, "any": any, "all": all, "sorted": sorted, "round": round,
          "True": True, "False": False, "None": None}


class Expr:
    __slots__ = ("src", "code", "const")
    _cache: dict = {}

    def __init__(self, src: str):
        self.src = src = src.strip()
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as e:
            raise ILError(f"bad expression {src!r}: {e.msg}") from None
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                raise ILError(f"expression {src!r}: {type(node).__name__} is not allowed")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ILError(f"expression {src!r}: private attribute access")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ILError(f"expression {src!r}: reserved name")
        self.code = compile(tree, "<seil>", "eval")
        self.const = None
        if isinstance(tree.body, ast.Constant):
            self.const = (tree.body.value,)

    @classmethod
    def of(cls, src: str) -> "Expr":
        e = cls._cache.get(src)
        if e is None:
            e = cls._cache[src] = cls(src)
        return e

    def eval(self, env):
        if self.const is not None:
            return self.const[0]
        try:
            return eval(self.code, {"__builtins__": {}, **_FUNCS}, env)
        except ILError:
            raise
        except Exception as e:
            raise ILError(f"evaluating {self.src!r}: {type(e).__name__}: {e}") from None


class Cfg:
    """Attribute/subscript view of the (nested) model configuration."""
    __slots__ = ("_d",)

    def __init__(self, d):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, k):
        try:
            v = self._d[k]
        except KeyError:
            raise AttributeError(f"cfg has no key {k!r}") from None
        return Cfg(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        v = self._d[k]
        return Cfg(v) if isinstance(v, dict) else v

    def __contains__(self, k):
        return k in self._d

    def get(self, k, default=None):
        v = self._d.get(k, default)
        return Cfg(v) if isinstance(v, dict) else v


# ---------------------------------------------------------------- operands
@dataclass(slots=True)
class Reg:
    name: str


@dataclass(slots=True)
class WeightT:
    """Weight-name template, e.g. ``model.layers.{i}.mlp.up_proj.weight``."""
    parts: list            # alternating str / Expr

    def render(self, env) -> str:
        return "".join(p if isinstance(p, str) else str(p.eval(env)) for p in self.parts)


@dataclass(slots=True)
class ListT:
    items: list


def _split_top(s: str, sep: str = ",") -> list[str]:
    out, depth, quote, cur = [], 0, None, []
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


_TEMPLATE = re.compile(r"\{([^{}]+)\}")


def parse_operand(tok: str):
    tok = tok.strip()
    if tok.startswith("%"):
        if not re.fullmatch(r"%[A-Za-z_]\w*", tok):
            raise ILError(f"bad register name {tok!r}")
        return Reg(tok[1:])
    if tok.startswith("@"):
        m = re.fullmatch(r'@"([^"]*)"', tok)
        if not m:
            raise ILError(f"bad weight reference {tok!r} (expected @\"name\")")
        parts, pos, text = [], 0, m.group(1)
        for t in _TEMPLATE.finditer(text):
            if t.start() > pos:
                parts.append(text[pos:t.start()])
            parts.append(Expr.of(t.group(1)))
            pos = t.end()
        if pos < len(text):
            parts.append(text[pos:])
        return WeightT(parts)
    if tok.startswith("[") and tok.endswith("]") and ("%" in tok or "@" in tok):
        return ListT([parse_operand(t) for t in _split_top(tok[1:-1])])
    return Expr.of(tok)


# -------------------------------------------------------------- statements
@dataclass(slots=True)
class Op:
    outs: list
    name: str
    args: list
    kwargs: dict
    line: int


@dataclass(slots=True)
class Call:
    outs: list
    func: str
    args: list
    line: int


@dataclass(slots=True)
class Let:
    name: str
    expr: Expr
    line: int


@dataclass(slots=True)
class For:
    var: str
    expr: Expr
    body: list
    line: int


@dataclass(slots=True)
class If:
    arms: list             # [(Expr | None, body)]
    line: int


@dataclass(slots=True)
class Ret:
    values: list
    line: int


@dataclass(slots=True)
class Func:
    name: str
    params: list           # [("reg" | "scalar", name)]
    body: list = field(default_factory=list)
    line: int = 0


@dataclass(slots=True)
class Program:
    name: str
    funcs: dict
    source: str = ""


_KW = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", re.S)
_FUNC = re.compile(r"^func\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*(?:->.*)?$")
_ASSIGN = re.compile(r"^((?:%[A-Za-z_]\w*\s*,\s*)*%[A-Za-z_]\w*)\s*=(?!=)\s*(.+)$", re.S)
_CALL = re.compile(r"^call\s+([A-Za-z_]\w*)\s*\((.*)\)$", re.S)


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in ";#":
            return line[:i]
    return line


def parse(source: str, filename: str = "<seil>") -> Program:
    funcs: dict[str, Func] = {}
    name = "program"
    stack: list = []           # open blocks: (kind, node, body list)
    cur: Func | None = None

    def err(msg, ln):
        raise ILError(f"{filename}:{ln}: {msg}")

    def body():
        return stack[-1][2]

    # join continuation lines ending in a backslash or an open bracket
    lines, buf, start = [], "", 0
    for ln, raw in enumerate(source.splitlines(), 1):
        text = _strip_comment(raw).rstrip()
        if not buf:
            start = ln
        if text.endswith("\\"):
            buf += text[:-1] + " "
            continue
        buf += text
        if buf.count("(") + buf.count("[") > buf.count(")") + buf.count("]") or buf.rstrip().endswith(","):
            buf += " "
            continue
        lines.append((start, buf.strip()))
        buf = ""
    if buf.strip():
        lines.append((start, buf.strip()))

    for ln, text in lines:
        if not text:
            continue
        try:
            head = text.split(None, 1)[0]
            if head == "program":
                name = text.split(None, 1)[1].strip()
                continue
            if head == "func":
                m = _FUNC.match(text)
                if not m or stack:
                    err("malformed or nested func", ln)
                params = []
                for p in _split_top(m.group(2)):
                    params.append(("reg", p[1:]) if p.startswith("%") else ("scalar", p))
                cur = Func(m.group(1), params, [], ln)
                if cur.name in funcs:
                    err(f"function {cur.name!r} defined twice", ln)
                funcs[cur.name] = cur
                stack.append(("func", cur, cur.body))
                continue
            if cur is None:
                err("statement outside of a func", ln)
            if head == "end":
                if text != "end":
                    err("unexpected text after 'end'", ln)
                kind = stack.pop()[0]
                if kind == "func":
                    cur = None
                continue
            if head == "for":
                m = re.match(r"^for\s+([A-Za-z_]\w*)\s+in\s+(.+)$", text)
                if not m:
                    err("malformed for", ln)
                node = For(m.group(1), Expr.of(m.group(2)), [], ln)
                body().append(node)
                stack.append(("for", node, node.body))
                continue
            if head == "if":
                node = If([(Expr.of(text[2:]), [])], ln)
                body().append(node)
                stack.append(("if", node, node.arms[0][1]))
                continue
            if head in ("elif", "else"):
                if not stack or stack[-1][0] != "if":
                    err(f"{head} without if", ln)
                node = stack.pop()[1]
                cond = Expr.of(text[4:]) if head == "elif" else None
                if head == "else" and text != "else":
                    err("unexpected text after 'else'", ln)
                node.arms.append((cond, []))
                stack.append(("if", node, node.arms[-1][1]))
                continue
            if head == "let":
                m = re.match(r"^let\s+([A-Za-z_]\w*)\s*=\s*(.+)$", text)
                if not m:
                    err("malformed let", ln)
                body().append(Let(m.group(1), Expr.of(m.group(2)), ln))
                continue
            if head == "ret":
                rest = text[3:].strip()
                body().append(Ret([parse_operand(t) for t in _split_top(rest)] if rest else [], ln))
                continue
            outs, rhs = [], text
            m = _ASSIGN.match(text)
            if m:
                outs = [o.strip()[1:] for o in m.group(1).split(",")]
                rhs = m.group(2).strip()
            m = _CALL.match(rhs)
            if m:
                body().append(Call(outs, m.group(1),
                                   [parse_operand(t) for t in _split_top(m.group(2))], ln))
                continue
            parts = rhs.split(None, 1)
            opname = parts[0]
            if not re.fullmatch(r"[A-Za-z_][\w.]*", opname):
                err(f"cannot parse statement: {text}", ln)
            args, kwargs = [], {}
            for tok in _split_top(parts[1]) if len(parts) > 1 else []:
                k = _KW.match(tok)
                if k and not tok.startswith(("%", "@", "[")):
                    kwargs[k.group(1)] = parse_operand(k.group(2))
                elif kwargs:
                    err("positional argument after keyword argument", ln)
                else:
                    args.append(parse_operand(tok))
            body().append(Op(outs, opname, args, kwargs, ln))
        except ILError as e:
            if str(e).startswith(filename):
                raise
            err(str(e), ln)
    if stack:
        raise ILError(f"{filename}: missing 'end' for {stack[-1][0]} opened at line {stack[-1][1].line}")
    for f in funcs.values():
        _check_calls(f.body, funcs, filename)
    return Program(name, funcs, source)


def _check_calls(body, funcs, filename):
    for node in body:
        if isinstance(node, Call) and node.func not in funcs:
            raise ILError(f"{filename}:{node.line}: call to undefined function {node.func!r}")
        if isinstance(node, For):
            _check_calls(node.body, funcs, filename)
        if isinstance(node, If):
            for _, b in node.arms:
                _check_calls(b, funcs, filename)
