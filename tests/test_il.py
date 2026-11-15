import numpy as np
import pytest

from inferencese.vm import il
from inferencese.vm.interp import VM, Context, VMError
from inferencese.vm.ops import OPS


class _WS:
    def get(self, name):
        return {"w.0": np.eye(2, dtype=np.float32) * 2, "w.1": np.eye(2, dtype=np.float32) * 3}[name]

    def is_chunked(self, name):
        return False


def run(src, *args, cfg=None):
    from inferencese.backend import NumpyBackend
    ctx = Context(NumpyBackend(), _WS(), None, cfg or {"n": 2, "nested": {"k": [1, 3]}})
    return VM(il.parse(src), dict(OPS), ctx).run("main", *args)


def test_program_with_loops_calls_and_templates():
    src = """
    program demo
    func main(%x) -> %y
      let total = 0
      for i in range(cfg.n)                 ; two layers
        %x = call layer(i, %x)
        let total = total + i
      end
      if total == 1 and 3 in cfg.nested.k
        %x = scale %x, 10.0
      else
        %x = scale %x, 0.0
      end
      ret %x
    end
    func layer(i, %x) -> %x
      %x = linear %x, @"w.{i}"              # weight name built from the loop variable
      ret %x
    end
    """
    out = run(src, np.ones((1, 2), np.float32))
    assert np.allclose(out, 60.0)


def test_multiple_results_and_fstrings():
    src = """
    func main(%x)
      let key = f"k.{cfg.n + 1}"
      %a = scale %x, len(key)
      ret %a, %x
    end
    """
    a, x = run(src, np.ones(2, np.float32))
    assert np.allclose(a, 3.0) and np.allclose(x, 1.0)


@pytest.mark.parametrize("expr", ["__import__('os')", "cfg.__class__", "(lambda: 1)()",
                                  "[i for i in range(3)]", "open('x')"])
def test_expressions_are_sandboxed(expr):
    src = f"func main(%x)\n  let a = {expr}\n  ret %x\nend\n"
    with pytest.raises((il.ILError, VMError)):
        run(src, np.ones(1, np.float32))


@pytest.mark.parametrize("src,msg", [
    ("func main(%x)\n  %y = nosuchop %x\n  ret %y\nend", "undefined ops"),
    ("func main(%x)\n  %y = call nowhere(%x)\n  ret %y\nend", "undefined function"),
    ("func main(%x)\n  for i in range(2)\n  ret %x\nend", "missing 'end'"),
    ("%x = add %x, %x", "outside of a func"),
])
def test_static_errors(src, msg):
    with pytest.raises((il.ILError, VMError), match=msg):
        run(src, np.ones(1, np.float32))


def test_runtime_error_names_the_line():
    src = "func main(%x)\n  %y = add %x, %undefined\n  ret %y\nend"
    with pytest.raises(VMError, match=r":2: add: .*%undefined"):
        run(src, np.ones(1, np.float32))
