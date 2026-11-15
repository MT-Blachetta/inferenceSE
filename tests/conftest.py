import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Keep heat maps and swap files of test runs out of the user's cache."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(scope="session")
def tiny_llama(tmp_path_factory):
    from inferencese.tools.fixtures import make_tiny_llama
    d = tmp_path_factory.mktemp("tiny_llama")
    make_tiny_llama(str(d))
    return str(d)


@pytest.fixture(scope="session")
def tiny_kimi(tmp_path_factory):
    from inferencese.tools.fixtures import make_tiny_kimi
    d = tmp_path_factory.mktemp("tiny_kimi")
    make_tiny_kimi(str(d))
    return str(d)


def has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
