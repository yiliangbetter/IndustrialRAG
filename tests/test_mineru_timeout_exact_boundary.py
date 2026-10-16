"""MinerU's timeout uses a strict greater-than comparison.

``_run_mineru_command`` kills the child only when elapsed time is past
``timeout``. A process that is still running at exactly ``timeout`` seconds
must be left alone. The existing hung-process test jumps well past the limit,
so changing ``>`` to ``>=`` would still pass it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PARSER_PATH = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"


def _load_parser_module():
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_mineru_timeout_eq_1016",
        _PARSER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()


class _Pipe:
    def readline(self):
        return ""

    def close(self):
        return None


class _HungMineru:
    def __init__(self, clock):
        self.clock = clock
        self.stdout = _Pipe()
        self.stderr = _Pipe()
        self.killed_at_tick = None
        self.wait_calls = 0

    def poll(self):
        # Stop the loop if the timeout check disappears, instead of hanging.
        if self.clock["n"] > 6:
            return 0
        return None

    def kill(self):
        self.killed_at_tick = self.clock["n"]

    def wait(self):
        self.wait_calls += 1
        return -9


def test_exact_timeout_does_not_kill_and_the_next_tick_does(monkeypatch):
    clock = {"n": 0}

    def fake_monotonic():
        clock["n"] += 1
        if clock["n"] == 1:
            return 0.0
        if clock["n"] == 2:
            return 30.0
        return 31.0

    process = _HungMineru(clock)
    monkeypatch.setattr(parser_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(parser_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parser_module.subprocess, "Popen", lambda *_a, **_k: process)

    with pytest.raises(RuntimeError, match="did not finish within 30s") as excinfo:
        parser_module.MineruParser._run_mineru_command(
            "manual.pdf",
            "out",
            timeout=30,
        )

    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert process.killed_at_tick == 3
    assert process.wait_calls == 1
