"""Circuit-breaker window checks use a closed lower bound.

Failures exactly ``reset_timeout`` seconds apart must not accumulate, and an
open breaker must enter half-open at that same instant. A sleep-based test
cannot tell ``>=`` from ``>``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_RESILIENCE_PATH = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"


def _load_resilience():
    spec = importlib.util.spec_from_file_location(
        "_raganything_resilience_window_73e8", _RESILIENCE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def resilience():
    return _load_resilience()


def _freeze_time(resilience, monkeypatch, now: dict):
    monkeypatch.setattr(resilience.time, "time", lambda: now["t"])


def test_failure_exactly_at_reset_timeout_does_not_accumulate(resilience, monkeypatch):
    now = {"t": 1_000.0}
    _freeze_time(resilience, monkeypatch, now)
    breaker = resilience.CircuitBreaker(
        failure_threshold=2, reset_timeout=60.0, name="llm"
    )

    @breaker
    def fail():
        raise ConnectionError("upstream")

    with pytest.raises(ConnectionError, match="upstream"):
        fail()
    assert breaker.state == "closed"
    assert breaker._failure_count == 1

    now["t"] = 1_060.0
    with pytest.raises(ConnectionError, match="upstream"):
        fail()

    assert breaker.state == "closed"
    assert breaker._failure_count == 1


def test_open_breaker_half_opens_at_exact_reset_timeout(resilience, monkeypatch):
    now = {"t": 1_000.0}
    _freeze_time(resilience, monkeypatch, now)
    breaker = resilience.CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="embed"
    )

    @breaker
    def fail():
        raise ConnectionError("upstream")

    with pytest.raises(ConnectionError, match="upstream"):
        fail()
    assert breaker.state == "open"

    now["t"] = 1_059.0
    assert breaker.state == "open"
    assert breaker._state == "open"

    now["t"] = 1_060.0
    assert breaker.state == "half-open"
    assert breaker._state == "half-open"
