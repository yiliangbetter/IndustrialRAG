"""Default retry and circuit-breaker handling of read timeouts.

TimeoutError is an OSError, like PermissionError, but it is listed on its own
in the default retry tuple. ConnectionError does not cover it. The default
policy retries and counts a timeout, and it does not retry or open the breaker
for a local permission error.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest


def _load_resilience():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_resilience_timeout_1119", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resilience = _load_resilience()
retry = resilience.retry
async_retry = resilience.async_retry
CircuitBreaker = resilience.CircuitBreaker


def test_default_retry_retries_timeout_not_permission_error():
    calls = {"n": 0}
    seen = []

    def on_retry(exc, attempt, delay):
        seen.append((type(exc), str(exc), attempt, delay))

    @retry(
        max_attempts=2,
        base_delay=0,
        max_delay=0,
        jitter=False,
        on_retry=on_retry,
    )
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("read timed out")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 2
    assert seen == [(TimeoutError, "read timed out", 1, 0.0)]

    denied_calls = {"n": 0}

    @retry(max_attempts=3, base_delay=0, max_delay=0, jitter=False)
    def denied():
        denied_calls["n"] += 1
        raise PermissionError("manual.pdf")

    with pytest.raises(PermissionError, match="manual.pdf"):
        denied()
    assert denied_calls["n"] == 1


def test_default_async_retry_retries_timeout():
    async def scenario():
        calls = {"n": 0}
        seen = []

        def on_retry(exc, attempt, delay):
            seen.append((type(exc), attempt, delay))

        @async_retry(
            max_attempts=2,
            base_delay=0,
            max_delay=0,
            jitter=False,
            on_retry=on_retry,
        )
        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("read timed out")
            return "recovered"

        result = await flaky()
        return result, calls["n"], seen

    result, calls, seen = asyncio.run(scenario())
    assert result == "recovered"
    assert calls == 2
    assert seen == [(TimeoutError, 1, 0.0)]


def test_breaker_opens_on_timeout_not_permission_error():
    network = CircuitBreaker(failure_threshold=1, reset_timeout=60, name="timeout")
    calls = {"n": 0}

    @network
    def timed_out():
        calls["n"] += 1
        raise TimeoutError("read timed out")

    with pytest.raises(TimeoutError, match="read timed out"):
        timed_out()
    assert network.state == "open"
    assert calls["n"] == 1

    with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
        timed_out()
    assert calls["n"] == 1

    disk = CircuitBreaker(failure_threshold=1, reset_timeout=60, name="disk")

    @disk
    def denied():
        raise PermissionError("manual.pdf")

    with pytest.raises(PermissionError, match="manual.pdf"):
        denied()
    assert disk.state == "closed"
    assert disk._failure_count == 0


def test_async_breaker_opens_on_timeout():
    async def scenario():
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60, name="async")
        calls = {"n": 0}

        @breaker.async_call
        async def timed_out():
            calls["n"] += 1
            raise TimeoutError("read timed out")

        with pytest.raises(TimeoutError, match="read timed out"):
            await timed_out()
        opened = breaker.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            await timed_out()
        return opened, calls["n"]

    opened, calls = asyncio.run(scenario())
    assert opened is True
    assert calls == 1
