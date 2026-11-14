"""Default retry and circuit-breaker handling of socket resets.

ConnectionResetError is an OSError, like FileNotFoundError, but it is also a
ConnectionError. The default policy retries and counts the reset, and it does
not retry or open the breaker for a missing local file.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest


def _load_resilience():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_resilience_connreset_1114", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resilience = _load_resilience()
retry = resilience.retry
async_retry = resilience.async_retry
CircuitBreaker = resilience.CircuitBreaker


def test_default_retry_retries_connection_reset_not_missing_file():
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
            raise ConnectionResetError("connection reset by peer")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 2
    assert seen == [(ConnectionResetError, "connection reset by peer", 1, 0.0)]

    missing_calls = {"n": 0}

    @retry(max_attempts=3, base_delay=0, max_delay=0, jitter=False)
    def missing():
        missing_calls["n"] += 1
        raise FileNotFoundError("manual.pdf")

    with pytest.raises(FileNotFoundError, match="manual.pdf"):
        missing()
    assert missing_calls["n"] == 1


def test_default_async_retry_retries_connection_reset():
    async def scenario():
        calls = {"n": 0}
        seen = []

        async def on_retry(exc, attempt, delay):
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
                raise ConnectionResetError("connection reset by peer")
            return "recovered"

        result = await flaky()
        return result, calls["n"], seen

    result, calls, seen = asyncio.run(scenario())
    assert result == "recovered"
    assert calls == 2
    assert seen == [(ConnectionResetError, 1, 0.0)]


def test_breaker_opens_on_connection_reset_not_missing_file():
    network = CircuitBreaker(failure_threshold=1, reset_timeout=60, name="socket")
    calls = {"n": 0}

    @network
    def reset():
        calls["n"] += 1
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(ConnectionResetError, match="connection reset by peer"):
        reset()
    assert network.state == "open"
    assert calls["n"] == 1

    with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
        reset()
    assert calls["n"] == 1

    disk = CircuitBreaker(failure_threshold=1, reset_timeout=60, name="disk")

    @disk
    def missing():
        raise FileNotFoundError("manual.pdf")

    with pytest.raises(FileNotFoundError, match="manual.pdf"):
        missing()
    assert disk.state == "closed"
    assert disk._failure_count == 0


def test_async_breaker_opens_on_connection_reset():
    async def scenario():
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=60, name="async")
        calls = {"n": 0}

        @breaker.async_call
        async def reset():
            calls["n"] += 1
            raise ConnectionResetError("connection reset by peer")

        with pytest.raises(ConnectionResetError, match="connection reset by peer"):
            await reset()
        opened = breaker.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            await reset()
        return opened, calls["n"]

    opened, calls = asyncio.run(scenario())
    assert opened is True
    assert calls == 1
