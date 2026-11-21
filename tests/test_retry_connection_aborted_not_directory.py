"""Default retry of aborted connections, not directory path errors.

ConnectionAbortedError is a ConnectionError subclass, so the default retry
tuple must treat it as transient. IsADirectoryError is an OSError that is not
a connection failure: retrying it, or letting it open the breaker, would treat
a local path mistake as an upstream outage.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_resilience():
    name = "_raganything_resilience_aborted_1121"
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resilience = _load_resilience()
retry = resilience.retry
async_retry = resilience.async_retry
CircuitBreaker = resilience.CircuitBreaker


def test_sync_retry_retries_connection_aborted_not_directory():
    calls = {"n": 0}
    seen = []

    def on_retry(exc, attempt, delay):
        seen.append((type(exc), str(exc), attempt, delay))

    @retry(max_attempts=2, base_delay=0.0, on_retry=on_retry)
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionAbortedError("connection aborted by peer")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 2
    assert seen == [(ConnectionAbortedError, "connection aborted by peer", 1, 0.0)]

    directory = {"n": 0}

    @retry(max_attempts=3, base_delay=0.0)
    def local_directory():
        directory["n"] += 1
        raise IsADirectoryError("manuals")

    with pytest.raises(IsADirectoryError, match="manuals"):
        local_directory()
    assert directory["n"] == 1


@pytest.mark.asyncio
async def test_async_retry_retries_connection_aborted_not_directory():
    calls = {"n": 0}
    seen = []

    def on_retry(exc, attempt, delay):
        seen.append((type(exc), str(exc), attempt, delay))

    @async_retry(max_attempts=2, base_delay=0.0, on_retry=on_retry)
    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionAbortedError("connection aborted by peer")
        return "recovered"

    assert await flaky() == "recovered"
    assert calls["n"] == 2
    assert seen == [(ConnectionAbortedError, "connection aborted by peer", 1, 0.0)]

    directory = {"n": 0}

    @async_retry(max_attempts=3, base_delay=0.0)
    async def local_directory():
        directory["n"] += 1
        raise IsADirectoryError("manuals")

    with pytest.raises(IsADirectoryError, match="manuals"):
        await local_directory()
    assert directory["n"] == 1


def test_breaker_opens_on_connection_aborted_not_directory():
    aborted = CircuitBreaker(failure_threshold=1, reset_timeout=60.0, name="aborted")
    aborted_runs = []

    @aborted
    def aborted_call():
        aborted_runs.append("run")
        raise ConnectionAbortedError("connection aborted by peer")

    with pytest.raises(ConnectionAbortedError, match="connection aborted by peer"):
        aborted_call()
    assert aborted.state == "open"
    assert aborted._failure_count == 1
    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="is open"):
        aborted_call()
    assert aborted_runs == ["run"]

    directory = CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="directory"
    )
    directory_runs = []

    @directory
    def directory_call():
        directory_runs.append("run")
        raise IsADirectoryError("manuals")

    with pytest.raises(IsADirectoryError, match="manuals"):
        directory_call()
    assert directory.state == "closed"
    assert directory._failure_count == 0
    assert directory_runs == ["run"]


@pytest.mark.asyncio
async def test_async_breaker_opens_on_connection_aborted_not_directory():
    aborted = CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="async-aborted"
    )
    aborted_runs = []

    @aborted.async_call
    async def aborted_call():
        aborted_runs.append("run")
        raise ConnectionAbortedError("connection aborted by peer")

    with pytest.raises(ConnectionAbortedError, match="connection aborted by peer"):
        await aborted_call()
    assert aborted.state == "open"
    assert aborted._failure_count == 1
    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="is open"):
        await aborted_call()
    assert aborted_runs == ["run"]

    directory = CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="async-directory"
    )
    directory_runs = []

    @directory.async_call
    async def directory_call():
        directory_runs.append("run")
        raise IsADirectoryError("manuals")

    with pytest.raises(IsADirectoryError, match="manuals"):
        await directory_call()
    assert directory.state == "closed"
    assert directory._failure_count == 0
    assert directory_runs == ["run"]
