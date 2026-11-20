"""Default retry of refused connections and broken pipes.

ConnectionRefusedError and BrokenPipeError are ConnectionError subclasses, so
the default retry tuple must treat them as transient. InterruptedError is an
OSError that is not a connection failure: retrying it, or letting it open the
breaker, would treat a local signal as an upstream outage.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_resilience():
    name = "_raganything_resilience_refused_1120"
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


def test_sync_retry_retries_connection_refused_not_interrupted():
    calls = {"n": 0}
    seen = []

    def on_retry(exc, attempt, delay):
        seen.append((type(exc), str(exc), attempt, delay))

    @retry(max_attempts=2, base_delay=0.0, on_retry=on_retry)
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionRefusedError("connection refused")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 2
    assert seen == [(ConnectionRefusedError, "connection refused", 1, 0.0)]

    interrupted = {"n": 0}

    @retry(max_attempts=3, base_delay=0.0)
    def local_interrupt():
        interrupted["n"] += 1
        raise InterruptedError("interrupted system call")

    with pytest.raises(InterruptedError, match="interrupted system call"):
        local_interrupt()
    assert interrupted["n"] == 1


@pytest.mark.asyncio
async def test_async_retry_retries_broken_pipe_not_interrupted():
    calls = {"n": 0}
    seen = []

    def on_retry(exc, attempt, delay):
        seen.append((type(exc), str(exc), attempt, delay))

    @async_retry(max_attempts=2, base_delay=0.0, on_retry=on_retry)
    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise BrokenPipeError("broken pipe")
        return "recovered"

    assert await flaky() == "recovered"
    assert calls["n"] == 2
    assert seen == [(BrokenPipeError, "broken pipe", 1, 0.0)]

    interrupted = {"n": 0}

    @async_retry(max_attempts=3, base_delay=0.0)
    async def local_interrupt():
        interrupted["n"] += 1
        raise InterruptedError("interrupted system call")

    with pytest.raises(InterruptedError, match="interrupted system call"):
        await local_interrupt()
    assert interrupted["n"] == 1


def test_breaker_opens_on_connection_refused_not_interrupted():
    refused = CircuitBreaker(failure_threshold=1, reset_timeout=60.0, name="refused")
    refused_runs = []

    @refused
    def refused_call():
        refused_runs.append("run")
        raise ConnectionRefusedError("connection refused")

    with pytest.raises(ConnectionRefusedError, match="connection refused"):
        refused_call()
    assert refused.state == "open"
    assert refused._failure_count == 1
    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="is open"):
        refused_call()
    assert refused_runs == ["run"]

    interrupted = CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="interrupt"
    )
    interrupt_runs = []

    @interrupted
    def interrupt_call():
        interrupt_runs.append("run")
        raise InterruptedError("interrupted system call")

    with pytest.raises(InterruptedError, match="interrupted system call"):
        interrupt_call()
    assert interrupted.state == "closed"
    assert interrupted._failure_count == 0
    assert interrupt_runs == ["run"]


@pytest.mark.asyncio
async def test_async_breaker_opens_on_broken_pipe_not_interrupted():
    pipe = CircuitBreaker(failure_threshold=1, reset_timeout=60.0, name="pipe")
    pipe_runs = []

    @pipe.async_call
    async def pipe_call():
        pipe_runs.append("run")
        raise BrokenPipeError("broken pipe")

    with pytest.raises(BrokenPipeError, match="broken pipe"):
        await pipe_call()
    assert pipe.state == "open"
    assert pipe._failure_count == 1
    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="is open"):
        await pipe_call()
    assert pipe_runs == ["run"]

    interrupted = CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="async-interrupt"
    )
    interrupt_runs = []

    @interrupted.async_call
    async def interrupt_call():
        interrupt_runs.append("run")
        raise InterruptedError("interrupted system call")

    with pytest.raises(InterruptedError, match="interrupted system call"):
        await interrupt_call()
    assert interrupted.state == "closed"
    assert interrupted._failure_count == 0
    assert interrupt_runs == ["run"]
