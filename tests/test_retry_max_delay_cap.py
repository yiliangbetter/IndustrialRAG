"""Retry backoff must respect max_delay so ingest cannot stall on rate limits.

Exponential backoff without a cap waits 10s, 20s, 40s… and can appear hung
during process_document_complete. jitter=False keeps the assertion deterministic.
"""

import pytest

from raganything.resilience import CircuitBreaker, async_retry, retry


def test_retry_caps_sleep_at_max_delay(monkeypatch):
    sleeps = []
    monkeypatch.setattr("raganything.resilience.time.sleep", sleeps.append)

    call_count = 0

    @retry(
        max_attempts=4,
        base_delay=10.0,
        max_delay=12.0,
        exponential_base=2.0,
        jitter=False,
        retryable_exceptions=[ConnectionError],
    )
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("upstream")

    with pytest.raises(ConnectionError, match="upstream"):
        always_fail()

    assert call_count == 4
    # min(10, 12), min(20, 12), min(40, 12)
    assert sleeps == [10.0, 12.0, 12.0]


@pytest.mark.asyncio
async def test_async_retry_caps_sleep_at_max_delay(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("raganything.resilience.asyncio.sleep", fake_sleep)

    call_count = 0

    @async_retry(
        max_attempts=4,
        base_delay=10.0,
        max_delay=12.0,
        exponential_base=2.0,
        jitter=False,
        retryable_exceptions=[TimeoutError],
    )
    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise TimeoutError("upstream")

    with pytest.raises(TimeoutError, match="upstream"):
        await always_fail()

    assert call_count == 4
    assert sleeps == [10.0, 12.0, 12.0]


def test_circuit_breaker_open_message_includes_name():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0, name="embed-calls")

    @cb
    def fail_upstream():
        raise ConnectionError("upstream")

    with pytest.raises(ConnectionError):
        fail_upstream()

    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="embed-calls"):
        fail_upstream()
