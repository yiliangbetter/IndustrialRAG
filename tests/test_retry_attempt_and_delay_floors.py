"""Retry floors: one attempt does not sleep, and a zero delay is legal.

``max_attempts`` is a total count, so 1 means "call once and raise".
``base_delay`` and ``max_delay`` may be 0 (retry immediately). Negative values
stay rejected by the existing validation tests. Sync and async copies of the
decorator both have to keep this floor.
"""

import pytest

from raganything.resilience import async_retry, retry


def test_single_attempt_does_not_sleep_or_notify(monkeypatch):
    sleeps = []
    retries = []
    monkeypatch.setattr(
        "raganything.resilience.time.sleep", lambda delay: sleeps.append(delay)
    )

    def on_retry(exc, attempt, delay):
        retries.append((type(exc), attempt, delay))

    calls = 0

    @retry(
        max_attempts=1,
        base_delay=0,
        max_delay=0,
        jitter=False,
        on_retry=on_retry,
    )
    def func():
        nonlocal calls
        calls += 1
        raise TimeoutError("down")

    with pytest.raises(TimeoutError, match="down"):
        func()

    assert calls == 1
    assert sleeps == []
    assert retries == []


def test_zero_delays_retry_immediately(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        "raganything.resilience.time.sleep", lambda delay: sleeps.append(delay)
    )
    calls = 0

    @retry(max_attempts=2, base_delay=0, max_delay=0, jitter=False)
    def func():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("down")
        return "ok"

    assert func() == "ok"
    assert calls == 2
    assert sleeps == [0.0]


@pytest.mark.asyncio
async def test_async_single_attempt_does_not_sleep_or_notify(monkeypatch):
    sleeps = []
    retries = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("raganything.resilience.asyncio.sleep", fake_sleep)

    def on_retry(exc, attempt, delay):
        retries.append((type(exc), attempt, delay))

    calls = 0

    @async_retry(
        max_attempts=1,
        base_delay=0,
        max_delay=0,
        jitter=False,
        on_retry=on_retry,
    )
    async def func():
        nonlocal calls
        calls += 1
        raise TimeoutError("down")

    with pytest.raises(TimeoutError, match="down"):
        await func()

    assert calls == 1
    assert sleeps == []
    assert retries == []


@pytest.mark.asyncio
async def test_async_zero_delays_retry_immediately(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("raganything.resilience.asyncio.sleep", fake_sleep)
    calls = 0

    @async_retry(max_attempts=2, base_delay=0, max_delay=0, jitter=False)
    async def func():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("down")
        return "ok"

    assert await func() == "ok"
    assert calls == 2
    assert sleeps == [0.0]
