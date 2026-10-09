"""Default retry jitter must stay inside the documented 0–50% band.

``jitter=True`` is the default for both sync and async retry. Ingest and
query paths rely on it to avoid a thundering herd without stretching a
single backoff into an unrelated delay. Cap behavior with ``jitter=False``
is covered elsewhere; these cases stay under ``max_delay`` so they only
lock the multiplier applied to the uncapped exponential delay.
"""

import pytest

from raganything.resilience import async_retry, retry


def _uniform_sequence(*values):
    pending = iter(values)

    def fake_uniform(low, high):
        assert (low, high) == (0, 0.5)
        return next(pending)

    return fake_uniform


def test_sync_retry_default_jitter_scales_uncapped_backoff(monkeypatch):
    sleeps = []
    reported = []
    monkeypatch.setattr(
        "raganything.resilience.time.sleep", lambda delay: sleeps.append(delay)
    )
    monkeypatch.setattr("random.uniform", _uniform_sequence(0.0, 0.5))

    def on_retry(exc, attempt, delay):
        reported.append((type(exc).__name__, attempt, delay))

    @retry(
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        exponential_base=2.0,
        retryable_exceptions=[ConnectionError],
        on_retry=on_retry,
    )
    def always_fail():
        raise ConnectionError("upstream")

    with pytest.raises(ConnectionError, match="upstream"):
        always_fail()

    # 2.0 * (1 + 0), then 4.0 * (1 + 0.5). on_retry sees the jittered delay.
    assert sleeps == [2.0, 6.0]
    assert reported == [("ConnectionError", 1, 2.0), ("ConnectionError", 2, 6.0)]


@pytest.mark.asyncio
async def test_async_retry_default_jitter_scales_uncapped_backoff(monkeypatch):
    sleeps = []
    reported = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("raganything.resilience.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("random.uniform", _uniform_sequence(0.5, 0.0))

    def on_retry(exc, attempt, delay):
        reported.append((attempt, delay))

    @async_retry(
        max_attempts=3,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=3.0,
        retryable_exceptions=[TimeoutError],
        on_retry=on_retry,
    )
    async def always_fail():
        raise TimeoutError("upstream")

    with pytest.raises(TimeoutError, match="upstream"):
        await always_fail()

    # 1.0 * 1.5, then 3.0 * 1.0.
    assert sleeps == [1.5, 3.0]
    assert reported == [(1, 1.5), (2, 3.0)]
