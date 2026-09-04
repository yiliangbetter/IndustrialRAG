"""Retry backoff with jitter=False must follow exact exponential delays.

Jitter is on by default, so operators disable it for reproducible ingest
retries. If jitter stays applied, delays become non-deterministic and the
documented exponential_base contract is lost. max_delay capping is covered
by an open coverage PR; this file only locks the uncapped sequence.
"""

from unittest.mock import patch

import pytest

from raganything.resilience import async_retry, retry


class TestJitterFalseExponential:
    def test_sync_retry_sleeps_exact_exponential_delays(self):
        sleeps = []

        @retry(
            max_attempts=4,
            base_delay=0.5,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
            retryable_exceptions=[ConnectionError],
        )
        def always_fail():
            raise ConnectionError("transient")

        with patch("raganything.resilience.time.sleep", side_effect=sleeps.append):
            with pytest.raises(ConnectionError, match="transient"):
                always_fail()

        assert sleeps == [0.5, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_async_retry_sleeps_exact_exponential_delays(self):
        sleeps = []

        @async_retry(
            max_attempts=4,
            base_delay=0.5,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
            retryable_exceptions=[TimeoutError],
        )
        async def always_fail():
            raise TimeoutError("transient")

        async def fake_sleep(delay):
            sleeps.append(delay)

        with patch("raganything.resilience.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(TimeoutError, match="transient"):
                await always_fail()

        assert sleeps == [0.5, 1.0, 2.0]
