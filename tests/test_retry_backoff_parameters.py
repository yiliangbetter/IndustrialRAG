"""Regression tests for retry/async_retry backoff parameter behavior.

Complements tests/test_resilience.py (basic retry/circuit paths) and open PR #86
(async CircuitBreaker / half-open). Focus: max_delay capping, jitter off,
exponential_base validation, and async on_retry awaiting coroutines.
"""

import asyncio

import pytest

from raganything.resilience import async_retry, retry


class TestRetryBackoffParameters:
    def test_rejects_non_positive_exponential_base(self):
        with pytest.raises(ValueError, match="exponential_base"):

            @retry(exponential_base=0)
            def _zero():
                return "ok"

        with pytest.raises(ValueError, match="exponential_base"):

            @retry(exponential_base=-2.0)
            def _negative():
                return "ok"

    def test_max_delay_caps_computed_backoff(self):
        delays = []

        def on_retry(_exc, _attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry(
            max_attempts=4,
            base_delay=10.0,
            max_delay=0.5,
            exponential_base=2.0,
            jitter=False,
            retryable_exceptions=[ConnectionError],
            on_retry=on_retry,
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ConnectionError("transient")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 4
        # Without cap: 10, 20, 40 — with max_delay=0.5 all sleeps are capped.
        assert delays == [0.5, 0.5, 0.5]

    def test_jitter_disabled_uses_exact_exponential_delay(self):
        delays = []

        def on_retry(_exc, _attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry(
            max_attempts=4,
            base_delay=0.2,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
            retryable_exceptions=[TimeoutError],
            on_retry=on_retry,
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise TimeoutError("slow")
            return "recovered"

        assert flaky() == "recovered"
        assert delays == [0.2, 0.4, 0.8]


class TestAsyncRetryBackoffParameters:
    @pytest.mark.asyncio
    async def test_rejects_non_positive_exponential_base(self):
        with pytest.raises(ValueError, match="exponential_base"):

            @async_retry(exponential_base=0)
            async def _zero():
                return "ok"

        with pytest.raises(ValueError, match="exponential_base"):

            @async_retry(exponential_base=-1.0)
            async def _negative():
                return "ok"

    @pytest.mark.asyncio
    async def test_max_delay_caps_and_awaits_async_on_retry(self):
        delays = []
        callback_states = []

        async def on_retry(_exc, attempt, delay):
            delays.append(delay)
            callback_states.append(("start", attempt))
            await asyncio.sleep(0)
            callback_states.append(("done", attempt))

        call_count = 0

        @async_retry(
            max_attempts=3,
            base_delay=5.0,
            max_delay=0.25,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=[ConnectionError],
            on_retry=on_retry,
        )
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        assert await flaky() == "ok"
        assert call_count == 3
        assert delays == [0.25, 0.25]
        assert callback_states == [
            ("start", 1),
            ("done", 1),
            ("start", 2),
            ("done", 2),
        ]

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable_errors(self):
        call_count = 0

        @async_retry(
            max_attempts=4,
            base_delay=0.01,
            retryable_exceptions=[ConnectionError],
        )
        async def buggy():
            nonlocal call_count
            call_count += 1
            raise ValueError("application bug")

        with pytest.raises(ValueError, match="application bug"):
            await buggy()
        assert call_count == 1
