"""async_retry edges and sync circuit-breaker half-open failure.

Existing resilience tests cover sync retry non-retryable/on_retry and the
async breaker. async_retry's non-retryable path, on_retry (including
awaited coroutines), exponential_base validation, and a failed *sync*
half-open probe reopening the breaker were untested.
"""

import time

import pytest

from raganything.resilience import CircuitBreaker, async_retry, retry


class TestAsyncRetryEdges:
    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable(self):
        call_count = 0

        @async_retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ConnectionError],
        )
        async def type_error_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await type_error_func()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_sync_callback(self):
        retries_seen = []

        def on_retry_cb(exc, attempt, delay):
            retries_seen.append((type(exc).__name__, attempt, delay >= 0))

        call_count = 0

        @async_retry(
            max_attempts=3,
            base_delay=0.01,
            jitter=False,
            retryable_exceptions=[OSError],
            on_retry=on_retry_cb,
        )
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("oops")
            return "ok"

        assert await flaky() == "ok"
        assert retries_seen == [("OSError", 1, True), ("OSError", 2, True)]

    @pytest.mark.asyncio
    async def test_on_retry_async_callback_is_awaited(self):
        retries_seen = []

        async def on_retry_cb(exc, attempt, delay):
            retries_seen.append(attempt)

        call_count = 0

        @async_retry(
            max_attempts=2,
            base_delay=0.01,
            jitter=False,
            retryable_exceptions=[TimeoutError],
            on_retry=on_retry_cb,
        )
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("slow")
            return "recovered"

        assert await flaky() == "recovered"
        assert retries_seen == [1]

    def test_async_retry_rejects_non_positive_exponential_base(self):
        with pytest.raises(ValueError, match="exponential_base"):

            @async_retry(exponential_base=0)
            async def _afunc():
                return "ok"

        with pytest.raises(ValueError, match="exponential_base"):

            @async_retry(exponential_base=-2.0)
            async def _afunc2():
                return "ok"

    def test_retry_rejects_non_positive_exponential_base(self):
        with pytest.raises(ValueError, match="exponential_base"):

            @retry(exponential_base=0)
            def _func():
                return "ok"


class TestSyncHalfOpenFailure:
    def test_failed_half_open_probe_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0, name="sync-test")

        @cb
        def fail():
            raise ConnectionError("upstream down")

        with pytest.raises(ConnectionError):
            fail()
        assert cb.state == "open"

        # Expire the open window without sleeping.
        cb._last_failure_time = time.time() - cb.reset_timeout - 1
        assert cb.state == "half-open"

        with pytest.raises(ConnectionError):
            fail()
        assert cb.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            fail()
