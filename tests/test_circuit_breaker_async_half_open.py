"""Async CircuitBreaker half-open single-flight and trial-gate behavior."""

import asyncio

import pytest

from raganything.resilience import CircuitBreaker


def _open_breaker(cb: CircuitBreaker) -> None:
    @cb
    def fail():
        raise ConnectionError("fail")

    with pytest.raises(ConnectionError):
        fail()


class TestCircuitBreakerAsync:
    @pytest.mark.asyncio
    async def test_closed_allows_async_calls(self):
        cb = CircuitBreaker(failure_threshold=3, name="async-test")

        @cb.async_call
        async def ok():
            return "ok"

        assert await ok() == "ok"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_opens_after_async_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=10.0, name="async-test")

        @cb.async_call
        async def fail():
            raise ConnectionError("fail")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await fail()

        assert cb.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            await fail()

    @pytest.mark.asyncio
    async def test_half_open_allows_single_async_trial(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="async-test")

        _open_breaker(cb)
        await asyncio.sleep(0.06)
        assert cb.state == "half-open"

        executed = 0

        @cb.async_call
        async def trial():
            nonlocal executed
            executed += 1
            await asyncio.sleep(0.05)
            return "ok"

        results = await asyncio.gather(
            trial(), trial(), trial(), trial(), trial(), return_exceptions=True
        )

        oks = [r for r in results if r == "ok"]
        rejected = [
            r for r in results if isinstance(r, CircuitBreaker.CircuitBreakerOpen)
        ]
        assert executed == 1
        assert oks == ["ok"]
        assert len(rejected) == 4
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_async_application_error_clears_half_open_trial(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="async-test")
        _open_breaker(cb)
        await asyncio.sleep(0.06)
        assert cb.state == "half-open"

        @cb.async_call
        async def buggy():
            raise TypeError("bug")

        with pytest.raises(TypeError):
            await buggy()

        assert cb.state == "half-open"
        assert cb._trial_in_flight is False

        @cb.async_call
        async def recovered():
            return "ok"

        assert await recovered() == "ok"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_failed_half_open_async_probe_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="async-test")
        _open_breaker(cb)
        await asyncio.sleep(0.06)

        @cb.async_call
        async def still_fail():
            raise TimeoutError("still down")

        with pytest.raises(TimeoutError):
            await still_fail()

        assert cb.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            await still_fail()
