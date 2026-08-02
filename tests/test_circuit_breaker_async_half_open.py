"""Regression tests for CircuitBreaker async decorator and half-open edges.

Complements tests/test_resilience.py sync coverage with:
- failed half-open probe reopening immediately
- async_call success / failure parity
- non-retryable errors clearing the half-open trial gate
"""

import asyncio
import time

import pytest

from raganything.resilience import CircuitBreaker


class TestHalfOpenFailureReopens:
    def test_failed_probe_reopens_immediately(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="reopen")

        @cb
        def fail():
            raise ConnectionError("upstream")

        with pytest.raises(ConnectionError):
            fail()
        assert cb.state == "open"

        time.sleep(0.06)
        assert cb.state == "half-open"

        with pytest.raises(ConnectionError):
            fail()
        assert cb.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            fail()

    def test_non_retryable_error_clears_trial_gate(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="trial-gate")

        @cb
        def open_breaker():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            open_breaker()
        time.sleep(0.06)
        assert cb.state == "half-open"

        @cb
        def buggy():
            raise TypeError("app bug")

        with pytest.raises(TypeError):
            buggy()

        # Trial gate must be cleared so a later call can run (and succeed).
        assert cb._trial_in_flight is False
        assert cb.state == "half-open"

        @cb
        def recover():
            return "ok"

        assert recover() == "ok"
        assert cb.state == "closed"


class TestAsyncCircuitBreaker:
    @pytest.mark.asyncio
    async def test_async_success_and_open(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05, name="async")

        @cb.async_call
        async def flaky():
            raise TimeoutError("boom")

        with pytest.raises(TimeoutError):
            await flaky()
        with pytest.raises(TimeoutError):
            await flaky()
        assert cb.state == "open"
        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            await flaky()

    @pytest.mark.asyncio
    async def test_async_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="async-ok")
        calls = {"n": 0}

        @cb.async_call
        async def sometimes():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("first")
            return "recovered"

        with pytest.raises(ConnectionError):
            await sometimes()
        await asyncio.sleep(0.06)
        assert cb.state == "half-open"
        assert await sometimes() == "recovered"
        assert cb.state == "closed"
