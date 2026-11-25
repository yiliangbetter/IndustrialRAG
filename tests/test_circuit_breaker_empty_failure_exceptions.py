"""Empty circuit-breaker failure filters keep the default network errors.

``retry`` treats an empty ``retryable_exceptions`` sequence as an opt-out.
``CircuitBreaker`` uses truthiness, so ``[]`` and ``()`` fall back to the
default network errors instead of disabling the breaker or counting every
exception.
"""

import pytest

from raganything.resilience import CircuitBreaker


def test_empty_failure_exception_list_keeps_default_network_errors():
    breaker = CircuitBreaker(
        failure_threshold=1,
        failure_exceptions=[],
        name="empty-list",
    )
    calls = {"n": 0}

    @breaker
    def fail():
        calls["n"] += 1
        raise ConnectionError("connection reset")

    with pytest.raises(ConnectionError, match="connection reset"):
        fail()
    assert calls["n"] == 1
    assert breaker.state == "open"
    assert breaker._failure_count == 1

    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="is open"):
        fail()
    assert calls["n"] == 1

    local = CircuitBreaker(
        failure_threshold=1,
        failure_exceptions=[],
        name="empty-list-local",
    )

    @local
    def bug():
        raise ValueError("bad caption")

    with pytest.raises(ValueError, match="bad caption"):
        bug()
    assert local.state == "closed"
    assert local._failure_count == 0


@pytest.mark.asyncio
async def test_empty_failure_exception_tuple_keeps_default_network_errors():
    breaker = CircuitBreaker(
        failure_threshold=1,
        failure_exceptions=(),
        name="empty-tuple",
    )
    calls = {"n": 0}

    @breaker.async_call
    async def fail():
        calls["n"] += 1
        raise ConnectionError("connection reset")

    with pytest.raises(ConnectionError, match="connection reset"):
        await fail()
    assert calls["n"] == 1
    assert breaker.state == "open"
    assert breaker._failure_count == 1

    with pytest.raises(CircuitBreaker.CircuitBreakerOpen, match="is open"):
        await fail()
    assert calls["n"] == 1

    local = CircuitBreaker(
        failure_threshold=1,
        failure_exceptions=(),
        name="empty-tuple-local",
    )

    @local.async_call
    async def bug():
        raise ValueError("bad caption")

    with pytest.raises(ValueError, match="bad caption"):
        await bug()
    assert local.state == "closed"
    assert local._failure_count == 0
