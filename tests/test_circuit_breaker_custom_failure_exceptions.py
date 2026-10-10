"""A custom failure set must replace the default network exceptions.

Callers can tell the breaker which errors are upstream failures. Passing
``ValueError`` must open the circuit on that type alone. A ``ConnectionError``
must not count, and a non-listed error during half-open must release the
single-flight gate so the next call can run.
"""

import time

import pytest

from raganything.resilience import CircuitBreaker


def test_configured_exception_opens_breaker_and_default_network_error_does_not():
    cb = CircuitBreaker(
        failure_threshold=1,
        reset_timeout=60.0,
        name="custom-failures",
        failure_exceptions=(ValueError,),
    )

    @cb
    def fail_value():
        raise ValueError("configured upstream")

    with pytest.raises(ValueError, match="configured upstream"):
        fail_value()

    assert cb.state == "open"
    with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
        fail_value()

    other = CircuitBreaker(
        failure_threshold=1,
        reset_timeout=60.0,
        name="ignore-connection",
        failure_exceptions=(ValueError,),
    )

    @other
    def fail_connection():
        raise ConnectionError("not in the custom set")

    with pytest.raises(ConnectionError, match="not in the custom set"):
        fail_connection()

    assert other.state == "closed"
    assert other._failure_count == 0


def test_unlisted_half_open_error_clears_trial_and_allows_the_next_call():
    cb = CircuitBreaker(
        failure_threshold=2,
        reset_timeout=60.0,
        name="custom-half-open",
        failure_exceptions=(ValueError,),
    )

    @cb
    def fail_value():
        raise ValueError("configured upstream")

    with pytest.raises(ValueError):
        fail_value()
    with pytest.raises(ValueError):
        fail_value()
    assert cb.state == "open"

    cb._last_failure_time = time.time() - cb.reset_timeout - 1
    assert cb.state == "half-open"

    @cb
    def fail_connection():
        raise ConnectionError("local or unlisted")

    with pytest.raises(ConnectionError, match="local or unlisted"):
        fail_connection()

    assert cb.state == "half-open"
    assert cb._trial_in_flight is False
    assert cb._failure_count == 2

    @cb
    def recovered():
        return "ok"

    assert recovered() == "ok"
    assert cb.state == "closed"
    assert cb._failure_count == 0
