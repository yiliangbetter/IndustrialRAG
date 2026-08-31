"""Sync circuit breaker: application errors during half-open must not stick.

#135 covers the async decorator. A TypeError while the sync breaker is
half-open must clear ``_trial_in_flight`` so the next call is not
permanently rejected as if a trial were still running.
"""

import pytest

from raganything.resilience import CircuitBreaker


def test_half_open_typeerror_clears_trial_gate_so_next_call_proceeds():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0, name="sync-app-err")

    @cb
    def fail_upstream():
        raise ConnectionError("upstream")

    with pytest.raises(ConnectionError):
        fail_upstream()

    @cb
    def application_bug():
        raise TypeError("local bug")

    with pytest.raises(TypeError, match="local bug"):
        application_bug()

    assert cb.state == "half-open"
    assert cb._trial_in_flight is False
    assert cb._failure_count == 1

    @cb
    def recovered():
        return "ok"

    assert recovered() == "ok"
    assert cb.state == "closed"
    assert cb._failure_count == 0
