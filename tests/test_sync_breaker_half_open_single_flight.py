"""Sync circuit breaker allows one half-open probe and rejects the rest.

The async decorator's single-flight behavior is covered elsewhere. Sync
``retry`` callers (embedding and parser helpers) use ``CircuitBreaker.__call__``.
A second caller that slips through while the probe is still running would
stampede a recovering host. A rejection must not be recorded as another
upstream failure, or the probe's later success could not close the breaker.
"""

import threading
import time

import pytest

from raganything.resilience import CircuitBreaker


def test_sync_half_open_rejects_second_caller_while_trial_runs():
    breaker = CircuitBreaker(
        failure_threshold=1, reset_timeout=60.0, name="bearing-embed"
    )

    @breaker
    def fail_closed():
        raise ConnectionError("embedding host closed the socket")

    with pytest.raises(ConnectionError, match="embedding host closed the socket"):
        fail_closed()
    assert breaker.state == "open"
    assert breaker._failure_count == 1

    # Expire the open window without sleeping.
    breaker._last_failure_time = time.time() - breaker.reset_timeout - 1
    assert breaker.state == "half-open"
    assert breaker._trial_in_flight is False

    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    results = []
    errors = []

    @breaker
    def probe():
        calls["n"] += 1
        entered.set()
        assert release.wait(timeout=2)
        return "probe ok"

    def run_probe():
        try:
            results.append(probe())
        except Exception as exc:
            errors.append(exc)

    holder = threading.Thread(target=run_probe)
    holder.start()
    try:
        assert entered.wait(timeout=2)
        assert calls["n"] == 1
        assert breaker._trial_in_flight is True
        assert breaker.state == "half-open"

        with pytest.raises(
            CircuitBreaker.CircuitBreakerOpen, match="half-open — trial in progress"
        ):
            probe()

        assert calls["n"] == 1
        assert breaker._failure_count == 1
        assert breaker.state == "half-open"
        assert breaker._trial_in_flight is True
    finally:
        release.set()
        holder.join(timeout=2)

    assert holder.is_alive() is False
    assert errors == []
    assert results == ["probe ok"]
    assert breaker.state == "closed"
    assert breaker._failure_count == 0
    assert breaker._trial_in_flight is False

    assert probe() == "probe ok"
    assert calls["n"] == 2
