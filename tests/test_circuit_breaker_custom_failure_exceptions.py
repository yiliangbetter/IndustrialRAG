"""CircuitBreaker honors custom failure_exceptions instead of default network errors.

A misconfigured breaker that treats local bugs as upstream failures (or ignores
real upstream errors) either flaps open on application mistakes or never isolates
a failing LLM/embed call. These tests lock the custom exception filter.
"""

import pytest

from raganything.resilience import CircuitBreaker


class TestCircuitBreakerCustomFailureExceptions:
    def test_only_configured_exceptions_open_the_breaker(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            name="custom",
            failure_exceptions=(ValueError,),
        )

        @cb
        def raise_value():
            raise ValueError("bad input from upstream schema")

        with pytest.raises(ValueError):
            raise_value()

        assert cb.state == "open"
        assert cb._failure_count >= 1

        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            raise_value()

    def test_unlisted_exceptions_do_not_count_as_failures(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            name="custom",
            failure_exceptions=(ValueError,),
        )

        @cb
        def raise_connection():
            # Default retryable error, but not in this breaker's filter.
            raise ConnectionError("network")

        with pytest.raises(ConnectionError):
            raise_connection()

        assert cb.state == "closed"
        assert cb._failure_count == 0

        # Subsequent calls still execute because the breaker never opened.
        with pytest.raises(ConnectionError):
            raise_connection()
        assert cb.state == "closed"

    def test_application_bugs_still_do_not_trip_custom_breaker(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            name="custom",
            failure_exceptions=(ConnectionError, TimeoutError),
        )

        @cb
        def buggy():
            raise TypeError("local bug")

        with pytest.raises(TypeError):
            buggy()

        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_successful_call_resets_custom_breaker(self):
        cb = CircuitBreaker(
            failure_threshold=2,
            reset_timeout=30.0,
            name="custom",
            failure_exceptions=(OSError,),
        )

        @cb
        def flaky(should_fail):
            if should_fail:
                raise OSError("transient disk")
            return "ok"

        with pytest.raises(OSError):
            flaky(True)
        assert cb._failure_count == 1
        assert cb.state == "closed"

        assert flaky(False) == "ok"
        assert cb._failure_count == 0
        assert cb.state == "closed"
