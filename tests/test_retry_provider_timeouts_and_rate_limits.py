"""Default retry of OpenAI and httpx failures that are not builtin OS errors.

Those exceptions are not subclasses of ConnectionError or TimeoutError. Tests that
only raise socket errors stay green if the provider entries are dropped.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")


def _load_resilience():
    path = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"
    spec = importlib.util.spec_from_file_location(
        "resilience_provider_retry_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resilience = _load_resilience()

_REQUEST = httpx.Request("POST", "https://llm.example.test/v1/chat")


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST)


PROVIDER_ERRORS = [
    openai.RateLimitError(
        "rate limit",
        response=_response(429),
        body=None,
    ),
    openai.InternalServerError(
        "upstream 500",
        response=_response(500),
        body=None,
    ),
    openai.APITimeoutError(request=_REQUEST),
    openai.APIConnectionError(message="embedding host reset", request=_REQUEST),
    httpx.PoolTimeout("pool exhausted"),
    httpx.ReadTimeout("read timed out"),
    httpx.WriteTimeout("write timed out"),
    httpx.ConnectError("connect failed"),
]


def _error_id(exc: BaseException) -> str:
    return type(exc).__name__


@pytest.mark.parametrize("exc", PROVIDER_ERRORS, ids=_error_id)
def test_default_sync_retry_retries_provider_errors(exc):
    assert not issubclass(type(exc), (ConnectionError, TimeoutError, OSError))
    assert type(exc) in resilience._DEFAULT_RETRYABLE

    calls = []
    seen = []

    @resilience.retry(
        max_attempts=2,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        on_retry=lambda error, attempt, delay: seen.append(
            (type(error), attempt, delay)
        ),
    )
    def embed():
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return "embedded"

    assert embed() == "embedded"
    assert calls == [1, 1]
    assert seen == [(type(exc), 1, 0.0)]


def test_default_sync_retry_does_not_retry_permission_error():
    calls = []

    @resilience.retry(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=False)
    def read_manual():
        calls.append(1)
        raise PermissionError("manual.pdf")

    with pytest.raises(PermissionError, match="manual.pdf"):
        read_manual()
    assert calls == [1]
    assert PermissionError not in resilience._DEFAULT_RETRYABLE


def test_default_async_retry_awaits_on_retry_for_api_timeout():
    exc = openai.APITimeoutError(request=_REQUEST)
    seen = []

    async def on_retry(error, attempt, delay):
        seen.append((type(error), attempt, delay))

    async def scenario():
        calls = []

        @resilience.async_retry(
            max_attempts=2,
            base_delay=0.0,
            max_delay=0.0,
            jitter=False,
            on_retry=on_retry,
        )
        async def embed():
            calls.append(1)
            if len(calls) == 1:
                raise exc
            return "embedded"

        assert await embed() == "embedded"
        assert calls == [1, 1]

    asyncio.run(scenario())
    assert seen == [(openai.APITimeoutError, 1, 0.0)]


def test_breaker_opens_on_internal_server_error_not_permission():
    exc = openai.InternalServerError(
        "upstream 500",
        response=_response(500),
        body=None,
    )
    breaker = resilience.CircuitBreaker(
        failure_threshold=1,
        reset_timeout=60.0,
        name="manual-llm",
    )
    calls = []

    @breaker
    def embed():
        calls.append(1)
        raise exc

    with pytest.raises(openai.InternalServerError, match="upstream 500"):
        embed()
    assert breaker._failure_count == 1
    assert breaker.state == "open"
    with pytest.raises(breaker.CircuitBreakerOpen, match="manual-llm"):
        embed()
    assert calls == [1]

    closed = resilience.CircuitBreaker(
        failure_threshold=1,
        reset_timeout=60.0,
        name="manual-read",
    )

    @closed
    def read_manual():
        raise PermissionError("manual.pdf")

    with pytest.raises(PermissionError, match="manual.pdf"):
        read_manual()
    assert closed._failure_count == 0
    assert closed.state == "closed"


def test_async_breaker_opens_on_pool_timeout():
    exc = httpx.PoolTimeout("pool exhausted")
    breaker = resilience.CircuitBreaker(
        failure_threshold=1,
        reset_timeout=60.0,
        name="manual-pool",
    )
    calls = []

    @breaker.async_call
    async def embed():
        calls.append(1)
        raise exc

    async def scenario():
        with pytest.raises(httpx.PoolTimeout, match="pool exhausted"):
            await embed()
        assert breaker._failure_count == 1
        assert breaker.state == "open"
        with pytest.raises(breaker.CircuitBreakerOpen, match="manual-pool"):
            await embed()

    asyncio.run(scenario())
    assert calls == [1]
