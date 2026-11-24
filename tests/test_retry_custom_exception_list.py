"""Caller-supplied retry exception lists must replace the defaults.

An empty list means "do not retry." Treating that as missing would turn a
deliberate no-retry embedding call back into the network-error default.
"""

import asyncio
import importlib.util
from pathlib import Path


def _load_resilience():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"
    spec = importlib.util.spec_from_file_location(
        "resilience_custom_retry_list_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resilience = _load_resilience()
retry = resilience.retry
async_retry = resilience.async_retry


class ManualRejected(Exception):
    """Transient rejection from a manual-processing worker."""


class ManualRateLimited(ManualRejected):
    """Subclass that must follow the parent retry rule."""


class ManualPermanent(Exception):
    """Terminal rejection that must not be retried."""


def test_custom_list_replaces_defaults_and_includes_subclasses():
    attempts = {"limited": 0, "connection": 0, "permanent": 0}
    retries = []

    def on_retry(exc, attempt, delay):
        retries.append((type(exc), attempt, delay))

    @retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(ManualRejected,),
        on_retry=on_retry,
    )
    def rate_limited():
        attempts["limited"] += 1
        if attempts["limited"] < 3:
            raise ManualRateLimited("embedding queue full")
        return "bearing seat"

    @retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(ManualRejected,),
        on_retry=on_retry,
    )
    def connection_reset():
        attempts["connection"] += 1
        raise ConnectionError("embedding host reset")

    @retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(ManualRejected,),
    )
    def permanent():
        attempts["permanent"] += 1
        raise ManualPermanent("manual checksum mismatch")

    assert rate_limited() == "bearing seat"
    assert attempts["limited"] == 3
    assert retries == [
        (ManualRateLimited, 1, 0.0),
        (ManualRateLimited, 2, 0.0),
    ]

    try:
        connection_reset()
    except ConnectionError as exc:
        assert str(exc) == "embedding host reset"
    else:
        raise AssertionError("ConnectionError was swallowed")

    try:
        permanent()
    except ManualPermanent as exc:
        assert str(exc) == "manual checksum mismatch"
    else:
        raise AssertionError("ManualPermanent was swallowed")

    assert attempts["connection"] == 1
    assert attempts["permanent"] == 1
    assert len(retries) == 2


def test_empty_retry_list_does_not_restore_default_network_errors():
    calls = {"count": 0}

    @retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=[],
    )
    def embed_manual():
        calls["count"] += 1
        raise ConnectionError("embedding host reset")

    @retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(),
    )
    def rejected_once():
        calls["count"] += 1
        raise ManualRejected("worker rejected the page")

    try:
        embed_manual()
    except ConnectionError as exc:
        assert str(exc) == "embedding host reset"
    else:
        raise AssertionError("empty retry list retried or swallowed the error")

    try:
        rejected_once()
    except ManualRejected as exc:
        assert str(exc) == "worker rejected the page"
    else:
        raise AssertionError("empty retry tuple retried ManualRejected")

    assert calls["count"] == 2


def test_async_custom_list_accepts_sync_callback_and_empty_tuple():
    attempts = {"limited": 0, "connection": 0, "empty": 0}
    retries = []

    def on_retry(exc, attempt, delay):
        retries.append((type(exc), attempt, delay))

    @async_retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(ManualRejected,),
        on_retry=on_retry,
    )
    async def rate_limited():
        attempts["limited"] += 1
        if attempts["limited"] < 2:
            raise ManualRateLimited("embedding queue full")
        return "torque spec"

    @async_retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(ManualRejected,),
        on_retry=on_retry,
    )
    async def connection_reset():
        attempts["connection"] += 1
        raise ConnectionError("embedding host reset")

    @async_retry(
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        jitter=False,
        retryable_exceptions=(),
        on_retry=on_retry,
    )
    async def no_retry():
        attempts["empty"] += 1
        raise ConnectionError("pool exhausted")

    async def scenario():
        assert await rate_limited() == "torque spec"
        try:
            await connection_reset()
        except ConnectionError as exc:
            assert str(exc) == "embedding host reset"
        else:
            raise AssertionError("ConnectionError was swallowed")
        try:
            await no_retry()
        except ConnectionError as exc:
            assert str(exc) == "pool exhausted"
        else:
            raise AssertionError("empty async retry tuple swallowed the error")

    asyncio.run(scenario())

    assert attempts == {"limited": 2, "connection": 1, "empty": 1}
    assert retries == [(ManualRateLimited, 1, 0.0)]
