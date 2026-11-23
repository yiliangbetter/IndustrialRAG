"""Retry config is rejected before an embedding or LLM call runs.

``retry`` and ``async_retry`` each validate their own arguments. A zero or
negative attempt count, a negative delay, or a non-positive backoff base must
raise when the decorator is built. The wrapped function is not called, and the
two decorators must keep the same checks.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_NAME = "resilience_retry_bounds_under_test"
_RESILIENCE_PATH = Path(__file__).resolve().parents[1] / "raganything" / "resilience.py"


def _load_resilience():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _RESILIENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resilience = _load_resilience()


def _embed_bearing_manual():
    raise AssertionError("bearing manual embedding must not run")


@pytest.mark.parametrize("decorator_name", ["retry", "async_retry"])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts must be >= 1"),
        ({"max_attempts": -1}, "max_attempts must be >= 1"),
        ({"base_delay": -0.1}, "base_delay and max_delay must be >= 0"),
        ({"max_delay": -1.0}, "base_delay and max_delay must be >= 0"),
        ({"exponential_base": 0}, "exponential_base must be > 0"),
        ({"exponential_base": -2.0}, "exponential_base must be > 0"),
    ],
)
def test_retry_rejects_invalid_bounds_before_calling(decorator_name, kwargs, message):
    decorator = getattr(resilience, decorator_name)

    with pytest.raises(ValueError, match=message):
        decorator(**kwargs)

    with pytest.raises(ValueError, match=message):
        decorator(**kwargs)(_embed_bearing_manual)


@pytest.mark.parametrize("decorator_name", ["retry", "async_retry"])
def test_attempt_count_is_checked_before_delay(decorator_name):
    """A broken attempt count is reported even when the delay is also invalid."""
    decorator = getattr(resilience, decorator_name)

    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        decorator(max_attempts=0, base_delay=-5, max_delay=-5, exponential_base=0)
