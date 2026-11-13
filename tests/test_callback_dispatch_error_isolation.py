"""A failing listener must not hide the rest of a callback dispatch.

Metrics and audit hooks are registered beside user callbacks. If one listener
raises, later listeners still have to see the original query error, and the
event log has to keep that error rather than the listener's exception.
"""

import importlib.util
import logging
import sys
from pathlib import Path


def _load_callbacks():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "callbacks.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_callbacks_error_isolation_1113", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass string annotations look the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_raising_listener_does_not_stop_later_callbacks(caplog):
    callbacks = _load_callbacks()
    manager = callbacks.CallbackManager()
    seen = []

    class Broken(callbacks.ProcessingCallback):
        def on_query_error(self, query, error="", **kwargs):
            seen.append(("broken", query, str(error)))
            raise RuntimeError("listener exploded")

    class CountingMetrics(callbacks.MetricsCallback):
        def on_query_error(self, query, error="", **kw):
            seen.append(("metrics", query, str(error)))
            super().on_query_error(query, error=error, **kw)

    class Recorder(callbacks.ProcessingCallback):
        def on_query_error(self, query, error="", **kwargs):
            seen.append(("record", query, str(error)))

    metrics = CountingMetrics()
    manager.register(Broken())
    manager.register(metrics)
    manager.register(Recorder())
    manager.enable_event_log(True)

    query_error = RuntimeError("index missing")
    with caplog.at_level(logging.ERROR, logger=callbacks.logger.name):
        manager.dispatch(
            "on_query_error",
            query="torque spec",
            mode="mix",
            error=query_error,
        )

    assert seen == [
        ("broken", "torque spec", "index missing"),
        ("metrics", "torque spec", "index missing"),
        ("record", "torque spec", "index missing"),
    ]
    assert metrics.metrics["errors"] == [
        {"file": None, "error": "index missing", "stage": "query"}
    ]
    assert metrics.metrics["documents_failed"] == 0
    assert len(manager.event_log) == 1
    event = manager.event_log[0]
    assert event.event_type == "on_query_error"
    assert event.error == "index missing"
    assert event.details["error"] is query_error
    assert event.file_path is None
    assert "listener exploded" in caplog.text
    assert "index missing" not in caplog.text

    seen.clear()
    manager.dispatch(
        "on_query_error",
        query="bearing clearance",
        mode="mix",
        error=ValueError("still down"),
    )

    assert [item[0] for item in seen] == ["broken", "metrics", "record"]
    assert seen[2] == ("record", "bearing clearance", "still down")
    assert metrics.metrics["errors"][1] == {
        "file": None,
        "error": "still down",
        "stage": "query",
    }
    assert metrics.metrics["documents_failed"] == 0
    assert [logged.error for logged in manager.event_log] == [
        "index missing",
        "still down",
    ]
    assert manager.event_log[1].details["error"].args == ("still down",)
