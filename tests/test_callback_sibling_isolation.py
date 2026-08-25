"""Callback dispatch must isolate handler failures and record query errors.

One exploding observer must not drop parse/query events for remaining
callbacks. MetricsCallback must count query failures without treating them
as successful queries.
"""

from raganything.callbacks import (
    CallbackManager,
    MetricsCallback,
    ProcessingCallback,
)


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_parse_start(self, file_path, **kw):
        self.events.append(("parse_start", file_path))

    def on_query_error(self, query, mode="", error="", **kw):
        self.events.append(("query_error", query, mode, str(error)))


class BoomCallback(ProcessingCallback):
    def on_parse_start(self, **kw):
        raise RuntimeError("observer crashed")

    def on_query_error(self, **kw):
        raise RuntimeError("metrics observer crashed")


def test_later_callback_still_runs_after_earlier_handler_raises():
    mgr = CallbackManager()
    boom = BoomCallback()
    good = RecordingCallback()
    mgr.register(boom)
    mgr.register(good)

    mgr.dispatch("on_parse_start", file_path="manual.pdf", parser="mineru")

    assert good.events == [("parse_start", "manual.pdf")]


def test_earlier_callback_still_ran_when_later_handler_raises():
    mgr = CallbackManager()
    good = RecordingCallback()
    boom = BoomCallback()
    mgr.register(good)
    mgr.register(boom)

    mgr.dispatch("on_parse_start", file_path="manual.pdf")

    assert good.events == [("parse_start", "manual.pdf")]


def test_metrics_on_query_error_records_without_counting_success():
    metrics = MetricsCallback()
    metrics.on_query_error(query="pump trip?", error=TimeoutError("upstream timeout"))

    assert metrics.metrics["queries_executed"] == 0
    assert metrics.metrics["total_query_time"] == 0.0
    assert metrics.metrics["errors"] == [
        {"file": None, "error": "upstream timeout", "stage": "query"}
    ]

    summary = metrics.summary()
    assert "[query]" in summary
    assert "upstream timeout" in summary


def test_query_error_dispatch_reaches_metrics_after_sibling_failure():
    mgr = CallbackManager()
    boom = BoomCallback()
    metrics = MetricsCallback()
    mgr.register(boom)
    mgr.register(metrics)

    mgr.dispatch(
        "on_query_error",
        query="bearing temp",
        mode="mix",
        error=RuntimeError("llm down"),
    )

    assert metrics.metrics["errors"] == [
        {"file": None, "error": "llm down", "stage": "query"}
    ]
    assert metrics.metrics["queries_executed"] == 0
