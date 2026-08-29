"""Query failures must be recorded without counting as successful queries.

MetricsCallback is the built-in operator dashboard. Incrementing
queries_executed on error would hide outages; omitting the error list would
hide the failure entirely. Event-log serialization must stringify exceptions
so later inspection does not store live exception objects.
"""

from raganything.callbacks import CallbackManager, MetricsCallback


def test_query_error_records_error_without_incrementing_executed():
    metrics = MetricsCallback()
    metrics.on_query_error(
        query="what is the rating?", error=RuntimeError("index missing")
    )

    assert metrics.metrics["queries_executed"] == 0
    assert metrics.metrics["total_query_time"] == 0.0
    assert metrics.metrics["errors"] == [
        {"file": None, "error": "index missing", "stage": "query"}
    ]
    summary = metrics.summary()
    assert "Queries executed    : 0" in summary
    assert "Errors              : 1" in summary
    assert "[query]" in summary
    assert "index missing" in summary


def test_query_complete_still_increments_after_an_error():
    metrics = MetricsCallback()
    metrics.on_query_error(query="bad", error="timeout")
    metrics.on_query_complete(query="ok", duration_seconds=0.25)

    assert metrics.metrics["queries_executed"] == 1
    assert metrics.metrics["total_query_time"] == 0.25
    assert len(metrics.metrics["errors"]) == 1


def test_event_log_stringifies_error_payload():
    manager = CallbackManager()
    manager.enable_event_log(True)
    boom = RuntimeError("cuda oom")

    manager.dispatch("on_query_error", query="q", mode="mix", error=boom)

    assert len(manager.event_log) == 1
    event = manager.event_log[0]
    assert event.event_type == "on_query_error"
    assert event.error == "cuda oom"
    assert event.details["error"] is boom
