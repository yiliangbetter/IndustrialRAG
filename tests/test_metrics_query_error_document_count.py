"""Query errors must not be counted as failed documents.

``MetricsCallback`` keeps one error list for ingest failures and query
failures. A question that raises is an operator-visible error, and it must
not increment ``documents_failed``. The summary still prints only the first
five entries of that shared list, so a later query error is counted and omitted
from the printed lines.
"""

import importlib.util
import sys
from pathlib import Path


def _load_callbacks():
    """Load callbacks.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "callbacks.py"
    module_name = "_raganything_callbacks_query_error_doc_count_1029"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Dataclass modules must be registered before exec_module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MetricsCallback = _load_callbacks().MetricsCallback


def test_query_error_does_not_count_as_a_failed_document():
    metrics = MetricsCallback()
    metrics.on_document_complete(file_path="manual.pdf")
    metrics.on_query_error(
        query="rated current?",
        error=RuntimeError("index missing"),
    )

    assert metrics.metrics["documents_processed"] == 1
    assert metrics.metrics["documents_failed"] == 0
    assert metrics.metrics["queries_executed"] == 0
    assert metrics.metrics["total_query_time"] == 0.0
    assert metrics.metrics["errors"] == [
        {"file": None, "error": "index missing", "stage": "query"}
    ]

    summary = metrics.summary()
    assert "Documents processed : 1" in summary
    assert "Documents failed    : 0" in summary
    assert "Queries executed    : 0" in summary
    assert "Errors              : 1" in summary
    assert "  - [query] None: index missing" in summary
    assert "rated current?" not in summary


def test_query_errors_share_the_five_line_summary_cap():
    metrics = MetricsCallback()
    for index in range(4):
        metrics.on_document_error(
            file_path=f"manual-{index}.pdf",
            error=f"parse-boom-{index}",
            stage="parse",
        )
    metrics.on_query_error(query="first question", error="first-query")
    metrics.on_query_error(query="second question", error="second-query")

    assert metrics.metrics["documents_failed"] == 4
    assert metrics.metrics["queries_executed"] == 0
    assert len(metrics.metrics["errors"]) == 6
    assert metrics.metrics["errors"][4] == {
        "file": None,
        "error": "first-query",
        "stage": "query",
    }
    assert metrics.metrics["errors"][5]["error"] == "second-query"

    summary = metrics.summary()
    assert "Documents failed    : 4" in summary
    assert "Errors              : 6" in summary
    assert "  - [query] None: first-query" in summary
    for index in range(4):
        assert f"[parse] manual-{index}.pdf: parse-boom-{index}" in summary
    assert "second-query" not in summary
    assert "second question" not in summary
