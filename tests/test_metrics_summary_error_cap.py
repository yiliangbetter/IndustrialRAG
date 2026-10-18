"""Metrics summaries count every error and print only the first five.

A batch that fails many files still has to report the full failure count.
Printing the whole list in ``MetricsCallback.summary`` floods operator logs;
dropping the count hides how many files failed.
"""

from raganything.callbacks import MetricsCallback


def test_summary_counts_all_errors_and_prints_the_first_five():
    metrics = MetricsCallback()
    for index in range(6):
        metrics.on_document_error(
            file_path=f"manual-{index}.pdf",
            error=f"parse-boom-{index}",
            stage="parse",
        )

    summary = metrics.summary()

    assert "Documents failed    : 6" in summary
    assert "Errors              : 6" in summary
    for index in range(5):
        assert f"[parse] manual-{index}.pdf: parse-boom-{index}" in summary
    assert "manual-5.pdf" not in summary
    assert "parse-boom-5" not in summary
    assert len(metrics.metrics["errors"]) == 6
