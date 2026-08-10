"""Regression tests for BatchParser CLI exit-code contracts.

CI and operator scripts rely on `raganything.batch_parser.main` returning
nonzero when construction fails or any file fails to parse. These paths are
not covered by library-focused batch_parser tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raganything.batch_parser import BatchProcessingResult, main


def _result(
    *,
    successful: list[str] | None = None,
    failed: list[str] | None = None,
    dry_run: bool = False,
) -> BatchProcessingResult:
    successful = successful or []
    failed = failed or []
    return BatchProcessingResult(
        successful_files=successful,
        failed_files=failed,
        total_files=len(successful) + len(failed),
        processing_time=0.01,
        errors={path: "boom" for path in failed},
        output_dir="/tmp/out",
        dry_run=dry_run,
    )


def test_cli_returns_zero_on_full_success(monkeypatch, capsys):
    fake = MagicMock()
    fake.process_batch.return_value = _result(successful=["a.pdf"])
    monkeypatch.setattr(
        "raganything.batch_parser.BatchParser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["batch_parser", "a.pdf", "--output", "/tmp/out"],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "Batch Processing Summary" in out
    fake.process_batch.assert_called_once()
    kwargs = fake.process_batch.call_args.kwargs
    assert kwargs["file_paths"] == ["a.pdf"]
    assert kwargs["output_dir"] == "/tmp/out"
    assert kwargs["dry_run"] is False


def test_cli_returns_one_when_any_file_failed(monkeypatch, capsys):
    fake = MagicMock()
    fake.process_batch.return_value = _result(
        successful=["ok.pdf"],
        failed=["bad.pdf"],
    )
    monkeypatch.setattr(
        "raganything.batch_parser.BatchParser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["batch_parser", "ok.pdf", "bad.pdf", "--output", "/tmp/out"],
    )

    assert main() == 1
    assert "Failed: 1" in capsys.readouterr().out


def test_cli_dry_run_returns_zero_even_with_listed_files(monkeypatch, capsys):
    fake = MagicMock()
    fake.process_batch.return_value = _result(
        successful=["a.pdf", "b.pdf"],
        dry_run=True,
    )
    monkeypatch.setattr(
        "raganything.batch_parser.BatchParser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["batch_parser", "docs/", "--output", "/tmp/out", "--dry-run"],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "Dry run: files that would be processed:" in out
    assert "a.pdf" in out
    assert fake.process_batch.call_args.kwargs["dry_run"] is True


def test_cli_dry_run_with_no_supported_files_returns_zero(monkeypatch, capsys):
    fake = MagicMock()
    fake.process_batch.return_value = _result(successful=[], dry_run=True)
    monkeypatch.setattr(
        "raganything.batch_parser.BatchParser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["batch_parser", "empty/", "--output", "/tmp/out", "--dry-run"],
    )

    assert main() == 0
    assert "Dry run: no supported files found." in capsys.readouterr().out


def test_cli_returns_one_when_batch_parser_construction_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        "raganything.batch_parser.BatchParser",
        MagicMock(side_effect=ValueError("Unsupported parser type: nope")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["batch_parser", "a.pdf", "--output", "/tmp/out", "--parser", "nope"],
    )

    assert main() == 1
    assert "Unsupported parser type: nope" in capsys.readouterr().out


def test_cli_forwards_worker_progress_and_timeout_flags(monkeypatch):
    fake = MagicMock()
    fake.process_batch.return_value = _result(successful=["a.pdf"])
    ctor = MagicMock(return_value=fake)
    monkeypatch.setattr("raganything.batch_parser.BatchParser", ctor)
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_parser",
            "a.pdf",
            "--output",
            "/tmp/out",
            "--workers",
            "2",
            "--no-progress",
            "--timeout",
            "12",
            "--parser",
            "docling",
            "--method",
            "ocr",
        ],
    )

    assert main() == 0
    assert ctor.call_args.kwargs == {
        "parser_type": "docling",
        "max_workers": 2,
        "show_progress": False,
        "timeout_per_file": 12,
    }
    assert fake.process_batch.call_args.kwargs["parse_method"] == "ocr"


def test_cli_requires_output_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["batch_parser", "a.pdf"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
