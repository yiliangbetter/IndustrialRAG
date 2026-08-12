"""BatchParser.process_batch mid-batch interrupt marking.

When as_completed / the executor loop raises a non-timeout error, pending
futures must be recorded as failed with a clear interrupt message. Without
this, operators can treat incomplete batches as clean success.

Distinct from #93 (timeout hang) and #110 (CLI / process_single_file).
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import raganything.batch_parser as batch_parser_module
from raganything.batch_parser import BatchParser


@pytest.fixture
def batch_parser(monkeypatch):
    fake_parser = MagicMock()
    fake_parser.OFFICE_FORMATS = {".docx"}
    fake_parser.IMAGE_FORMATS = {".png"}
    fake_parser.TEXT_FORMATS = {".txt", ".md"}
    fake_parser.check_installation.return_value = True
    monkeypatch.setattr(
        "raganything.batch_parser.get_parser",
        MagicMock(return_value=fake_parser),
    )
    bp = BatchParser(
        parser_type="mineru",
        max_workers=2,
        show_progress=False,
        timeout_per_file=30,
        skip_installation_check=True,
    )
    bp.parser = fake_parser
    return bp


def test_process_batch_marks_pending_files_on_interrupt(batch_parser, tmp_path, monkeypatch):
    """Pending workers must appear in failed_files with Processing interrupted."""
    files = []
    for name in ("a.pdf", "b.pdf"):
        path = tmp_path / name
        path.write_bytes(b"%PDF")
        files.append(str(path))
    out = tmp_path / "out"

    started = threading.Barrier(3)  # 2 workers + main after submit
    release = threading.Event()

    def blocking_process(file_path, output_dir, parse_method="auto", **kwargs):
        started.wait(timeout=5)
        release.wait(timeout=5)
        return True, file_path, None

    batch_parser.process_single_file = blocking_process

    def raise_interrupt(futures, timeout=None):
        # Let workers enter process_single_file so futures stay not-done.
        started.wait(timeout=5)
        raise RuntimeError("executor exploded")

    monkeypatch.setattr(batch_parser_module, "as_completed", raise_interrupt)

    result = batch_parser.process_batch(files, str(out), parse_method="auto")

    release.set()  # unblock any remaining workers for clean shutdown

    assert result.total_files == 2
    assert result.successful_files == []
    assert sorted(result.failed_files) == sorted(files)
    assert len(result.errors) == 2
    for path in files:
        assert path in result.errors
        assert result.errors[path].startswith("Processing interrupted:")
        assert "executor exploded" in result.errors[path]


def test_process_batch_interrupt_preserves_already_completed(
    batch_parser, tmp_path, monkeypatch
):
    """Files completed before the interrupt must remain in successful_files."""
    files = []
    for name in ("done.pdf", "pending.pdf"):
        path = tmp_path / name
        path.write_bytes(b"%PDF")
        files.append(str(path))
    out = tmp_path / "out"
    done_path, pending_path = files

    release_pending = threading.Event()
    completed_first = threading.Event()

    def selective_process(file_path, output_dir, parse_method="auto", **kwargs):
        if Path(file_path).name == "done.pdf":
            completed_first.set()
            return True, file_path, None
        completed_first.wait(timeout=5)
        release_pending.wait(timeout=5)
        return True, file_path, None

    batch_parser.process_single_file = selective_process

    real_as_completed = batch_parser_module.as_completed

    def as_completed_then_interrupt(futures, timeout=None):
        # Yield the first completed future, then raise on the next iteration.
        gen = real_as_completed(futures, timeout=timeout)
        yield next(gen)
        raise RuntimeError("interrupted after first")

    monkeypatch.setattr(
        batch_parser_module, "as_completed", as_completed_then_interrupt
    )

    result = batch_parser.process_batch(files, str(out))

    release_pending.set()

    assert done_path in result.successful_files
    assert pending_path in result.failed_files
    assert result.errors[pending_path].startswith("Processing interrupted:")
    assert "interrupted after first" in result.errors[pending_path]
    assert done_path not in result.failed_files
