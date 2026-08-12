"""BatchParser.process_batch mid-batch interrupt marking.

When the as_completed loop raises, pending futures must be recorded as failed
with a clear interrupt message. Without this, incomplete batches can look like
clean success.

Distinct from #93 (timeout hang) and #110 (CLI / process_single_file).

These tests mock ThreadPoolExecutor so shutdown does not race with the
interrupt handler; they exercise the marking logic itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import raganything.batch_parser as batch_parser_module
from raganything.batch_parser import BatchParser


class _PendingFuture:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._done = False

    def done(self) -> bool:
        return self._done

    def result(self):
        raise AssertionError("pending future should not be awaited")


class _CompletedFuture:
    def __init__(self, file_path: str, success: bool = True, error: str | None = None):
        self.file_path = file_path
        self._done = True
        self._success = success
        self._error = error

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._success, self.file_path, self._error


class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, file_path, *args, **kwargs):
        # process_batch submits process_single_file(file_path, ...)
        future = _PendingFuture(file_path)
        self.submitted.append(future)
        return future


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


def test_process_batch_marks_pending_files_on_interrupt(
    batch_parser, tmp_path, monkeypatch
):
    """Pending workers must appear in failed_files with Processing interrupted."""
    files = []
    for name in ("a.pdf", "b.pdf"):
        path = tmp_path / name
        path.write_bytes(b"%PDF")
        files.append(str(path))
    out = tmp_path / "out"

    monkeypatch.setattr(batch_parser_module, "ThreadPoolExecutor", _FakeExecutor)

    def raise_interrupt(futures, timeout=None):
        raise RuntimeError("executor exploded")

    monkeypatch.setattr(batch_parser_module, "as_completed", raise_interrupt)

    result = batch_parser.process_batch(files, str(out), parse_method="auto")

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

    completed = _CompletedFuture(done_path, success=True)
    pending = _PendingFuture(pending_path)
    submit_map = {done_path: completed, pending_path: pending}

    class SelectiveExecutor(_FakeExecutor):
        def submit(self, fn, file_path, *args, **kwargs):
            return submit_map[file_path]

    monkeypatch.setattr(batch_parser_module, "ThreadPoolExecutor", SelectiveExecutor)

    def as_completed_then_interrupt(futures, timeout=None):
        yield completed
        raise RuntimeError("interrupted after first")

    monkeypatch.setattr(
        batch_parser_module, "as_completed", as_completed_then_interrupt
    )

    result = batch_parser.process_batch(files, str(out))

    assert done_path in result.successful_files
    assert pending_path in result.failed_files
    assert result.errors[pending_path].startswith("Processing interrupted:")
    assert "interrupted after first" in result.errors[pending_path]
    assert done_path not in result.failed_files
