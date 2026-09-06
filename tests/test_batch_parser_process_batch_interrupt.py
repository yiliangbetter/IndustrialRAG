"""BatchParser must account for futures still running when as_completed dies.

#125 covers dry-run, mixed completed results, and async success. The leftover
ops contract from closed #115 is the collective interrupt path: if the
completion loop raises (timeout, executor failure), pending work is marked
failed with "Processing interrupted" while already-collected successes stay
successful. Silent drops here hide partial folder jobs.
"""

from concurrent.futures import Future

import raganything.batch_parser as batch_parser_module
from raganything.batch_parser import BatchParser


class _ControlledExecutor:
    """First submit completes immediately; later submits stay pending."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self._submitted = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, file_path, *args, **kwargs):
        future = Future()
        self._submitted += 1
        if self._submitted == 1:
            future.set_result((True, file_path, None))
        return future


def _as_completed_raise_after_first(futures, timeout=None):
    yielded = False
    for future in futures:
        yield future
        yielded = True
        break
    if yielded:
        raise TimeoutError("simulated batch timeout")
    raise TimeoutError("simulated batch timeout")


def test_process_batch_marks_pending_files_interrupted(tmp_path, monkeypatch):
    good = tmp_path / "good.pdf"
    pending = tmp_path / "pending.pdf"
    good.write_bytes(b"%PDF-1.4\n")
    pending.write_bytes(b"%PDF-1.4\n")

    bp = BatchParser(
        parser_type="mineru",
        skip_installation_check=True,
        show_progress=False,
        max_workers=2,
    )

    monkeypatch.setattr(batch_parser_module, "ThreadPoolExecutor", _ControlledExecutor)
    monkeypatch.setattr(
        batch_parser_module, "as_completed", _as_completed_raise_after_first
    )

    result = bp.process_batch(
        file_paths=[str(good), str(pending)],
        output_dir=str(tmp_path / "out"),
        parse_method="auto",
    )

    assert result.total_files == 2
    assert result.dry_run is False
    assert len(result.successful_files) == 1
    assert len(result.failed_files) == 1

    failed_path = result.failed_files[0]
    assert failed_path != result.successful_files[0]
    assert failed_path in {str(good), str(pending)}
    assert "Processing interrupted" in result.errors[failed_path]
    assert "simulated batch timeout" in result.errors[failed_path]
