"""Regression tests for BatchParser timeout behavior."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from raganything.batch_parser import BatchParser


@pytest.fixture
def pdf_files(tmp_path):
    paths = []
    for name in ("fast.pdf", "slow.pdf"):
        path = tmp_path / name
        path.write_bytes(b"%PDF-1.4")
        paths.append(str(path))
    return paths


def test_process_batch_timeout_returns_without_waiting_for_hung_worker(
    tmp_path, pdf_files
):
    """timeout_per_file must return promptly; do not wait on hung workers."""
    bp = BatchParser(
        parser_type="mineru",
        max_workers=2,
        show_progress=False,
        timeout_per_file=1,
        skip_installation_check=True,
    )

    def fake_process(file_path, output_dir, parse_method="auto", **kwargs):
        if Path(file_path).name == "slow.pdf":
            time.sleep(5)
            return True, file_path, None
        time.sleep(0.05)
        return True, file_path, None

    out_dir = str(tmp_path / "out")
    start = time.time()
    with patch.object(bp, "process_single_file", side_effect=fake_process):
        result = bp.process_batch(pdf_files, out_dir, parse_method="auto")
    elapsed = time.time() - start

    assert elapsed < 2.5, f"timed out path hung for {elapsed:.2f}s (expected ~1s)"
    assert "fast.pdf" in "".join(result.successful_files)
    assert any(Path(p).name == "slow.pdf" for p in result.failed_files)
    assert any("Timed out" in msg for msg in result.errors.values())


def test_process_batch_timeout_is_per_file_not_batch_total(tmp_path):
    """Sequential files each under timeout must all succeed despite total > timeout."""
    paths = []
    for i in range(3):
        path = tmp_path / f"doc{i}.pdf"
        path.write_bytes(b"%PDF-1.4")
        paths.append(str(path))

    bp = BatchParser(
        parser_type="mineru",
        max_workers=1,
        show_progress=False,
        timeout_per_file=1,
        skip_installation_check=True,
    )

    def fake_process(file_path, output_dir, parse_method="auto", **kwargs):
        time.sleep(0.4)
        return True, file_path, None

    out_dir = str(tmp_path / "out")
    with patch.object(bp, "process_single_file", side_effect=fake_process):
        result = bp.process_batch(paths, out_dir, parse_method="auto")

    # Total work ~1.2s > timeout_per_file=1, but each file is under the limit.
    assert result.failed_files == []
    assert len(result.successful_files) == 3
