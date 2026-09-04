"""Folder ingest must honor config.max_concurrent_files when max_workers is omitted.

Batch jobs that only pass a folder path would otherwise default to unlimited
concurrency, overloading MinerU/LibreOffice on large plant-manual trees.
"""

from unittest.mock import patch

import pytest

from raganything.batch import BatchMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_batch(tmp_path, max_concurrent_files=3):
    class DummyBatch(BatchMixin):
        pass

    dummy = DummyBatch()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "auto",
            "supported_file_extensions": [".pdf"],
            "recursive_folder_processing": False,
            "max_concurrent_files": max_concurrent_files,
            "parser": "mineru",
        },
    )()
    dummy.process_calls = []

    async def _ensure():
        return {"success": True}

    async def _process(file_path, **kwargs):
        dummy.process_calls.append(file_path)

    dummy._ensure_lightrag_initialized = _ensure
    dummy.process_document_complete = _process
    return dummy


@pytest.mark.asyncio
async def test_omitted_max_workers_uses_config_max_concurrent_files(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "b.pdf").write_bytes(b"%PDF-1.4\n")

    dummy = _make_batch(tmp_path, max_concurrent_files=3)
    real_semaphore = __import__("asyncio").Semaphore

    with patch("raganything.batch.asyncio.Semaphore", wraps=real_semaphore) as sem:
        await dummy.process_folder_complete(str(folder))

    sem.assert_called_once_with(3)
    assert len(dummy.process_calls) == 2


@pytest.mark.asyncio
async def test_explicit_max_workers_overrides_config(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF-1.4\n")

    dummy = _make_batch(tmp_path, max_concurrent_files=8)
    real_semaphore = __import__("asyncio").Semaphore

    with patch("raganything.batch.asyncio.Semaphore", wraps=real_semaphore) as sem:
        await dummy.process_folder_complete(str(folder), max_workers=1)

    sem.assert_called_once_with(1)
    assert len(dummy.process_calls) == 1
