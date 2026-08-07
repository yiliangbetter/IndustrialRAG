#!/usr/bin/env python3
"""Regression: process_batch_async must forward parser kwargs without crashing."""

from unittest.mock import MagicMock, patch

import pytest

from raganything.batch_parser import BatchParser, BatchProcessingResult


@pytest.mark.asyncio
async def test_process_batch_async_forwards_parser_kwargs():
    batch_parser = BatchParser(
        parser_type="mineru",
        max_workers=1,
        show_progress=False,
        skip_installation_check=True,
    )
    expected = BatchProcessingResult(
        successful_files=["doc.pdf"],
        failed_files=[],
        total_files=1,
        processing_time=0.1,
        errors={},
        output_dir="./out",
    )

    with patch.object(batch_parser, "process_batch", return_value=expected) as mock_sync:
        result = await batch_parser.process_batch_async(
            file_paths=["doc.pdf"],
            output_dir="./out",
            parse_method="auto",
            recursive=False,
            dry_run=False,
            backend="pipeline",
            lang="en",
        )

    assert result is expected
    mock_sync.assert_called_once_with(
        ["doc.pdf"],
        "./out",
        "auto",
        False,
        False,
        backend="pipeline",
        lang="en",
    )


@pytest.mark.asyncio
async def test_process_documents_batch_async_forwards_kwargs():
    from raganything.batch import BatchMixin

    class _Harness(BatchMixin):
        def __init__(self):
            self.config = MagicMock()
            self.config.parser = "mineru"
            self.config.parser_output_dir = "./out"
            self.config.parse_method = "auto"
            self.config.max_concurrent_files = 2
            self.config.recursive_folder_processing = True

    expected = BatchProcessingResult(
        successful_files=["a.pdf"],
        failed_files=[],
        total_files=1,
        processing_time=0.01,
        errors={},
        output_dir="./out",
    )
    harness = _Harness()
    called = {}

    with patch("raganything.batch.BatchParser") as mock_cls:
        mock_instance = mock_cls.return_value

        async def _fake_async(**kwargs):
            called.update(kwargs)
            return expected

        mock_instance.process_batch_async = _fake_async

        result = await harness.process_documents_batch_async(
            file_paths=["a.pdf"],
            backend="vlm-transformers",
            formula=True,
        )

    assert result is expected
    assert called["backend"] == "vlm-transformers"
    assert called["formula"] is True
    assert called["file_paths"] == ["a.pdf"]
