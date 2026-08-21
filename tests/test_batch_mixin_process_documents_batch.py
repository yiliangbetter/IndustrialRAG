"""Tests for BatchMixin.process_documents_batch(_async) config wiring.

Distinct from process_documents_with_rag_batch and file-filter wrappers:
these paths construct BatchParser and forward config defaults into the
parse-only batch APIs.
"""

import pytest

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchProcessingResult
from raganything.config import RAGAnythingConfig


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_result(output_dir: str) -> BatchProcessingResult:
    return BatchProcessingResult(
        successful_files=["a.pdf"],
        failed_files=[],
        total_files=1,
        processing_time=0.01,
        errors={},
        output_dir=output_dir,
    )


def test_process_documents_batch_uses_config_defaults(tmp_path, monkeypatch):
    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured["parser_init"] = kwargs

        def process_batch(self, **kwargs):
            captured["process_batch"] = kwargs
            return _make_result(kwargs["output_dir"])

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(
        parser="docling",
        parser_output_dir=str(tmp_path / "default-out"),
        parse_method="ocr",
        max_concurrent_files=3,
        recursive_folder_processing=False,
    )

    result = batch.process_documents_batch(
        ["doc.pdf"],
        show_progress=False,
        lang="en",
    )

    assert captured["parser_init"] == {
        "parser_type": "docling",
        "max_workers": 3,
        "show_progress": False,
        "skip_installation_check": True,
    }
    assert captured["process_batch"]["file_paths"] == ["doc.pdf"]
    assert captured["process_batch"]["output_dir"] == str(tmp_path / "default-out")
    assert captured["process_batch"]["parse_method"] == "ocr"
    assert captured["process_batch"]["recursive"] is False
    assert captured["process_batch"]["lang"] == "en"
    assert result.successful_files == ["a.pdf"]


def test_process_documents_batch_explicit_args_override_config(tmp_path, monkeypatch):
    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured["parser_init"] = kwargs

        def process_batch(self, **kwargs):
            captured["process_batch"] = kwargs
            return _make_result(kwargs["output_dir"])

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(
        parser="mineru",
        parser_output_dir=str(tmp_path / "default-out"),
        parse_method="auto",
        max_concurrent_files=1,
        recursive_folder_processing=True,
    )

    out = str(tmp_path / "override-out")
    batch.process_documents_batch(
        ["a.pdf", "b.pdf"],
        output_dir=out,
        parse_method="txt",
        max_workers=5,
        recursive=False,
        show_progress=True,
    )

    assert captured["parser_init"]["parser_type"] == "mineru"
    assert captured["parser_init"]["max_workers"] == 5
    assert captured["parser_init"]["show_progress"] is True
    assert captured["process_batch"]["output_dir"] == out
    assert captured["process_batch"]["parse_method"] == "txt"
    assert captured["process_batch"]["recursive"] is False


@pytest.mark.asyncio
async def test_process_documents_batch_async_uses_config_defaults(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured["parser_init"] = kwargs

        async def process_batch_async(self, **kwargs):
            captured["process_batch_async"] = kwargs
            return _make_result(kwargs["output_dir"])

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(
        parser="paddleocr",
        parser_output_dir=str(tmp_path / "async-out"),
        parse_method="auto",
        max_concurrent_files=4,
        recursive_folder_processing=True,
    )

    result = await batch.process_documents_batch_async(
        ["scan.pdf"],
        show_progress=False,
    )

    assert captured["parser_init"] == {
        "parser_type": "paddleocr",
        "max_workers": 4,
        "show_progress": False,
        "skip_installation_check": True,
    }
    assert captured["process_batch_async"]["file_paths"] == ["scan.pdf"]
    assert captured["process_batch_async"]["output_dir"] == str(tmp_path / "async-out")
    assert captured["process_batch_async"]["parse_method"] == "auto"
    assert captured["process_batch_async"]["recursive"] is True
    assert result.total_files == 1
