"""BatchMixin async batch parsing must honor config defaults and forward kwargs.

process_documents_batch_async is the non-blocking counterpart of the sync batch
API. If defaults (output dir, parse method, concurrency, recursion) are dropped,
folder ingest silently uses the wrong parser settings or overwrites shared output.
"""

import pytest

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchParser, BatchProcessingResult


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _dummy_batch(tmp_path):
    dummy = BatchMixin()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "paddleocr",
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "ocr",
            "max_concurrent_files": 3,
            "recursive_folder_processing": False,
        },
    )()
    return dummy


def _result(output_dir: str) -> BatchProcessingResult:
    return BatchProcessingResult(
        successful_files=["a.pdf"],
        failed_files=[],
        total_files=1,
        processing_time=0.01,
        errors={},
        output_dir=output_dir,
    )


class TestProcessDocumentsBatchAsync:
    @pytest.mark.asyncio
    async def test_uses_config_defaults_when_kwargs_omitted(
        self, monkeypatch, tmp_path
    ):
        dummy = _dummy_batch(tmp_path)
        constructed = {}
        forwarded = {}

        class FakeBatchParser:
            def __init__(self, **kwargs):
                constructed.update(kwargs)

            async def process_batch_async(self, **kwargs):
                forwarded.update(kwargs)
                return _result(kwargs["output_dir"])

        monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

        result = await dummy.process_documents_batch_async(
            file_paths=["docs"],
            show_progress=False,
            lang="ch",
        )

        assert constructed == {
            "parser_type": "paddleocr",
            "max_workers": 3,
            "show_progress": False,
            "skip_installation_check": True,
        }
        assert forwarded["file_paths"] == ["docs"]
        assert forwarded["output_dir"] == str(tmp_path / "parsed")
        assert forwarded["parse_method"] == "ocr"
        assert forwarded["recursive"] is False
        assert forwarded["lang"] == "ch"
        assert result.successful_files == ["a.pdf"]

    @pytest.mark.asyncio
    async def test_explicit_kwargs_override_config(self, monkeypatch, tmp_path):
        dummy = _dummy_batch(tmp_path)
        constructed = {}
        forwarded = {}

        class FakeBatchParser:
            def __init__(self, **kwargs):
                constructed.update(kwargs)

            async def process_batch_async(self, **kwargs):
                forwarded.update(kwargs)
                return _result(kwargs["output_dir"])

        monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

        await dummy.process_documents_batch_async(
            file_paths=["a.pdf"],
            output_dir=str(tmp_path / "custom"),
            parse_method="txt",
            max_workers=8,
            recursive=True,
            show_progress=True,
        )

        assert constructed["max_workers"] == 8
        assert forwarded["output_dir"] == str(tmp_path / "custom")
        assert forwarded["parse_method"] == "txt"
        assert forwarded["recursive"] is True


class TestBatchParserProcessBatchAsync:
    @pytest.mark.asyncio
    async def test_async_wrapper_delegates_to_process_batch(
        self, monkeypatch, tmp_path
    ):
        captured = {}

        def fake_process_batch(
            self,
            file_paths,
            output_dir,
            parse_method="auto",
            recursive=True,
            dry_run=False,
            **kwargs,
        ):
            captured["file_paths"] = file_paths
            captured["output_dir"] = output_dir
            captured["parse_method"] = parse_method
            captured["recursive"] = recursive
            captured["dry_run"] = dry_run
            captured["kwargs"] = kwargs
            return _result(output_dir)

        monkeypatch.setattr(BatchParser, "process_batch", fake_process_batch)

        parser = BatchParser(
            parser_type="mineru",
            show_progress=False,
            skip_installation_check=True,
        )
        result = await parser.process_batch_async(
            file_paths=["a.pdf"],
            output_dir=str(tmp_path),
            parse_method="ocr",
            recursive=False,
            dry_run=True,
        )

        assert captured["file_paths"] == ["a.pdf"]
        assert captured["parse_method"] == "ocr"
        assert captured["recursive"] is False
        assert captured["dry_run"] is True
        assert captured["kwargs"] == {}
        assert result.successful_files == ["a.pdf"]
        assert result.total_files == 1
