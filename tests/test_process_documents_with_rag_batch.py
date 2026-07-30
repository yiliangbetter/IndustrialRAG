"""Regression tests for BatchMixin.process_documents_with_rag_batch.

Parse→RAG orchestration must isolate per-file RAG failures, fail closed on
LightRAG init errors, honor config defaults when wiring BatchParser, and emit
batch callbacks. Regressions here silently drop batch ingest outcomes or mark
partial batches as fully successful.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchProcessingResult


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


def _make_batch(tmp_path):
    class DummyBatch(BatchMixin):
        pass

    batch = DummyBatch()
    batch.logger = FakeLogger()
    batch.config = SimpleNamespace(
        parser="mineru",
        parser_output_dir=str(tmp_path / "out"),
        parse_method="auto",
        max_concurrent_files=2,
        recursive_folder_processing=True,
    )
    batch.callback_manager = None
    return batch


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_init_failure_raises(tmp_path):
    batch = _make_batch(tmp_path)

    batch.process_documents_batch = MagicMock(
        return_value=BatchProcessingResult(
            successful_files=[str(tmp_path / "a.pdf")],
            failed_files=[],
            total_files=1,
            processing_time=0.1,
            errors={},
            output_dir=str(tmp_path / "out"),
        )
    )

    async def fail_init():
        return {"success": False, "error": "missing llm"}

    batch._ensure_lightrag_initialized = fail_init

    with pytest.raises(RuntimeError, match="LightRAG initialization failed"):
        await batch.process_documents_with_rag_batch([str(tmp_path / "a.pdf")])


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_isolates_rag_failures(tmp_path):
    batch = _make_batch(tmp_path)
    ok = str(tmp_path / "ok.pdf")
    bad = str(tmp_path / "bad.pdf")

    batch.process_documents_batch = MagicMock(
        return_value=BatchProcessingResult(
            successful_files=[ok, bad],
            failed_files=[],
            total_files=2,
            processing_time=0.2,
            errors={},
            output_dir=str(tmp_path / "out"),
        )
    )

    async def ok_init():
        return {"success": True}

    processed = []

    async def fake_process(file_path, **kwargs):
        if file_path == bad:
            raise RuntimeError("rag boom")
        processed.append(file_path)

    batch._ensure_lightrag_initialized = ok_init
    batch.process_document_complete = fake_process

    result = await batch.process_documents_with_rag_batch(
        [ok, bad],
        output_dir=str(tmp_path / "out"),
        parse_method="ocr",
    )

    assert processed == [ok]
    assert result["successful_rag_files"] == 1
    assert result["failed_rag_files"] == 1
    assert result["rag_results"][ok]["status"] == "success"
    assert result["rag_results"][bad]["status"] == "failed"
    assert "rag boom" in result["rag_results"][bad]["error"]
    assert result["parse_result"].successful_files == [ok, bad]
    assert any("rag boom" in e for e in batch.logger.errors)


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_skips_rag_when_parse_empty(tmp_path):
    batch = _make_batch(tmp_path)

    batch.process_documents_batch = MagicMock(
        return_value=BatchProcessingResult(
            successful_files=[],
            failed_files=[str(tmp_path / "x.pdf")],
            total_files=1,
            processing_time=0.05,
            errors={str(tmp_path / "x.pdf"): "parse failed"},
            output_dir=str(tmp_path / "out"),
        )
    )

    async def ok_init():
        return {"success": True}

    called = []

    async def fake_process(file_path, **kwargs):
        called.append(file_path)

    batch._ensure_lightrag_initialized = ok_init
    batch.process_document_complete = fake_process

    result = await batch.process_documents_with_rag_batch([str(tmp_path / "x.pdf")])

    assert called == []
    assert result["rag_results"] == {}
    assert result["successful_rag_files"] == 0
    assert result["failed_rag_files"] == 0


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_dispatches_callbacks(tmp_path):
    batch = _make_batch(tmp_path)
    ok = str(tmp_path / "ok.pdf")

    events = []

    class FakeCallbacks:
        def dispatch(self, event, **kwargs):
            events.append((event, kwargs))

    batch.callback_manager = FakeCallbacks()
    batch.process_documents_batch = MagicMock(
        return_value=BatchProcessingResult(
            successful_files=[ok],
            failed_files=[],
            total_files=1,
            processing_time=0.1,
            errors={},
            output_dir=str(tmp_path / "out"),
        )
    )

    async def ok_init():
        return {"success": True}

    async def fake_process(file_path, **kwargs):
        return None

    batch._ensure_lightrag_initialized = ok_init
    batch.process_document_complete = fake_process

    result = await batch.process_documents_with_rag_batch([ok, str(tmp_path / "other.pdf")])

    assert events[0][0] == "on_batch_start"
    assert events[0][1]["file_count"] == 2
    assert events[-1][0] == "on_batch_complete"
    assert events[-1][1]["successful"] == 1
    assert events[-1][1]["failed"] == 0
    assert events[-1][1]["total_files"] == 2
    assert result["successful_rag_files"] == 1


def test_process_documents_batch_uses_config_defaults(tmp_path, monkeypatch):
    batch = _make_batch(tmp_path)
    created = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            created["init"] = kwargs

        def process_batch(self, **kwargs):
            created["process"] = kwargs
            return BatchProcessingResult(
                successful_files=[],
                failed_files=[],
                total_files=0,
                processing_time=0.0,
                errors={},
                output_dir=str(tmp_path / "out"),
            )

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    result = batch.process_documents_batch([str(tmp_path / "a.pdf")])

    assert created["init"]["parser_type"] == "mineru"
    assert created["init"]["max_workers"] == 2
    assert created["init"]["skip_installation_check"] is True
    assert created["process"]["output_dir"] == str(tmp_path / "out")
    assert created["process"]["parse_method"] == "auto"
    assert created["process"]["recursive"] is True
    assert isinstance(result, BatchProcessingResult)


@pytest.mark.asyncio
async def test_process_documents_batch_async_forwards_to_parser(tmp_path, monkeypatch):
    batch = _make_batch(tmp_path)
    created = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            created["init"] = kwargs

        async def process_batch_async(self, **kwargs):
            created["process"] = kwargs
            return BatchProcessingResult(
                successful_files=["ok"],
                failed_files=[],
                total_files=1,
                processing_time=0.01,
                errors={},
                output_dir=str(tmp_path / "custom"),
            )

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    result = await batch.process_documents_batch_async(
        ["a.pdf"],
        output_dir=str(tmp_path / "custom"),
        parse_method="ocr",
        max_workers=4,
        recursive=False,
    )

    assert created["init"]["max_workers"] == 4
    assert created["process"]["output_dir"] == str(tmp_path / "custom")
    assert created["process"]["parse_method"] == "ocr"
    assert created["process"]["recursive"] is False
    assert result.successful_files == ["ok"]
