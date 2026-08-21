"""Tests for BatchMixin.process_documents_with_rag_batch.

This is the parse-then-ingest path used by folder/batch jobs. A regression
here can swallow RAG failures, skip remaining files, or report success
for documents that never entered LightRAG.
"""

import pytest

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchProcessingResult
from raganything.callbacks import CallbackManager, ProcessingCallback
from raganything.config import RAGAnythingConfig


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_batch_start(self, file_count=0, **kw):
        self.events.append(("batch_start", file_count))

    def on_batch_complete(self, total_files=0, successful=0, failed=0, **kw):
        self.events.append(("batch_complete", total_files, successful, failed))


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _batch(tmp_path) -> BatchMixin:
    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(
        parser="mineru",
        parser_output_dir=str(tmp_path / "out"),
        parse_method="ocr",
        max_concurrent_files=2,
        recursive_folder_processing=False,
    )
    return batch


def _parse_result(output_dir: str, successful, failed=None) -> BatchProcessingResult:
    failed = failed or []
    return BatchProcessingResult(
        successful_files=list(successful),
        failed_files=list(failed),
        total_files=len(successful) + len(failed),
        processing_time=0.01,
        errors={path: "parse error" for path in failed},
        output_dir=output_dir,
    )


@pytest.mark.asyncio
async def test_rag_batch_isolates_per_file_failures_and_skips_parse_failures(
    tmp_path,
):
    batch = _batch(tmp_path)
    captured = {}
    rag_calls = []

    def fake_process_documents_batch(**kwargs):
        captured["parse"] = kwargs
        return _parse_result(
            kwargs["output_dir"],
            successful=["good.pdf", "bad.pdf"],
            failed=["skip.pdf"],
        )

    async def fake_ensure():
        return {"success": True}

    async def fake_process_document_complete(file_path, **kwargs):
        rag_calls.append((file_path, kwargs))
        if file_path == "bad.pdf":
            raise RuntimeError("kg extract failed")

    batch.process_documents_batch = fake_process_documents_batch
    batch._ensure_lightrag_initialized = fake_ensure
    batch.process_document_complete = fake_process_document_complete

    result = await batch.process_documents_with_rag_batch(
        ["good.pdf", "bad.pdf", "skip.pdf"],
        lang="en",
    )

    assert captured["parse"]["file_paths"] == ["good.pdf", "bad.pdf", "skip.pdf"]
    assert captured["parse"]["parse_method"] == "ocr"
    assert captured["parse"]["lang"] == "en"
    assert [path for path, _ in rag_calls] == ["good.pdf", "bad.pdf"]
    assert rag_calls[0][1]["parse_method"] == "ocr"
    assert rag_calls[0][1]["lang"] == "en"
    assert "skip.pdf" not in result["rag_results"]
    assert result["rag_results"]["good.pdf"] == {
        "status": "success",
        "processed": True,
    }
    assert result["rag_results"]["bad.pdf"]["status"] == "failed"
    assert result["rag_results"]["bad.pdf"]["processed"] is False
    assert "kg extract failed" in result["rag_results"]["bad.pdf"]["error"]
    assert result["successful_rag_files"] == 1
    assert result["failed_rag_files"] == 1


@pytest.mark.asyncio
async def test_rag_batch_raises_when_lightrag_init_fails_after_parse(tmp_path):
    batch = _batch(tmp_path)
    rag_called = []

    def fake_process_documents_batch(**kwargs):
        return _parse_result(kwargs["output_dir"], successful=["a.pdf"])

    async def fake_ensure():
        return {"success": False, "error": "missing llm_model_func"}

    async def fake_process_document_complete(file_path, **kwargs):
        rag_called.append(file_path)

    batch.process_documents_batch = fake_process_documents_batch
    batch._ensure_lightrag_initialized = fake_ensure
    batch.process_document_complete = fake_process_document_complete

    with pytest.raises(RuntimeError, match="missing llm_model_func"):
        await batch.process_documents_with_rag_batch(["a.pdf"])

    assert rag_called == []


@pytest.mark.asyncio
async def test_rag_batch_dispatches_start_and_complete_callbacks(tmp_path):
    batch = _batch(tmp_path)
    recorder = RecordingCallback()
    manager = CallbackManager()
    manager.register(recorder)
    batch.callback_manager = manager

    def fake_process_documents_batch(**kwargs):
        return _parse_result(
            kwargs["output_dir"],
            successful=["ok.pdf", "fail.pdf"],
        )

    async def fake_ensure():
        return {"success": True}

    async def fake_process_document_complete(file_path, **kwargs):
        if file_path == "fail.pdf":
            raise RuntimeError("boom")

    batch.process_documents_batch = fake_process_documents_batch
    batch._ensure_lightrag_initialized = fake_ensure
    batch.process_document_complete = fake_process_document_complete

    result = await batch.process_documents_with_rag_batch(["ok.pdf", "fail.pdf"])

    assert recorder.events[0] == ("batch_start", 2)
    assert recorder.events[-1] == ("batch_complete", 2, 1, 1)
    assert result["successful_rag_files"] == 1
    assert result["failed_rag_files"] == 1
