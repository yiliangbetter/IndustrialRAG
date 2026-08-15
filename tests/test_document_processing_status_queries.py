"""Regression tests for document completion status queries.

is_document_fully_processed and get_document_processing_status drive
re-ingest decisions. A false "fully processed" skip drops multimodal
content; a false negative re-runs expensive pipelines.
"""

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeDocStatusStorage:
    def __init__(self, records=None, get_error=None):
        self.records = records or {}
        self.get_error = get_error
        self.index_done_calls = 0

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor(storage):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type("FakeLightRAG", (), {"doc_status": storage})()
    return processor


@pytest.mark.asyncio
async def test_is_document_fully_processed_requires_text_and_multimodal():
    storage = FakeDocStatusStorage(
        {
            "doc-full": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": True,
            },
            "doc-text-only": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": False,
            },
            "doc-pending": {
                "status": DocStatus.PROCESSING,
                "multimodal_processed": True,
            },
        }
    )
    processor = _make_processor(storage)

    assert await processor.is_document_fully_processed("doc-full") is True
    assert await processor.is_document_fully_processed("doc-text-only") is False
    assert await processor.is_document_fully_processed("doc-pending") is False
    assert await processor.is_document_fully_processed("doc-missing") is False


@pytest.mark.asyncio
async def test_is_document_fully_processed_returns_false_on_storage_error():
    processor = _make_processor(
        FakeDocStatusStorage(get_error=RuntimeError("status backend down"))
    )

    assert await processor.is_document_fully_processed("doc-x") is False


@pytest.mark.asyncio
async def test_get_document_processing_status_for_existing_partial_doc():
    storage = FakeDocStatusStorage(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": False,
                "chunks_count": 3,
                "chunks_list": ["c1", "c2", "c3"],
                "updated_at": "2026-08-15T00:00:00+00:00",
            }
        }
    )
    processor = _make_processor(storage)

    status = await processor.get_document_processing_status("doc-1")

    assert status["exists"] is True
    assert status["text_processed"] is True
    assert status["multimodal_processed"] is False
    assert status["fully_processed"] is False
    assert status["chunks_count"] == 3
    assert status["chunks_list"] == ["c1", "c2", "c3"]
    assert status["status"] == DocStatus.PROCESSED


@pytest.mark.asyncio
async def test_get_document_processing_status_for_missing_doc():
    processor = _make_processor(FakeDocStatusStorage())

    status = await processor.get_document_processing_status("doc-missing")

    assert status == {
        "exists": False,
        "text_processed": False,
        "multimodal_processed": False,
        "fully_processed": False,
        "chunks_count": 0,
    }


@pytest.mark.asyncio
async def test_get_document_processing_status_includes_error_on_failure():
    processor = _make_processor(
        FakeDocStatusStorage(get_error=RuntimeError("status backend down"))
    )

    status = await processor.get_document_processing_status("doc-x")

    assert status["exists"] is False
    assert status["fully_processed"] is False
    assert "status backend down" in status["error"]


@pytest.mark.asyncio
async def test_mark_multimodal_processing_complete_updates_existing_doc():
    storage = FakeDocStatusStorage(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": False,
                "file_path": "manual.pdf",
            }
        }
    )
    processor = _make_processor(storage)

    await processor._mark_multimodal_processing_complete("doc-1")

    updated = storage.records["doc-1"]
    assert updated["multimodal_processed"] is True
    assert updated["status"] == DocStatus.PROCESSED
    assert updated["file_path"] == "manual.pdf"
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_mark_multimodal_processing_complete_noops_for_missing_doc():
    storage = FakeDocStatusStorage()
    processor = _make_processor(storage)

    await processor._mark_multimodal_processing_complete("doc-missing")

    assert storage.records == {}
    assert storage.index_done_calls == 0
