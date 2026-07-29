"""Regression tests for document text/multimodal processing status helpers."""

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
    def __init__(self, records=None, error=None):
        self.records = records or {}
        self.error = error

    async def get_by_id(self, key):
        if self.error is not None:
            raise self.error
        return self.records.get(key)


def _make_processor(records=None, error=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage(records=records, error=error)},
    )()
    return processor


@pytest.mark.asyncio
async def test_is_document_fully_processed_requires_text_and_multimodal():
    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": True,
            }
        }
    )
    assert await processor.is_document_fully_processed("doc-1") is True

    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": False,
            }
        }
    )
    assert await processor.is_document_fully_processed("doc-1") is False

    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSING,
                "multimodal_processed": True,
            }
        }
    )
    assert await processor.is_document_fully_processed("doc-1") is False


@pytest.mark.asyncio
async def test_is_document_fully_processed_missing_or_error_is_false():
    processor = _make_processor({})
    assert await processor.is_document_fully_processed("missing") is False

    processor = _make_processor(error=RuntimeError("storage down"))
    assert await processor.is_document_fully_processed("doc-1") is False


@pytest.mark.asyncio
async def test_get_document_processing_status_details_when_present():
    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": True,
                "chunks_count": 3,
                "chunks_list": ["c1", "c2", "c3"],
                "updated_at": "2026-01-01T00:00:00",
            }
        }
    )

    status = await processor.get_document_processing_status("doc-1")

    assert status["exists"] is True
    assert status["text_processed"] is True
    assert status["multimodal_processed"] is True
    assert status["fully_processed"] is True
    assert status["chunks_count"] == 3
    assert status["chunks_list"] == ["c1", "c2", "c3"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["updated_at"] == "2026-01-01T00:00:00"
    assert status["raw_status"]["chunks_count"] == 3


@pytest.mark.asyncio
async def test_get_document_processing_status_partial_and_missing():
    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                # multimodal_processed omitted -> treated as False
                "chunks_count": 1,
            }
        }
    )
    status = await processor.get_document_processing_status("doc-1")
    assert status["exists"] is True
    assert status["text_processed"] is True
    assert status["multimodal_processed"] is False
    assert status["fully_processed"] is False
    assert status["chunks_count"] == 1

    processor = _make_processor({})
    missing = await processor.get_document_processing_status("nope")
    assert missing == {
        "exists": False,
        "text_processed": False,
        "multimodal_processed": False,
        "fully_processed": False,
        "chunks_count": 0,
    }


@pytest.mark.asyncio
async def test_get_document_processing_status_storage_error():
    processor = _make_processor(error=RuntimeError("boom"))
    status = await processor.get_document_processing_status("doc-1")

    assert status["exists"] is False
    assert status["error"] == "boom"
    assert status["text_processed"] is False
    assert status["multimodal_processed"] is False
    assert status["fully_processed"] is False
    assert status["chunks_count"] == 0
