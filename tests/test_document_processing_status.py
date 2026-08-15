"""Document completion flags used to skip or report ingest.

Wrong fully-processed checks cause skipped multimodal stages or repeated
LLM extraction. Status helpers must fail closed on missing docs and errors.
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
    def __init__(self, records=None, *, raise_on_get=False):
        self.records = records or {}
        self.raise_on_get = raise_on_get

    async def get_by_id(self, key):
        if self.raise_on_get:
            raise RuntimeError("storage down")
        return self.records.get(key)


def _processor(storage):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.lightrag = type("LR", (), {"doc_status": storage})()
    return dummy


@pytest.mark.asyncio
async def test_is_document_fully_processed_requires_text_and_multimodal():
    dummy = _processor(
        FakeDocStatusStorage(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSED,
                    "multimodal_processed": True,
                }
            }
        )
    )
    assert await dummy.is_document_fully_processed("doc-1") is True


@pytest.mark.asyncio
async def test_is_document_fully_processed_false_when_multimodal_pending():
    dummy = _processor(
        FakeDocStatusStorage(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSED,
                    "multimodal_processed": False,
                }
            }
        )
    )
    assert await dummy.is_document_fully_processed("doc-1") is False


@pytest.mark.asyncio
async def test_is_document_fully_processed_false_when_text_not_processed():
    dummy = _processor(
        FakeDocStatusStorage(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSING,
                    "multimodal_processed": True,
                }
            }
        )
    )
    assert await dummy.is_document_fully_processed("doc-1") is False


@pytest.mark.asyncio
async def test_is_document_fully_processed_false_when_missing_or_storage_error():
    dummy = _processor(FakeDocStatusStorage())
    assert await dummy.is_document_fully_processed("missing") is False

    broken = _processor(FakeDocStatusStorage(raise_on_get=True))
    assert await broken.is_document_fully_processed("doc-1") is False


@pytest.mark.asyncio
async def test_get_document_processing_status_for_existing_doc():
    dummy = _processor(
        FakeDocStatusStorage(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSED,
                    "multimodal_processed": True,
                    "chunks_count": 3,
                    "chunks_list": ["c1", "c2", "c3"],
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            }
        )
    )
    status = await dummy.get_document_processing_status("doc-1")
    assert status["exists"] is True
    assert status["text_processed"] is True
    assert status["multimodal_processed"] is True
    assert status["fully_processed"] is True
    assert status["chunks_count"] == 3
    assert status["chunks_list"] == ["c1", "c2", "c3"]
    assert status["status"] == DocStatus.PROCESSED


@pytest.mark.asyncio
async def test_get_document_processing_status_missing_and_error_fail_closed():
    dummy = _processor(FakeDocStatusStorage())
    missing = await dummy.get_document_processing_status("nope")
    assert missing == {
        "exists": False,
        "text_processed": False,
        "multimodal_processed": False,
        "fully_processed": False,
        "chunks_count": 0,
    }

    broken = _processor(FakeDocStatusStorage(raise_on_get=True))
    errored = await broken.get_document_processing_status("doc-1")
    assert errored["exists"] is False
    assert errored["fully_processed"] is False
    assert "storage down" in errored["error"]
    assert errored["chunks_count"] == 0
