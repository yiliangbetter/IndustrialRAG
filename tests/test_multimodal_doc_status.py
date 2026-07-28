"""Regression: multimodal-only ingest must create/update doc_status."""

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
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


@pytest.mark.asyncio
async def test_mark_multimodal_complete_creates_missing_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG", (), {"doc_status": FakeDocStatusStorage()}
    )()

    await processor._mark_multimodal_processing_complete("doc-image-only")

    record = processor.lightrag.doc_status.records["doc-image-only"]
    assert record["multimodal_processed"] is True
    assert record["status"] == DocStatus.PROCESSED
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_update_chunks_creates_missing_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG", (), {"doc_status": FakeDocStatusStorage()}
    )()

    await processor._update_doc_status_with_chunks_type_aware(
        "doc-image-only", ["chunk-a", "chunk-b"]
    )

    record = processor.lightrag.doc_status.records["doc-image-only"]
    assert record["chunks_list"] == ["chunk-a", "chunk-b"]
    assert record["chunks_count"] == 2
    assert record["status"] == DocStatus.PROCESSED
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_mark_multimodal_complete_preserves_existing_fields():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    storage = FakeDocStatusStorage()
    storage.records["doc-1"] = {
        "status": DocStatus.PROCESSED,
        "chunks_list": ["c1"],
        "chunks_count": 1,
        "file_path": "a.pdf",
        "multimodal_processed": False,
    }
    processor.lightrag = type("FakeLightRAG", (), {"doc_status": storage})()

    await processor._mark_multimodal_processing_complete("doc-1")

    record = storage.records["doc-1"]
    assert record["multimodal_processed"] is True
    assert record["chunks_list"] == ["c1"]
    assert record["file_path"] == "a.pdf"
