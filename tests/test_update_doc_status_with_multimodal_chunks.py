"""Regression tests for multimodal chunk IDs merging into doc_status.

`_update_doc_status_with_chunks_type_aware` is the last stage of type-aware
batch ingest. Dropping IDs hides figures from retrieval; clobbering text
chunks_list or raising on a missing doc leaves ingest looking complete while
the index is wrong.
"""

import pytest

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
    def __init__(self, records=None, *, get_error=None, upsert_error=None):
        self.records = dict(records or {})
        self.get_error = get_error
        self.upsert_error = upsert_error
        self.index_done_calls = 0
        self.upserts = []

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserts.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _processor(storage):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.lightrag = type("FakeLightRAG", (), {"doc_status": storage})()
    return dummy


@pytest.mark.asyncio
async def test_appends_chunk_ids_and_preserves_existing_fields():
    storage = FakeDocStatusStorage(
        {
            "doc-1": {
                "status": "processed",
                "chunks_list": ["chunk-text-1"],
                "chunks_count": 1,
                "content_summary": "keep me",
                "file_path": "manual.pdf",
            }
        }
    )
    dummy = _processor(storage)

    await dummy._update_doc_status_with_chunks_type_aware(
        "doc-1", ["chunk-mm-1", "chunk-mm-2"]
    )

    updated = storage.records["doc-1"]
    assert updated["chunks_list"] == ["chunk-text-1", "chunk-mm-1", "chunk-mm-2"]
    assert updated["chunks_count"] == 3
    assert updated["content_summary"] == "keep me"
    assert updated["file_path"] == "manual.pdf"
    assert updated["status"] == "processed"
    assert "updated_at" in updated
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_missing_chunks_list_starts_from_empty():
    storage = FakeDocStatusStorage({"doc-1": {"status": "processed"}})
    dummy = _processor(storage)

    await dummy._update_doc_status_with_chunks_type_aware("doc-1", ["chunk-mm-1"])

    assert storage.records["doc-1"]["chunks_list"] == ["chunk-mm-1"]
    assert storage.records["doc-1"]["chunks_count"] == 1


@pytest.mark.asyncio
async def test_missing_document_is_noop():
    storage = FakeDocStatusStorage()
    dummy = _processor(storage)

    await dummy._update_doc_status_with_chunks_type_aware("doc-missing", ["chunk-mm-1"])

    assert storage.records == {}
    assert storage.upserts == []
    assert storage.index_done_calls == 0


@pytest.mark.asyncio
async def test_storage_errors_are_swallowed():
    dummy = _processor(FakeDocStatusStorage(get_error=RuntimeError("disk down")))
    await dummy._update_doc_status_with_chunks_type_aware("doc-1", ["chunk-mm-1"])

    dummy = _processor(
        FakeDocStatusStorage(
            {"doc-1": {"chunks_list": [], "chunks_count": 0}},
            upsert_error=RuntimeError("write failed"),
        )
    )
    await dummy._update_doc_status_with_chunks_type_aware("doc-1", ["chunk-mm-1"])
