"""Regression tests for multimodal chunk storage and doc_status chunk lists."""

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class FakeKVStorage:
    def __init__(self, fail=False):
        self.fail = fail
        self.upserts = []

    async def upsert(self, data):
        if self.fail:
            raise RuntimeError("storage failed")
        self.upserts.append(data)


class FakeDocStatusStorage:
    def __init__(self, records=None, get_error=None):
        self.records = records or {}
        self.get_error = get_error
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        for doc_id, payload in data.items():
            self.records[doc_id] = payload

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor(*, text_chunks=None, chunks_vdb=None, doc_status=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "text_chunks": text_chunks if text_chunks is not None else FakeKVStorage(),
            "chunks_vdb": chunks_vdb if chunks_vdb is not None else FakeKVStorage(),
            "doc_status": doc_status
            if doc_status is not None
            else FakeDocStatusStorage(),
        },
    )()
    return processor


@pytest.mark.asyncio
async def test_store_chunks_writes_text_chunks_and_vector_db():
    text_chunks = FakeKVStorage()
    chunks_vdb = FakeKVStorage()
    processor = _make_processor(text_chunks=text_chunks, chunks_vdb=chunks_vdb)
    chunks = {"chunk-1": {"content": "table chunk", "tokens": 3}}

    await processor._store_chunks_to_lightrag_storage_type_aware(chunks)

    assert text_chunks.upserts == [chunks]
    assert chunks_vdb.upserts == [chunks]


@pytest.mark.asyncio
async def test_store_chunks_reraises_storage_errors():
    processor = _make_processor(text_chunks=FakeKVStorage(fail=True))

    with pytest.raises(RuntimeError, match="storage failed"):
        await processor._store_chunks_to_lightrag_storage_type_aware(
            {"chunk-1": {"content": "x"}}
        )

    assert any("Error storing chunks to storage" in e for e in processor.logger.errors)


@pytest.mark.asyncio
async def test_update_doc_status_appends_multimodal_chunk_ids():
    storage = FakeDocStatusStorage(
        {
            "doc-1": {
                "status": "processed",
                "chunks_list": ["chunk-text"],
                "chunks_count": 1,
                "file_path": "manual.pdf",
            }
        }
    )
    processor = _make_processor(doc_status=storage)

    await processor._update_doc_status_with_chunks_type_aware(
        "doc-1", ["chunk-mm-a", "chunk-mm-b"]
    )

    assert len(storage.upserts) == 1
    payload = storage.upserts[0]["doc-1"]
    assert payload["chunks_list"] == ["chunk-text", "chunk-mm-a", "chunk-mm-b"]
    assert payload["chunks_count"] == 3
    assert payload["file_path"] == "manual.pdf"
    assert "updated_at" in payload
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_update_doc_status_noop_when_document_missing():
    storage = FakeDocStatusStorage({})
    processor = _make_processor(doc_status=storage)

    await processor._update_doc_status_with_chunks_type_aware("missing", ["chunk-1"])

    assert storage.upserts == []
    assert storage.index_done_calls == 0


@pytest.mark.asyncio
async def test_update_doc_status_swallows_storage_errors():
    storage = FakeDocStatusStorage(get_error=RuntimeError("status down"))
    processor = _make_processor(doc_status=storage)

    await processor._update_doc_status_with_chunks_type_aware("doc-1", ["chunk-1"])

    assert storage.upserts == []
    assert any(
        "Error updating doc_status with multimodal chunks" in w
        for w in processor.logger.warnings
    )
