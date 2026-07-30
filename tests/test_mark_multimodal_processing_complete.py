"""Regression tests for ProcessorMixin._mark_multimodal_processing_complete."""

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeDocStatusStorage:
    def __init__(self, records=None, get_error=None, upsert_error=None):
        self.records = records or {}
        self.get_error = get_error
        self.upsert_error = upsert_error
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserts.append(data)
        for doc_id, payload in data.items():
            self.records[doc_id] = payload

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor(records=None, get_error=None, upsert_error=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    storage = FakeDocStatusStorage(
        records=records, get_error=get_error, upsert_error=upsert_error
    )
    processor.lightrag = type("FakeLightRAG", (), {"doc_status": storage})()
    return processor, storage


@pytest.mark.asyncio
async def test_mark_multimodal_sets_flag_and_persists():
    processor, storage = _make_processor(
        {
            "doc-1": {
                "status": "processed",
                "chunks_count": 2,
                "multimodal_processed": False,
            }
        }
    )

    await processor._mark_multimodal_processing_complete("doc-1")

    assert len(storage.upserts) == 1
    payload = storage.upserts[0]["doc-1"]
    assert payload["status"] == "processed"
    assert payload["chunks_count"] == 2
    assert payload["multimodal_processed"] is True
    assert "updated_at" in payload
    assert storage.index_done_calls == 1
    assert storage.records["doc-1"]["multimodal_processed"] is True


@pytest.mark.asyncio
async def test_mark_multimodal_noop_when_document_missing():
    processor, storage = _make_processor({})

    await processor._mark_multimodal_processing_complete("missing")

    assert storage.upserts == []
    assert storage.index_done_calls == 0


@pytest.mark.asyncio
async def test_mark_multimodal_swallows_storage_errors():
    processor, storage = _make_processor(get_error=RuntimeError("storage down"))

    # Must not raise — incomplete multimodal flags are warned, not fatal.
    await processor._mark_multimodal_processing_complete("doc-1")

    assert storage.upserts == []
    assert any(
        "Error marking multimodal processing as complete" in w
        for w in processor.logger.warnings
    )

    processor, storage = _make_processor(
        {"doc-1": {"status": "processed"}},
        upsert_error=RuntimeError("upsert failed"),
    )
    await processor._mark_multimodal_processing_complete("doc-1")
    assert storage.index_done_calls == 0
    assert any(
        "Error marking multimodal processing as complete" in w
        for w in processor.logger.warnings
    )
