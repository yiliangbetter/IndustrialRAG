"""Persist/swallow contracts for marking multimodal ingest complete.

If this helper raises, the caller can leave a document HANDLING forever.
If it writes when the doc is missing, scanners see a ghost complete record.
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


class FakeDocStatus:
    def __init__(self, records=None, get_error=None, upsert_error=None):
        self.records = records or {}
        self.index_done_calls = 0
        self.get_error = get_error
        self.upsert_error = upsert_error

    async def get_by_id(self, key):
        if self.get_error:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        if self.upsert_error:
            raise self.upsert_error
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _processor(doc_status):
    proc = ProcessorMixin.__new__(ProcessorMixin)
    proc.logger = FakeLogger()
    proc.lightrag = type("FakeLightRAG", (), {"doc_status": doc_status})()
    return proc


@pytest.mark.asyncio
async def test_missing_doc_is_a_noop():
    status = FakeDocStatus()
    proc = _processor(status)

    await proc._mark_multimodal_processing_complete("doc-missing")

    assert status.records == {}
    assert status.index_done_calls == 0


@pytest.mark.asyncio
async def test_existing_doc_sets_flag_and_preserves_fields():
    status = FakeDocStatus(
        {
            "doc-1": {
                "status": "processed",
                "chunks_count": 3,
                "file_path": "a.pdf",
            }
        }
    )
    proc = _processor(status)

    await proc._mark_multimodal_processing_complete("doc-1")

    record = status.records["doc-1"]
    assert record["multimodal_processed"] is True
    assert record["status"] == "processed"
    assert record["chunks_count"] == 3
    assert record["file_path"] == "a.pdf"
    assert "updated_at" in record
    assert status.index_done_calls == 1


@pytest.mark.asyncio
async def test_get_by_id_error_is_swallowed():
    status = FakeDocStatus(get_error=RuntimeError("storage down"))
    proc = _processor(status)

    await proc._mark_multimodal_processing_complete("doc-1")

    assert status.records == {}
    assert status.index_done_calls == 0


@pytest.mark.asyncio
async def test_upsert_error_is_swallowed():
    status = FakeDocStatus(
        {"doc-1": {"status": "processed"}},
        upsert_error=RuntimeError("write failed"),
    )
    proc = _processor(status)

    await proc._mark_multimodal_processing_complete("doc-1")

    assert status.records["doc-1"] == {"status": "processed"}
    assert status.index_done_calls == 0
