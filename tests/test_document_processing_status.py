"""Completion status is the gate for treating a document as indexed."""

import pytest

from raganything.processor import ProcessorMixin

FROZEN_UPDATED_AT = "2026-10-26T10:00:00+00:00"


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class StatusStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0
        self.upserts = []
        self.get_error = None
        self.upsert_error = None
        self.index_error = None

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
        if self.index_error is not None:
            raise self.index_error


def make_processor(storage):
    processor = ProcessorMixin()
    processor.logger = FakeLogger()
    processor.lightrag = type("LightRAG", (), {"doc_status": storage})()
    return processor


def freeze_updated_at(monkeypatch):
    monkeypatch.setattr(
        "raganything.processor.time.strftime",
        lambda _fmt: FROZEN_UPDATED_AT,
    )


@pytest.mark.asyncio
async def test_missing_document_stays_incomplete_and_mark_does_not_create_a_row():
    storage = StatusStorage()
    processor = make_processor(storage)

    assert await processor.is_document_fully_processed("doc-missing") is False

    status = await processor.get_document_processing_status("doc-missing")
    assert status == {
        "exists": False,
        "text_processed": False,
        "multimodal_processed": False,
        "fully_processed": False,
        "chunks_count": 0,
    }
    assert "error" not in status
    assert "raw_status" not in status

    await processor._mark_multimodal_processing_complete("doc-missing")

    assert storage.records == {}
    assert storage.upserts == []
    assert storage.index_done_calls == 0
    assert await processor.is_document_fully_processed("doc-missing") is False


@pytest.mark.asyncio
async def test_mark_sets_multimodal_flag_without_promoting_text_status(monkeypatch):
    freeze_updated_at(monkeypatch)
    storage = StatusStorage()
    storage.records["doc-text"] = {
        "status": "processed",
        "multimodal_processed": False,
        "chunks_count": 0,
        "chunks_list": [],
        "content_summary": "cover page",
        "scheme_name": "tenant-a",
        "file_path": "manual.pdf",
        "error_msg": "",
        "updated_at": "old",
    }
    storage.records["doc-failed"] = {
        "status": "failed",
        "multimodal_processed": False,
        "chunks_count": 4,
        "chunks_list": ["c1", "c2", "c3", "c4"],
        "content_summary": "partial extract",
        "scheme_name": "tenant-a",
        "file_path": "manual.pdf",
        "error_msg": "table parse failed",
        "updated_at": "old",
    }
    processor = make_processor(storage)

    assert await processor.is_document_fully_processed("doc-text") is False
    text_status = await processor.get_document_processing_status("doc-text")
    assert text_status["exists"] is True
    assert text_status["text_processed"] is True
    assert text_status["multimodal_processed"] is False
    assert text_status["fully_processed"] is False
    assert text_status["chunks_count"] == 0
    assert text_status["chunks_list"] == []
    assert text_status["status"] == "processed"
    assert text_status["raw_status"]["scheme_name"] == "tenant-a"
    assert text_status["raw_status"]["content_summary"] == "cover page"

    await processor._mark_multimodal_processing_complete("doc-text")
    await processor._mark_multimodal_processing_complete("doc-failed")

    text_row = storage.records["doc-text"]
    assert text_row["status"] == "processed"
    assert text_row["multimodal_processed"] is True
    assert text_row["chunks_count"] == 0
    assert text_row["chunks_list"] == []
    assert text_row["content_summary"] == "cover page"
    assert text_row["scheme_name"] == "tenant-a"
    assert text_row["file_path"] == "manual.pdf"
    assert text_row["error_msg"] == ""
    assert text_row["updated_at"] == FROZEN_UPDATED_AT
    assert await processor.is_document_fully_processed("doc-text") is True
    completed = await processor.get_document_processing_status("doc-text")
    assert completed["fully_processed"] is True
    assert completed["chunks_count"] == 0

    failed_row = storage.records["doc-failed"]
    assert failed_row["status"] == "failed"
    assert failed_row["multimodal_processed"] is True
    assert failed_row["chunks_count"] == 4
    assert failed_row["chunks_list"] == ["c1", "c2", "c3", "c4"]
    assert failed_row["error_msg"] == "table parse failed"
    assert failed_row["content_summary"] == "partial extract"
    assert failed_row["scheme_name"] == "tenant-a"
    assert failed_row["updated_at"] == FROZEN_UPDATED_AT
    assert await processor.is_document_fully_processed("doc-failed") is False
    failed_status = await processor.get_document_processing_status("doc-failed")
    assert failed_status["text_processed"] is False
    assert failed_status["multimodal_processed"] is True
    assert failed_status["fully_processed"] is False
    assert failed_status["chunks_count"] == 4
    assert failed_status["status"] == "failed"
    assert storage.index_done_calls == 2


@pytest.mark.asyncio
async def test_incomplete_text_status_stays_incomplete_after_mark(monkeypatch):
    freeze_updated_at(monkeypatch)
    storage = StatusStorage()
    storage.records["doc-case"] = {
        "status": "Processed",
        "multimodal_processed": True,
        "chunks_count": 1,
        "chunks_list": ["c1"],
        "file_path": "Manual.PDF",
    }
    storage.records["doc-partial"] = {"file_path": "notes.pdf"}
    processor = make_processor(storage)

    case_status = await processor.get_document_processing_status("doc-case")
    assert case_status["text_processed"] is False
    assert case_status["multimodal_processed"] is True
    assert case_status["fully_processed"] is False
    assert case_status["chunks_count"] == 1
    assert case_status["chunks_list"] == ["c1"]
    assert await processor.is_document_fully_processed("doc-case") is False

    partial_status = await processor.get_document_processing_status("doc-partial")
    assert partial_status["exists"] is True
    assert partial_status["text_processed"] is False
    assert partial_status["multimodal_processed"] is False
    assert partial_status["fully_processed"] is False
    assert partial_status["status"] == ""
    assert partial_status["chunks_count"] == 0
    assert partial_status["chunks_list"] == []
    assert partial_status["raw_status"]["file_path"] == "notes.pdf"
    partial_status["chunks_list"].append("leak")
    again = await processor.get_document_processing_status("doc-partial")
    assert again["chunks_list"] == []

    await processor._mark_multimodal_processing_complete("doc-case")
    await processor._mark_multimodal_processing_complete("doc-partial")

    assert storage.records["doc-case"]["status"] == "Processed"
    assert storage.records["doc-case"]["file_path"] == "Manual.PDF"
    assert storage.records["doc-case"]["chunks_list"] == ["c1"]
    assert await processor.is_document_fully_processed("doc-case") is False

    partial_row = storage.records["doc-partial"]
    assert partial_row["file_path"] == "notes.pdf"
    assert partial_row["multimodal_processed"] is True
    assert "status" not in partial_row
    assert partial_row["updated_at"] == FROZEN_UPDATED_AT
    assert await processor.is_document_fully_processed("doc-partial") is False
    marked = await processor.get_document_processing_status("doc-partial")
    assert marked["text_processed"] is False
    assert marked["multimodal_processed"] is True
    assert marked["fully_processed"] is False


@pytest.mark.asyncio
async def test_status_read_and_write_failures_fail_closed(monkeypatch):
    freeze_updated_at(monkeypatch)
    read_storage = StatusStorage()
    read_storage.records["doc-ready"] = {
        "status": "processed",
        "multimodal_processed": True,
    }
    read_storage.get_error = RuntimeError("doc status unreadable")
    processor = make_processor(read_storage)

    assert await processor.is_document_fully_processed("doc-ready") is False
    status = await processor.get_document_processing_status("doc-ready")
    assert status["exists"] is False
    assert status["error"] == "doc status unreadable"
    assert status["text_processed"] is False
    assert status["multimodal_processed"] is False
    assert status["fully_processed"] is False
    assert status["chunks_count"] == 0
    assert "raw_status" not in status

    await processor._mark_multimodal_processing_complete("doc-ready")
    assert read_storage.upserts == []
    assert read_storage.index_done_calls == 0
    assert read_storage.records["doc-ready"] == {
        "status": "processed",
        "multimodal_processed": True,
    }

    write_storage = StatusStorage()
    original = {
        "status": "processed",
        "multimodal_processed": False,
        "scheme_name": "tenant-a",
        "chunks_list": ["c1"],
        "chunks_count": 1,
    }
    write_storage.records["doc-write"] = dict(original)
    write_storage.records["doc-write"]["chunks_list"] = ["c1"]
    write_storage.upsert_error = RuntimeError("status upsert failed")
    writer = make_processor(write_storage)

    await writer._mark_multimodal_processing_complete("doc-write")

    assert write_storage.records["doc-write"]["multimodal_processed"] is False
    assert write_storage.records["doc-write"]["scheme_name"] == "tenant-a"
    assert write_storage.records["doc-write"]["chunks_list"] == ["c1"]
    assert write_storage.records["doc-write"]["chunks_count"] == 1
    assert write_storage.records["doc-write"]["status"] == "processed"
    assert write_storage.index_done_calls == 0
    assert await writer.is_document_fully_processed("doc-write") is False

    persist_storage = StatusStorage()
    persist_storage.records["doc-persist"] = {
        "status": "processed",
        "multimodal_processed": False,
        "scheme_name": "tenant-a",
    }
    persist_storage.index_error = RuntimeError("disk full")
    persister = make_processor(persist_storage)

    await persister._mark_multimodal_processing_complete("doc-persist")

    assert persist_storage.records["doc-persist"]["multimodal_processed"] is True
    assert persist_storage.records["doc-persist"]["status"] == "processed"
    assert persist_storage.records["doc-persist"]["scheme_name"] == "tenant-a"
    assert persist_storage.records["doc-persist"]["updated_at"] == FROZEN_UPDATED_AT
    assert persist_storage.index_done_calls == 1
    assert await persister.is_document_fully_processed("doc-persist") is True
