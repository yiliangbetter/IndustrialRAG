"""Regression tests for ProcessorMixin._process_multimodal_content gates.

A wrong skip looks like a finished ingest while figures/tables never enter the
graph. A batch crash without fallback drops the whole multimodal stage.
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
    def __init__(self, records=None):
        self.records = records or {}

    async def get_by_id(self, key):
        return self.records.get(key)


def _make_processor(doc_status=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": doc_status or FakeDocStatusStorage()},
    )()
    return processor


@pytest.mark.asyncio
async def test_empty_multimodal_items_do_not_initialize_or_mark():
    processor = _make_processor()
    calls = {"ensure": 0, "batch": 0, "individual": 0, "marked": []}

    async def fake_ensure():
        calls["ensure"] += 1
        return {"success": True}

    async def fake_batch(**kwargs):
        calls["batch"] += 1

    async def fake_individual(*args, **kwargs):
        calls["individual"] += 1

    async def fake_mark(doc_id):
        calls["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._process_multimodal_content_individual = fake_individual
    processor._mark_multimodal_processing_complete = fake_mark

    await processor._process_multimodal_content([], "manual.pdf", "doc-1")

    assert calls["ensure"] == 0
    assert calls["batch"] == 0
    assert calls["individual"] == 0
    assert calls["marked"] == []


@pytest.mark.asyncio
async def test_init_failure_skips_multimodal_without_marking_complete():
    processor = _make_processor()
    calls = {"batch": 0, "marked": []}

    async def fake_ensure():
        return {"success": False, "error": "parser not installed"}

    async def fake_batch(**kwargs):
        calls["batch"] += 1

    async def fake_mark(doc_id):
        calls["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._mark_multimodal_processing_complete = fake_mark

    await processor._process_multimodal_content(
        [{"type": "image", "img_path": "/abs/fig.png"}],
        "manual.pdf",
        "doc-1",
    )

    assert calls["batch"] == 0
    assert calls["marked"] == []


@pytest.mark.asyncio
async def test_already_processed_skips_batch():
    processor = _make_processor(
        FakeDocStatusStorage(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSED,
                    "multimodal_processed": True,
                }
            }
        )
    )
    calls = {"batch": 0}

    async def fake_ensure():
        return {"success": True}

    async def fake_batch(**kwargs):
        calls["batch"] += 1

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch

    await processor._process_multimodal_content(
        [{"type": "table", "table_body": "a|b"}],
        "manual.pdf",
        "doc-1",
    )

    assert calls["batch"] == 0


@pytest.mark.asyncio
async def test_text_processed_but_multimodal_pending_still_runs():
    processor = _make_processor(
        FakeDocStatusStorage(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSED,
                    "multimodal_processed": False,
                }
            }
        )
    )
    calls = {"batch": 0, "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_batch(**kwargs):
        calls["batch"] += 1

    async def fake_mark(doc_id):
        calls["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._mark_multimodal_processing_complete = fake_mark

    await processor._process_multimodal_content(
        [{"type": "image", "img_path": "/abs/fig.png"}],
        "manual.pdf",
        "doc-1",
    )

    assert calls["batch"] == 1
    assert calls["marked"] == ["doc-1"]


@pytest.mark.asyncio
async def test_status_lookup_error_fail_open_and_continues():
    class RaisingStorage:
        async def get_by_id(self, key):
            raise RuntimeError("storage down")

    processor = _make_processor()
    processor.lightrag = type("FakeLightRAG", (), {"doc_status": RaisingStorage()})()
    calls = {"batch": 0}

    async def fake_ensure():
        return {"success": True}

    async def fake_batch(**kwargs):
        calls["batch"] += 1

    async def fake_mark(doc_id):
        return None

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._mark_multimodal_processing_complete = fake_mark

    await processor._process_multimodal_content(
        [{"type": "equation", "text": "E=mc^2"}],
        "manual.pdf",
        "doc-1",
    )

    assert calls["batch"] == 1


@pytest.mark.asyncio
async def test_batch_failure_falls_back_to_individual_then_marks_complete():
    processor = _make_processor()
    calls = {"individual": [], "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_batch(**kwargs):
        raise RuntimeError("batch extract failed")

    async def fake_individual(items, file_path, doc_id):
        calls["individual"].append((list(items), file_path, doc_id))

    async def fake_mark(doc_id):
        calls["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._process_multimodal_content_individual = fake_individual
    processor._mark_multimodal_processing_complete = fake_mark

    items = [{"type": "image", "img_path": "/abs/fig.png"}]
    await processor._process_multimodal_content(items, "manual.pdf", "doc-1")

    assert calls["individual"] == [(items, "manual.pdf", "doc-1")]
    assert calls["marked"] == ["doc-1"]
