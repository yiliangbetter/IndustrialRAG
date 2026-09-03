"""Multimodal ingest must report pipeline status and callbacks.

Operators watch pipeline_status during figure/table extraction. Dropping the
start/complete messages looks like a hung multimodal stage. Skipping
on_multimodal_complete after a successful batch leaves metrics and scanners
thinking figures never entered the graph.
"""

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
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


class FakeLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeDocStatusStorage:
    def __init__(self, records=None):
        self.records = records or {}

    async def get_by_id(self, key):
        return self.records.get(key)


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_multimodal_start(self, file_path, item_count=0, doc_id=None, **kw):
        self.events.append(("start", file_path, item_count, doc_id))

    def on_multimodal_complete(self, file_path, processed_count=0, doc_id=None, **kw):
        self.events.append(("complete", file_path, processed_count, doc_id))


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    processor.callback_manager = CallbackManager()
    processor.cb = RecordingCallback()
    processor.callback_manager.register(processor.cb)
    processor.batch_calls = []
    processor.individual_calls = []
    processor.mark_calls = []

    async def fake_ensure():
        return {"success": True}

    async def fake_batch(*, multimodal_items, file_path, doc_id):
        processor.batch_calls.append((list(multimodal_items), file_path, doc_id))

    async def fake_individual(items, file_path, doc_id):
        processor.individual_calls.append((list(items), file_path, doc_id))

    async def fake_mark(doc_id):
        processor.mark_calls.append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._process_multimodal_content_individual = fake_individual
    processor._mark_multimodal_processing_complete = fake_mark
    return processor


ITEMS = [{"type": "image", "img_path": "/abs/fig.png"}]


@pytest.mark.asyncio
async def test_success_writes_pipeline_start_and_complete_and_callbacks():
    processor = _make_processor()
    pipeline_status = {"latest_message": "", "history_messages": []}

    await processor._process_multimodal_content(
        ITEMS,
        "manual.pdf",
        "doc-1",
        pipeline_status=pipeline_status,
        pipeline_status_lock=FakeLock(),
    )

    assert processor.batch_calls == [(ITEMS, "manual.pdf", "doc-1")]
    assert processor.mark_calls == ["doc-1"]
    assert pipeline_status["history_messages"] == [
        "Starting multimodal content processing...",
        "Multimodal content processing complete",
    ]
    assert pipeline_status["latest_message"] == (
        "Multimodal content processing complete"
    )
    assert processor.cb.events == [
        ("start", "manual.pdf", 1, "doc-1"),
        ("complete", "manual.pdf", 1, "doc-1"),
    ]


@pytest.mark.asyncio
async def test_missing_pipeline_status_still_dispatches_callbacks():
    processor = _make_processor()

    await processor._process_multimodal_content(ITEMS, "manual.pdf", "doc-1")

    assert processor.batch_calls == [(ITEMS, "manual.pdf", "doc-1")]
    assert processor.mark_calls == ["doc-1"]
    assert processor.cb.events == [
        ("start", "manual.pdf", 1, "doc-1"),
        ("complete", "manual.pdf", 1, "doc-1"),
    ]


@pytest.mark.asyncio
async def test_batch_fallback_still_records_pipeline_start():
    processor = _make_processor()
    pipeline_status = {"latest_message": "", "history_messages": []}

    async def failing_batch(**kwargs):
        raise RuntimeError("batch extract failed")

    processor._process_multimodal_content_batch_type_aware = failing_batch

    await processor._process_multimodal_content(
        ITEMS,
        "manual.pdf",
        "doc-1",
        pipeline_status=pipeline_status,
        pipeline_status_lock=FakeLock(),
    )

    assert processor.individual_calls == [(ITEMS, "manual.pdf", "doc-1")]
    assert processor.mark_calls == ["doc-1"]
    assert pipeline_status["history_messages"][0] == (
        "Starting multimodal content processing..."
    )
    assert processor.cb.events[0] == ("start", "manual.pdf", 1, "doc-1")


@pytest.mark.asyncio
async def test_init_failure_does_not_write_pipeline_status():
    processor = _make_processor()
    pipeline_status = {"latest_message": "", "history_messages": []}

    async def fake_ensure():
        return {"success": False, "error": "parser not installed"}

    processor._ensure_lightrag_initialized = fake_ensure

    await processor._process_multimodal_content(
        ITEMS,
        "manual.pdf",
        "doc-1",
        pipeline_status=pipeline_status,
        pipeline_status_lock=FakeLock(),
    )

    assert processor.batch_calls == []
    assert pipeline_status["history_messages"] == []
    assert pipeline_status["latest_message"] == ""
