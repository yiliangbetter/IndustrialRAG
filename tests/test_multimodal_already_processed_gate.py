"""Regression tests for the multimodal already-processed / init gates.

Re-running processors after LightRAG marks text PROCESSED would duplicate
KG entities. Skipping when init failed or when the multimodal flag is set
is the only thing that keeps ingest idempotent.
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


class FakeDocStatus:
    def __init__(self, records=None, raise_on_get=False):
        self.records = records or {}
        self.raise_on_get = raise_on_get

    async def get_by_id(self, key):
        if self.raise_on_get:
            raise RuntimeError("status storage unavailable")
        return self.records.get(key)


class RecordingCallback:
    def __init__(self):
        self.events = []

    def dispatch(self, event, **kwargs):
        self.events.append((event, kwargs))


def _processor(doc_records=None, raise_on_get=False):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type("Config", (), {"use_full_path": False})()
    dummy.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatus(doc_records, raise_on_get=raise_on_get)},
    )()
    dummy.callback_manager = RecordingCallback()
    dummy.batch_calls = []
    dummy.mark_calls = []
    dummy.init_calls = []
    return dummy


def _patch_hooks(dummy, init_result):
    async def fake_init():
        dummy.init_calls.append(True)
        return init_result

    async def fake_batch(*, multimodal_items, file_path, doc_id):
        dummy.batch_calls.append((multimodal_items, file_path, doc_id))

    async def fake_mark(doc_id):
        dummy.mark_calls.append(doc_id)

    dummy._ensure_lightrag_initialized = fake_init
    dummy._process_multimodal_content_batch_type_aware = fake_batch
    dummy._mark_multimodal_processing_complete = fake_mark


@pytest.mark.asyncio
async def test_empty_items_do_not_initialize_or_process():
    dummy = _processor()
    _patch_hooks(dummy, {"success": True})

    await dummy._process_multimodal_content([], "doc.pdf", "doc-1")

    assert dummy.init_calls == []
    assert dummy.batch_calls == []
    assert dummy.mark_calls == []
    assert dummy.callback_manager.events == []


@pytest.mark.asyncio
async def test_init_failure_skips_multimodal_batch():
    dummy = _processor()
    _patch_hooks(dummy, {"success": False, "error": "no llm"})
    items = [{"type": "image", "img_path": "/tmp/a.png"}]

    await dummy._process_multimodal_content(items, "doc.pdf", "doc-1")

    assert dummy.init_calls == [True]
    assert dummy.batch_calls == []
    assert dummy.mark_calls == []


@pytest.mark.asyncio
async def test_already_processed_flag_skips_batch():
    dummy = _processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": True,
            }
        }
    )
    _patch_hooks(dummy, {"success": True})
    items = [{"type": "table", "table_body": "a|b"}]

    await dummy._process_multimodal_content(items, "doc.pdf", "doc-1")

    assert dummy.batch_calls == []
    assert dummy.mark_calls == []


@pytest.mark.asyncio
async def test_text_processed_but_multimodal_pending_continues():
    dummy = _processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": False,
            }
        }
    )
    _patch_hooks(dummy, {"success": True})
    items = [{"type": "equation", "text": "E=mc^2"}]

    await dummy._process_multimodal_content(items, "doc.pdf", "doc-1")

    assert dummy.batch_calls == [(items, "doc.pdf", "doc-1")]
    assert dummy.mark_calls == ["doc-1"]
    assert dummy.callback_manager.events[0][0] == "on_multimodal_start"
    assert dummy.callback_manager.events[-1][0] == "on_multimodal_complete"


@pytest.mark.asyncio
async def test_status_lookup_error_fail_open_and_process():
    dummy = _processor(raise_on_get=True)
    _patch_hooks(dummy, {"success": True})
    items = [{"type": "image", "img_path": "/tmp/a.png"}]

    await dummy._process_multimodal_content(items, "doc.pdf", "doc-1")

    assert dummy.batch_calls == [(items, "doc.pdf", "doc-1")]
    assert dummy.mark_calls == ["doc-1"]
