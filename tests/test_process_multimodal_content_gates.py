"""Regression tests for multimodal processing early-exit gates."""

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.errors = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg))


class FakeDocStatusStorage:
    def __init__(self, records=None):
        self.records = records or {}

    async def get_by_id(self, key):
        return self.records.get(key)


def _make_processor(records=None, init_result=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG", (), {"doc_status": FakeDocStatusStorage(records)}
    )()
    processor._batch_called = False

    async def fake_ensure():
        return init_result if init_result is not None else {"success": True}

    async def fake_batch(**kwargs):
        processor._batch_called = True

    async def fake_mark(doc_id):
        processor._marked = doc_id

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._mark_multimodal_processing_complete = fake_mark
    return processor


@pytest.mark.asyncio
async def test_process_multimodal_content_noop_when_empty():
    processor = _make_processor()

    await processor._process_multimodal_content([], "doc.pdf", "doc-1")

    assert processor._batch_called is False
    assert any("No multimodal content to process" in d for d in processor.logger.debugs)


@pytest.mark.asyncio
async def test_process_multimodal_content_skips_when_init_fails():
    processor = _make_processor(init_result={"success": False, "error": "no llm"})

    await processor._process_multimodal_content(
        [{"type": "table", "table_body": "x"}], "doc.pdf", "doc-1"
    )

    assert processor._batch_called is False
    assert any(
        "LightRAG initialization failed; skipping multimodal processing" in e
        for e in processor.logger.errors
    )


@pytest.mark.asyncio
async def test_process_multimodal_content_skips_when_already_processed():
    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": True,
            }
        }
    )

    await processor._process_multimodal_content(
        [{"type": "image", "img_path": "/tmp/a.png"}], "doc.pdf", "doc-1"
    )

    assert processor._batch_called is False
    assert any("already processed" in i for i in processor.logger.infos)


@pytest.mark.asyncio
async def test_process_multimodal_content_continues_when_text_done_mm_pending():
    processor = _make_processor(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "multimodal_processed": False,
            }
        }
    )

    await processor._process_multimodal_content(
        [{"type": "table", "table_body": "| a |"}], "doc.pdf", "doc-1"
    )

    assert processor._batch_called is True
    assert processor._marked == "doc-1"
