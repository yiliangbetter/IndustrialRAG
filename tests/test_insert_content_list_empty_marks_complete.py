"""Empty content-list ingest must still mark the document complete.

An empty or whitespace-only list has no text insert and no multimodal items.
If the completion mark or ``on_document_complete`` is skipped, scanners treat
the doc as unfinished and re-ingest forever (or never surface it as ready).
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


class RecordingDocumentCallback(ProcessingCallback):
    def __init__(self):
        self.document_completes = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.document_completes.append((file_path, doc_id))


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
            "display_content_stats": False,
            "parse_method": "auto",
            "parser": "mineru",
            "use_full_path": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.lightrag = object()
    processor.callback_manager = CallbackManager()
    processor.parse_cache = None

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_empty_content_list_marks_complete_and_skips_insert(monkeypatch):
    import raganything.processor as processor_module

    processor = _processor()
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    mm_calls = []
    mark_calls = []
    insert_calls = []

    async def fake_mm(*args, **kwargs):
        mm_calls.append((args, kwargs))

    async def fake_mark(doc_id):
        mark_calls.append(doc_id)

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [],
        file_path="empty.pdf",
        doc_id="doc-empty",
    )

    assert insert_calls == []
    assert mm_calls == []
    assert mark_calls == ["doc-empty"]
    assert callback.document_completes == [("empty.pdf", "doc-empty")]


@pytest.mark.asyncio
async def test_whitespace_only_text_blocks_are_treated_as_empty(monkeypatch):
    import raganything.processor as processor_module

    processor = _processor()
    mark_calls = []
    insert_calls = []

    async def fake_mark(doc_id):
        mark_calls.append(doc_id)

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    async def fake_mm(*args, **kwargs):
        raise AssertionError("no multimodal items expected")

    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [{"type": "text", "text": "   "}, {"type": "text", "text": "\n\t"}],
        file_path="blank.pdf",
        doc_id="doc-blank",
    )

    assert insert_calls == []
    assert mark_calls == ["doc-blank"]
