"""insert_content_list must emit text-insert callbacks around LightRAG insert.

Operators and MetricsCallback use on_text_insert_start/complete to know that
searchable text entered storage. A silent skip of those events looks like a
successful ingest with no text stage.
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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_text_insert_start(self, file_path, text_length=0, doc_id=None, **kw):
        self.events.append(("start", file_path, text_length, doc_id))

    def on_text_insert_complete(self, file_path, doc_id=None, **kw):
        self.events.append(("complete", file_path, doc_id))

    def on_document_complete(self, file_path, doc_id=None, **kw):
        self.events.append(("document_complete", file_path, doc_id))


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "display_content_stats": False,
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
        },
    )()
    processor.lightrag = object()
    processor.callback_manager = CallbackManager()
    processor.cb = RecordingCallback()
    processor.callback_manager.register(processor.cb)
    processor.insert_calls = []
    processor.mark_calls = []

    async def fake_ensure():
        return {"success": True}

    async def fake_mark(doc_id):
        processor.mark_calls.append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._mark_multimodal_processing_complete = fake_mark
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_dispatches_text_insert_callbacks(monkeypatch):
    import raganything.processor as processor_module

    processor = _make_processor()

    async def fake_insert(lightrag, input, **kwargs):
        processor.insert_calls.append({"input": input, **kwargs})

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [{"type": "text", "text": "pump torque table"}],
        file_path="/abs/manuals/line-a.pdf",
        doc_id="doc-fixed",
    )

    assert processor.insert_calls[0]["input"] == "pump torque table"
    assert processor.insert_calls[0]["file_paths"] == "line-a.pdf"
    assert processor.insert_calls[0]["ids"] == "doc-fixed"
    assert processor.cb.events[0] == (
        "start",
        "line-a.pdf",
        len("pump torque table"),
        "doc-fixed",
    )
    assert processor.cb.events[1][0] == "complete"
    assert processor.cb.events[1][1] == "line-a.pdf"
    assert processor.cb.events[-1] == (
        "document_complete",
        "/abs/manuals/line-a.pdf",
        "doc-fixed",
    )
    assert processor.mark_calls == ["doc-fixed"]
