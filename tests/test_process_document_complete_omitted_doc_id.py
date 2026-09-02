"""process_document_complete must use the parse-time content id when doc_id is omitted.

Duplicate plant manuals collapse on that hash. If omitted doc_id were replaced
with a path-derived id, the same PDF ingested from two folders would fork the
graph and citations would no longer match retrieval.
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

    def on_document_complete(self, file_path, doc_id=None, **kw):
        self.events.append(("document_complete", file_path, doc_id))

    def on_text_insert_start(self, file_path, **kw):
        self.events.append(("text_insert_start", file_path))

    def on_text_insert_complete(self, file_path, **kw):
        self.events.append(("text_insert_complete", file_path))


def _dummy():
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": "./output",
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
        },
    )()
    dummy.lightrag = object()
    dummy.callback_manager = CallbackManager()
    dummy.cb = RecordingCallback()
    dummy.callback_manager.register(dummy.cb)
    dummy.mm_calls = []
    dummy.mark_calls = []
    dummy.insert_kwargs = []

    async def fake_mm(items, file_path, doc_id):
        dummy.mm_calls.append((list(items), file_path, doc_id))

    async def fake_mark(doc_id):
        dummy.mark_calls.append(doc_id)

    dummy._process_multimodal_content = fake_mm
    dummy._mark_multimodal_processing_complete = fake_mark
    return dummy


@pytest.mark.asyncio
async def test_omitted_doc_id_uses_content_based_id_for_insert(monkeypatch):
    dummy = _dummy()

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return (
            [{"type": "text", "text": "pump isolation procedure"}],
            "doc-from-content",
        )

    dummy._ensure_lightrag_initialized = fake_init
    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(lightrag, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete("/data/uploads/line-a/pump.pdf")

    assert dummy.insert_kwargs[0]["ids"] == "doc-from-content"
    assert dummy.insert_kwargs[0]["file_paths"] == "pump.pdf"
    assert dummy.insert_kwargs[0]["input"] == "pump isolation procedure"
    assert dummy.mark_calls == ["doc-from-content"]
    assert dummy.mm_calls == []
    assert dummy.cb.events[-1] == (
        "document_complete",
        "/data/uploads/line-a/pump.pdf",
        "doc-from-content",
    )
