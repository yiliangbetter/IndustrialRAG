"""Regression tests for the main process_document_complete ingest flow.

This is the user-facing parse → insert → multimodal pipeline. Wrong doc_id
selection, skipped multimodal on text-less PDFs, or missing error callbacks
look like successful ingest while retrieval and scanners disagree.
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

    def on_document_error(self, file_path, error="", stage="", doc_id=None, **kw):
        self.events.append(
            (
                "document_error",
                file_path,
                stage,
                doc_id,
                type(error).__name__,
                str(error),
            )
        )

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
async def test_init_failure_raises_and_dispatches_parse_error():
    dummy = _dummy()

    async def fake_init():
        return {"success": False, "error": "missing llm_model_func"}

    dummy._ensure_lightrag_initialized = fake_init

    with pytest.raises(RuntimeError, match="missing llm_model_func"):
        await dummy.process_document_complete("sample.pdf")

    kinds = [e[0] for e in dummy.cb.events]
    assert kinds == ["document_error"]
    assert dummy.cb.events[0][2] == "parse"
    assert dummy.mm_calls == []
    assert dummy.mark_calls == []


@pytest.mark.asyncio
async def test_provided_doc_id_wins_over_content_based_id(monkeypatch):
    dummy = _dummy()

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return (
            [{"type": "text", "text": "hello world"}],
            "doc-from-content",
        )

    dummy._ensure_lightrag_initialized = fake_init
    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(lightrag, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete(
        "/tmp/manual.pdf",
        doc_id="doc-override",
        split_by_character="\n",
        split_by_character_only=True,
    )

    assert dummy.insert_kwargs[0]["ids"] == "doc-override"
    assert dummy.insert_kwargs[0]["file_paths"] == "manual.pdf"
    assert dummy.insert_kwargs[0]["split_by_character"] == "\n"
    assert dummy.insert_kwargs[0]["split_by_character_only"] is True
    assert dummy.mark_calls == ["doc-override"]
    assert dummy.mm_calls == []
    assert dummy.cb.events[-1] == (
        "document_complete",
        "/tmp/manual.pdf",
        "doc-override",
    )


@pytest.mark.asyncio
async def test_empty_text_still_processes_multimodal(monkeypatch):
    dummy = _dummy()

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return (
            [{"type": "image", "img_path": "/abs/fig.png", "page_idx": 0}],
            "doc-img",
        )

    dummy._ensure_lightrag_initialized = fake_init
    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete("figures.pdf")

    assert dummy.insert_kwargs == []
    assert dummy.mm_calls == [
        (
            [{"type": "image", "img_path": "/abs/fig.png", "page_idx": 0}],
            "figures.pdf",
            "doc-img",
        )
    ]
    assert dummy.mark_calls == []


@pytest.mark.asyncio
async def test_no_multimodal_marks_processing_complete(monkeypatch):
    dummy = _dummy()

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "only prose"}], "doc-text")

    dummy._ensure_lightrag_initialized = fake_init
    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        return None

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete("notes.pdf")

    assert dummy.mm_calls == []
    assert dummy.mark_calls == ["doc-text"]


@pytest.mark.asyncio
async def test_insert_failure_reports_text_insert_stage(monkeypatch):
    dummy = _dummy()

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "hello"}], "doc-1")

    dummy._ensure_lightrag_initialized = fake_init
    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        raise RuntimeError("ainsert exploded")

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    with pytest.raises(RuntimeError, match="ainsert exploded"):
        await dummy.process_document_complete("notes.pdf")

    kinds = [e[0] for e in dummy.cb.events]
    assert "text_insert_start" in kinds
    error_events = [e for e in dummy.cb.events if e[0] == "document_error"]
    assert error_events[0][2] == "text_insert"
    assert error_events[0][3] == "doc-1"
    assert dummy.mark_calls == []
