"""Regression tests for process_document_complete graph (non-embedding) path.

Open PR #58 covers the embedding-only short-circuit. The default LightRAG
graph path still owns text insert wiring, multimodal staging, context-source
setup, empty-multimodal completion, and document_error callbacks. Regressions
here can skip KG multimodal work, leave documents incomplete, or drop errors.
"""

from __future__ import annotations

import pytest

import raganything.processor as processor_module
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
        self.text_insert_starts = []
        self.text_insert_completes = []
        self.document_completes = []
        self.document_errors = []

    def on_text_insert_start(self, file_path, text_length=0, doc_id=None, **kwargs):
        self.text_insert_starts.append((file_path, text_length, doc_id))

    def on_text_insert_complete(
        self, file_path, duration_seconds=0, doc_id=None, **kwargs
    ):
        self.text_insert_completes.append((file_path, doc_id, duration_seconds >= 0))

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.document_completes.append((file_path, doc_id))

    def on_document_error(
        self, file_path, doc_id=None, stage="", error="", **kwargs
    ):
        self.document_errors.append((file_path, doc_id, stage, str(error)))


class DummyProcessor(ProcessorMixin):
    pass


def make_graph_processor(tmp_path):
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
            "parser_output_dir": str(tmp_path / "output"),
            "use_full_path": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.lightrag = object()
    processor.callback_manager = CallbackManager()
    processor.content_source_calls = []
    processor.multimodal_calls = []
    processor.marked_complete_doc_ids = []

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    def set_content_source_for_context(content_list, content_format):
        processor.content_source_calls.append((content_list, content_format))

    async def fake_process_multimodal_content(items, file_ref, doc_id):
        processor.multimodal_calls.append((items, file_ref, doc_id))

    async def fake_mark_multimodal_processing_complete(doc_id):
        processor.marked_complete_doc_ids.append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.set_content_source_for_context = set_content_source_for_context
    processor._process_multimodal_content = fake_process_multimodal_content
    processor._mark_multimodal_processing_complete = (
        fake_mark_multimodal_processing_complete
    )
    return processor


@pytest.mark.asyncio
async def test_process_document_complete_graph_inserts_text_and_multimodal(
    tmp_path, monkeypatch
):
    """Text + multimodal docs must insert text, set context, and run multimodal."""
    processor = make_graph_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    content_list = [
        {"type": "text", "text": "Pump overview"},
        {
            "type": "image",
            "img_path": "/abs/pump.png",
            "image_caption": ["Pump"],
            "page_idx": 1,
        },
    ]

    async def fake_parse_document(*args, **kwargs):
        return content_list, "doc-graph-1"

    captured_insert = {}

    async def fake_insert_text_content(lightrag, **kwargs):
        captured_insert["lightrag"] = lightrag
        captured_insert.update(kwargs)

    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.process_document_complete(
        "/docs/manual.pdf",
        split_by_character="\n",
        split_by_character_only=True,
        doc_id="doc-override",
    )

    assert captured_insert["lightrag"] is processor.lightrag
    assert captured_insert["input"] == "Pump overview"
    assert captured_insert["file_paths"] == "manual.pdf"
    assert captured_insert["ids"] == "doc-override"
    assert captured_insert["split_by_character"] == "\n"
    assert captured_insert["split_by_character_only"] is True

    assert processor.content_source_calls == [(content_list, "minerU")]
    assert len(processor.multimodal_calls) == 1
    mm_items, file_ref, doc_id = processor.multimodal_calls[0]
    assert file_ref == "manual.pdf"
    assert doc_id == "doc-override"
    assert mm_items == [content_list[1]]
    assert processor.marked_complete_doc_ids == []

    assert callback.text_insert_starts == [
        ("manual.pdf", len("Pump overview"), "doc-override")
    ]
    assert callback.text_insert_completes == [("manual.pdf", "doc-override", True)]
    assert callback.document_completes == [("/docs/manual.pdf", "doc-override")]
    assert callback.document_errors == []


@pytest.mark.asyncio
async def test_process_document_complete_graph_marks_complete_without_multimodal(
    tmp_path, monkeypatch
):
    """Text-only graph docs must mark multimodal complete so status is not stuck."""
    processor = make_graph_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "Only prose"}], "doc-text-only")

    async def fake_insert_text_content(*args, **kwargs):
        return

    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.process_document_complete("notes.pdf")

    assert processor.content_source_calls == []
    assert processor.multimodal_calls == []
    assert processor.marked_complete_doc_ids == ["doc-text-only"]
    assert callback.document_completes == [("notes.pdf", "doc-text-only")]
    assert callback.document_errors == []


@pytest.mark.asyncio
async def test_process_document_complete_graph_reports_document_error_on_parse_failure(
    tmp_path,
):
    """Parse failures must emit document_error with stage=parse and re-raise."""
    processor = make_graph_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    async def fake_parse_document(*args, **kwargs):
        raise RuntimeError("parser exploded")

    processor.parse_document = fake_parse_document

    with pytest.raises(RuntimeError, match="parser exploded"):
        await processor.process_document_complete("broken.pdf")

    assert callback.document_completes == []
    assert len(callback.document_errors) == 1
    file_path, doc_id, stage, error = callback.document_errors[0]
    assert file_path == "broken.pdf"
    assert doc_id is None
    assert stage == "parse"
    assert "parser exploded" in error


@pytest.mark.asyncio
async def test_process_document_complete_graph_init_failure_raises(tmp_path):
    """Failed LightRAG init must abort before parse/insert."""
    processor = make_graph_processor(tmp_path)

    async def failing_init():
        return {"success": False, "error": "no llm"}

    parse_called = False

    async def fake_parse_document(*args, **kwargs):
        nonlocal parse_called
        parse_called = True
        return ([], "doc-x")

    processor._ensure_lightrag_initialized = failing_init
    processor.parse_document = fake_parse_document

    with pytest.raises(RuntimeError, match="no llm"):
        await processor.process_document_complete("any.pdf")

    assert parse_called is False
