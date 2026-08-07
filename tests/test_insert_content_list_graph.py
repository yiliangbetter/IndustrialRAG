"""Regression tests for insert_content_list graph-path edge contracts.

Open PRs #58/#67 cover embedding-only and MinerU v2 skip_multimodal text recovery.
The remaining graph edges still control whether multimodal processors run, whether
empty-multimodal docs are marked complete, and whether text-insert callbacks fire.
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

    def on_text_insert_start(self, file_path, text_length=0, doc_id=None, **kwargs):
        self.text_insert_starts.append((file_path, text_length, doc_id))

    def on_text_insert_complete(
        self, file_path, duration_seconds=0, doc_id=None, **kwargs
    ):
        self.text_insert_completes.append((file_path, doc_id))

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.document_completes.append((file_path, doc_id))


class DummyProcessor(ProcessorMixin):
    pass


def make_graph_processor():
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
            "display_content_stats": False,
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
        processor.content_source_calls.append((list(content_list), content_format))

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
async def test_insert_content_list_graph_runs_multimodal_when_present(monkeypatch):
    """Graph path with multimodal items must set context and process them."""
    processor = make_graph_processor()
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    content_list = [
        {"type": "text", "text": "Intro"},
        {
            "type": "table",
            "table_body": "| A | B |\n|---|---|\n| 1 | 2 |",
            "table_caption": ["Specs"],
            "page_idx": 2,
        },
    ]
    captured_insert = {}

    async def fake_insert_text_content(lightrag, **kwargs):
        captured_insert["lightrag"] = lightrag
        captured_insert.update(kwargs)

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.insert_content_list(
        content_list,
        file_path="/data/report.json",
        doc_id="doc-mm-1",
        skip_multimodal_processing=False,
    )

    assert captured_insert["input"] == "Intro"
    assert captured_insert["file_paths"] == "report.json"
    assert captured_insert["ids"] == "doc-mm-1"

    assert processor.content_source_calls == [(content_list, "minerU")]
    assert len(processor.multimodal_calls) == 1
    items, file_ref, doc_id = processor.multimodal_calls[0]
    assert file_ref == "report.json"
    assert doc_id == "doc-mm-1"
    assert items == [content_list[1]]
    assert processor.marked_complete_doc_ids == []

    assert callback.text_insert_starts == [("report.json", len("Intro"), "doc-mm-1")]
    assert callback.text_insert_completes == [("report.json", "doc-mm-1")]
    assert callback.document_completes == [("/data/report.json", "doc-mm-1")]


@pytest.mark.asyncio
async def test_insert_content_list_graph_marks_complete_when_no_multimodal(
    monkeypatch,
):
    """Text-only content lists must mark multimodal complete on the graph path."""
    processor = make_graph_processor()

    async def fake_insert_text_content(*args, **kwargs):
        return

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.insert_content_list(
        [{"type": "text", "text": "Only text"}],
        file_path="plain.json",
        doc_id="doc-plain",
    )

    assert processor.content_source_calls == []
    assert processor.multimodal_calls == []
    assert processor.marked_complete_doc_ids == ["doc-plain"]


@pytest.mark.asyncio
async def test_insert_content_list_graph_skip_multimodal_marks_complete(monkeypatch):
    """skip_multimodal_processing must ingest text and skip multimodal processors."""
    processor = make_graph_processor()
    captured_insert = {}

    async def fake_insert_text_content(lightrag, **kwargs):
        captured_insert.update(kwargs)

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.insert_content_list(
        [
            {"type": "text", "text": "Body"},
            {"type": "image", "img_path": "/abs/a.png", "image_caption": ["A"]},
        ],
        file_path="mixed.json",
        doc_id="doc-skip",
        skip_multimodal_processing=True,
    )

    assert captured_insert["input"] == "Body"
    assert captured_insert["ids"] == "doc-skip"
    # Context is still set before text insert when multimodal items exist.
    assert processor.content_source_calls
    assert processor.multimodal_calls == []
    assert processor.marked_complete_doc_ids == ["doc-skip"]


@pytest.mark.asyncio
async def test_insert_content_list_graph_init_failure_raises():
    """Failed LightRAG init must abort content-list insertion."""
    processor = make_graph_processor()

    async def failing_init():
        return {"success": False, "error": "missing model"}

    processor._ensure_lightrag_initialized = failing_init

    with pytest.raises(RuntimeError, match="missing model"):
        await processor.insert_content_list(
            [{"type": "text", "text": "x"}],
            file_path="x.json",
        )
