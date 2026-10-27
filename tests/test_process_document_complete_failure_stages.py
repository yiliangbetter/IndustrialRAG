"""Failure-stage callbacks for the full document ingest orchestrator.

Operators retry or abandon a manual based on ``on_document_error``. A figure
failure reported as a text-insert or parse failure, or a completion event after
that failure, hides an index that is missing images.
"""

from types import SimpleNamespace

import pytest

import raganything.processor as processor_module
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


class EventLog:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        use_full_path=False,
        allow_embedding_only_ingestion=False,
        parser_output_dir="/tmp/out",
        parse_method="auto",
        display_content_stats=False,
        content_format="minerU",
    )
    processor.callback_manager = EventLog()
    processor.lightrag = object()
    processor.mm_calls = []
    processor.mark_calls = []
    processor.insert_kwargs = []

    async def fake_init():
        return {"success": True}

    async def fake_mark(doc_id):
        processor.mark_calls.append(doc_id)

    processor._ensure_lightrag_initialized = fake_init
    processor._mark_multimodal_processing_complete = fake_mark
    return processor


def _error_events(processor):
    return [
        kwargs
        for name, kwargs in processor.callback_manager.events
        if name == "on_document_error"
    ]


@pytest.mark.asyncio
async def test_multimodal_failure_keeps_full_path_and_skips_complete(monkeypatch):
    processor = _processor()
    content = [
        {"type": "text", "text": "Torque spec 40 Nm"},
        {"type": "image", "img_path": "/abs/fig.png"},
    ]

    async def fake_parse(*args, **kwargs):
        return content, "doc-pump"

    async def fake_mm(items, file_path, doc_id):
        processor.mm_calls.append((items, file_path, doc_id))
        raise RuntimeError("figure extract failed")

    async def fake_insert(lightrag, **kwargs):
        processor.insert_kwargs.append(kwargs)

    processor.parse_document = fake_parse
    processor._process_multimodal_content = fake_mm
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    with pytest.raises(RuntimeError, match="figure extract failed"):
        await processor.process_document_complete("manuals/pump.pdf")

    assert processor.insert_kwargs == [
        {
            "input": "Torque spec 40 Nm",
            "file_paths": "pump.pdf",
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": "doc-pump",
        }
    ]
    assert processor.mm_calls == [
        ([{"type": "image", "img_path": "/abs/fig.png"}], "pump.pdf", "doc-pump")
    ]
    assert processor.mark_calls == []

    names = [name for name, _ in processor.callback_manager.events]
    assert "on_text_insert_complete" in names
    assert "on_document_complete" not in names
    errors = _error_events(processor)
    assert len(errors) == 1
    assert errors[0]["stage"] == "multimodal"
    assert errors[0]["doc_id"] == "doc-pump"
    assert errors[0]["file_path"] == "manuals/pump.pdf"
    assert str(errors[0]["error"]) == "figure extract failed"


@pytest.mark.asyncio
async def test_image_only_failure_is_not_reported_as_parse(monkeypatch):
    processor = _processor()

    async def fake_parse(*args, **kwargs):
        return ([{"type": "image", "img_path": "/abs/fig.png"}], "doc-fig")

    async def fake_mm(items, file_path, doc_id):
        processor.mm_calls.append((file_path, doc_id))
        raise RuntimeError("figure extract failed")

    async def fake_insert(lightrag, **kwargs):
        processor.insert_kwargs.append(kwargs)

    processor.parse_document = fake_parse
    processor._process_multimodal_content = fake_mm
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    with pytest.raises(RuntimeError, match="figure extract failed"):
        await processor.process_document_complete("manuals/figures.pdf")

    assert processor.insert_kwargs == []
    assert processor.mm_calls == [("figures.pdf", "doc-fig")]
    assert processor.mark_calls == []
    names = [name for name, _ in processor.callback_manager.events]
    assert "on_text_insert_start" not in names
    assert "on_document_complete" not in names
    errors = _error_events(processor)
    assert len(errors) == 1
    assert errors[0]["stage"] == "multimodal"
    assert errors[0]["doc_id"] == "doc-fig"
    assert errors[0]["file_path"] == "manuals/figures.pdf"


@pytest.mark.asyncio
async def test_parse_failure_preserves_caller_doc_id(monkeypatch):
    processor = _processor()

    async def fake_parse(*args, **kwargs):
        raise ValueError("No content was extracted")

    async def fake_insert(lightrag, **kwargs):
        processor.insert_kwargs.append(kwargs)

    processor.parse_document = fake_parse
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    with pytest.raises(ValueError, match="No content was extracted"):
        await processor.process_document_complete(
            "manuals/pump.pdf", doc_id="doc-manual-7"
        )

    assert processor.insert_kwargs == []
    assert processor.mm_calls == []
    assert processor.mark_calls == []
    names = [name for name, _ in processor.callback_manager.events]
    assert names == ["on_document_error"]
    errors = _error_events(processor)
    assert errors[0]["stage"] == "parse"
    assert errors[0]["doc_id"] == "doc-manual-7"
    assert errors[0]["file_path"] == "manuals/pump.pdf"
    assert isinstance(errors[0]["error"], ValueError)
