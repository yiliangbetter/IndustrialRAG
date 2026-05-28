"""Regression tests for direct content-list ingestion paths."""

from __future__ import annotations

from types import SimpleNamespace

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


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(*, allow_embedding_only_ingestion=False):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=allow_embedding_only_ingestion,
        content_format="minerU",
        display_content_stats=False,
        parser="mineru",
        use_full_path=False,
    )
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = None

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_embedding_only_content_list_flattens_mineru_v2_blocks_to_chunks():
    processor = make_processor(allow_embedding_only_ingestion=True)
    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Manual Title"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "First sentence"},
                        {"content": [{"type": "text", "content": "nested text"}]},
                    ]
                },
            },
        ],
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "1.",
                        "item_content": [{"type": "text", "content": "Check oil"}],
                    },
                    {"item_content": "Record pressure"},
                ]
            },
        },
        {"type": "table", "content": {"html": "<table><tr><td>42</td></tr></table>"}},
        {"type": "image", "content": {"image_caption": ["Pump diagram", "front view"]}},
        "ignored non-dict wrapper item",
    ]

    await processor.insert_content_list(
        content_list,
        file_path="manuals/pump_content_list_v2.json",
        doc_id="doc-fixed",
    )

    chunk_contents = [
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    ]
    assert chunk_contents == [
        "Manual Title",
        "First sentence nested text",
        "1. Check oil\nRecord pressure",
        "<table><tr><td>42</td></tr></table>",
        "Pump diagram front view",
    ]
    assert processor.lightrag.chunks_vdb.records == processor.lightrag.text_chunks.records

    status = processor.lightrag.doc_status.records["doc-fixed"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["file_path"] == "pump_content_list_v2.json"
    assert status["chunks_count"] == 5
    assert status["multimodal_processed"] is True
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_skip_multimodal_processing_marks_complete_after_text_insert(monkeypatch):
    processor = make_processor()
    inserted = {}
    marked_complete = []
    processed_multimodal = []

    async def fake_insert_text_content(*args, **kwargs):
        inserted["args"] = args
        inserted["kwargs"] = kwargs

    async def fake_mark_multimodal_processing_complete(doc_id):
        marked_complete.append(doc_id)

    async def fake_process_multimodal_content(items, file_path, doc_id):
        processed_multimodal.append((items, file_path, doc_id))

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert_text_content
    )
    processor._mark_multimodal_processing_complete = (
        fake_mark_multimodal_processing_complete
    )
    processor._process_multimodal_content = fake_process_multimodal_content

    await processor.insert_content_list(
        [
            {"type": "text", "text": "safe operating procedure"},
            {"type": "image", "img_path": "/tmp/diagram.png"},
        ],
        file_path="/tmp/manual.pdf",
        doc_id="doc-skip-mm",
        skip_multimodal_processing=True,
    )

    assert inserted["kwargs"]["input"] == "safe operating procedure"
    assert inserted["kwargs"]["file_paths"] == "manual.pdf"
    assert inserted["kwargs"]["ids"] == "doc-skip-mm"
    assert marked_complete == ["doc-skip-mm"]
    assert processed_multimodal == []
