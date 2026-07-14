from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import raganything.processor as processor_module
from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def make_processor():
    processor = ProcessorMixin()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=False,
        display_content_stats=False,
        use_full_path=False,
    )
    processor.lightrag = SimpleNamespace()
    processor._ensure_lightrag_initialized = AsyncMock(
        return_value={"success": True}
    )
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_recovers_nested_mineru_v2_text(monkeypatch):
    processor = make_processor()
    insert_text = AsyncMock()
    mark_multimodal_complete = AsyncMock()
    process_multimodal = AsyncMock()
    monkeypatch.setattr(processor_module, "insert_text_content", insert_text)
    processor._mark_multimodal_processing_complete = mark_multimodal_complete
    processor._process_multimodal_content = process_multimodal

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Safety Manual"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Disconnect power before service."}
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [
                                {"type": "text", "content": "Lock out equipment"}
                            ],
                        }
                    ]
                },
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/manuals/safety.json",
        doc_id="doc-safety",
        skip_multimodal_processing=True,
    )

    insert_text.assert_awaited_once_with(
        processor.lightrag,
        input=(
            "Safety Manual\n\n"
            "Disconnect power before service.\n\n"
            "1. Lock out equipment"
        ),
        file_paths="safety.json",
        split_by_character=None,
        split_by_character_only=False,
        ids="doc-safety",
    )
    mark_multimodal_complete.assert_awaited_once_with("doc-safety")
    process_multimodal.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_only_ingestion_persists_chunks_and_processed_status():
    processor = make_processor()
    text_chunks = FakeStorage()
    chunks_vdb = FakeStorage()
    doc_status = FakeStorage()
    insert_done = AsyncMock()
    processor.lightrag = SimpleNamespace(
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        doc_status=doc_status,
        tokenizer=SimpleNamespace(encode=lambda text: text.split()),
        _insert_done=insert_done,
    )

    await processor._insert_text_content_embedding_only(
        "First section\n\nSecond section", "manual.pdf", "doc-manual"
    )

    assert list(chunk["content"] for chunk in text_chunks.records.values()) == [
        "First section",
        "Second section",
    ]
    assert chunks_vdb.records == text_chunks.records
    assert all(
        chunk["full_doc_id"] == "doc-manual"
        and chunk["file_path"] == "manual.pdf"
        for chunk in text_chunks.records.values()
    )
    status = doc_status.records["doc-manual"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 2
    assert status["chunks_list"] == list(text_chunks.records)
    assert status["multimodal_processed"] is True
    assert text_chunks.index_done_calls == 1
    assert chunks_vdb.index_done_calls == 1
    assert doc_status.index_done_calls == 1
    insert_done.assert_awaited_once_with()
