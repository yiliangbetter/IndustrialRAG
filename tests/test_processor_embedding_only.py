from unittest.mock import AsyncMock

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


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self):
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class RecordingCallbackManager:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


class DummyProcessor(ProcessorMixin):
    def __init__(self, allow_embedding_only_ingestion=True):
        self.config = type(
            "Config",
            (),
            {
                "allow_embedding_only_ingestion": allow_embedding_only_ingestion,
                "content_format": "mineru",
                "display_content_stats": False,
                "use_full_path": False,
            },
        )()
        self.logger = FakeLogger()
        self.lightrag = FakeLightRAG()
        self.callback_manager = RecordingCallbackManager()
        self.marked_multimodal_complete = []

    async def _ensure_lightrag_initialized(self):
        return {"success": True}

    async def _mark_multimodal_processing_complete(self, doc_id):
        self.marked_multimodal_complete.append(doc_id)


@pytest.mark.asyncio
async def test_embedding_only_insert_persists_chunks_and_processed_status():
    processor = DummyProcessor()

    await processor._insert_text_content_embedding_only(
        text_content="first chunk\n\nsecond chunk",
        file_ref="source_content_list_v2.json",
        doc_id="doc-abc",
    )

    text_chunks = processor.lightrag.text_chunks.records
    assert len(text_chunks) == 2
    assert processor.lightrag.chunks_vdb.records == text_chunks
    assert [chunk["content"] for chunk in text_chunks.values()] == [
        "first chunk",
        "second chunk",
    ]
    assert [chunk["chunk_order_index"] for chunk in text_chunks.values()] == [0, 1]
    assert all(
        chunk["file_path"] == "source_content_list_v2.json"
        for chunk in text_chunks.values()
    )

    doc_status = processor.lightrag.doc_status.records["doc-abc"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_list"] == list(text_chunks.keys())
    assert doc_status["chunks_count"] == 2
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "source_content_list_v2.json"
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_recovers_nested_mineru_v2_text():
    processor = DummyProcessor(allow_embedding_only_ingestion=True)
    processor._insert_text_content_embedding_only = AsyncMock()
    nested_content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Quarterly Risk Report"}
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Revenue declined"},
                        {"type": "text", "content": "in APAC."},
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
                                {"type": "text", "content": "Audit inventory"}
                            ],
                        },
                        {
                            "prefix": "2.",
                            "item_content": [
                                {"type": "text", "content": "Reconcile invoices"}
                            ],
                        },
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>Q1</td></tr></table>"},
            },
            {
                "type": "image",
                "content": {"image_caption": ["warehouse", "variance chart"]},
            },
        ]
    ]

    await processor.insert_content_list(
        nested_content_list,
        file_path="/tmp/nested_content_list_v2.json",
    )

    call = processor._insert_text_content_embedding_only.await_args.kwargs
    assert call["text_content"] == (
        "Quarterly Risk Report\n\n"
        "Revenue declined in APAC.\n\n"
        "1. Audit inventory\n2. Reconcile invoices\n\n"
        "<table><tr><td>Q1</td></tr></table>\n\n"
        "warehouse variance chart"
    )
    normalized = processor._normalize_nested_content_list(nested_content_list)
    assert len(normalized) == 5
    assert call["doc_id"] == processor._generate_content_based_doc_id(normalized)
    assert call["file_ref"] == "nested_content_list_v2.json"
    assert [event[0] for event in processor.callback_manager.events] == [
        "on_document_complete"
    ]


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_complete_without_processing(
    monkeypatch,
):
    processor = DummyProcessor(allow_embedding_only_ingestion=False)
    processor._process_multimodal_content = AsyncMock()

    async def fake_insert_text_content(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert_text_content
    )

    await processor.insert_content_list(
        [
            {"type": "text", "text": "operational summary"},
            {"type": "image", "img_path": "/tmp/image.png"},
        ],
        file_path="/tmp/source.json",
        doc_id="doc-skip",
        skip_multimodal_processing=True,
    )

    assert processor.marked_multimodal_complete == ["doc-skip"]
    processor._process_multimodal_content.assert_not_called()
    assert [event[0] for event in processor.callback_manager.events] == [
        "on_text_insert_start",
        "on_text_insert_complete",
        "on_document_complete",
    ]
