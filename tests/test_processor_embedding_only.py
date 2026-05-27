from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raganything.base import DocStatus
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
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_text_insert_start(self, file_path, **kwargs):
        self.events.append(("on_text_insert_start", file_path))

    def on_text_insert_complete(self, file_path, **kwargs):
        self.events.append(("on_text_insert_complete", file_path))

    def on_multimodal_start(self, file_path, **kwargs):
        self.events.append(("on_multimodal_start", file_path))

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.events.append(("on_document_complete", file_path, doc_id))


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(allow_embedding_only_ingestion=False):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=allow_embedding_only_ingestion,
        content_format="minerU",
        display_content_stats=False,
        use_full_path=False,
    )
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()
    return processor


@pytest.mark.asyncio
async def test_embedding_only_insert_persists_chunks_vectors_and_doc_status():
    processor = make_processor(allow_embedding_only_ingestion=True)

    await processor._insert_text_content_embedding_only(
        "First paragraph.\n\nSecond paragraph.",
        file_ref="sample_content_list_v2.json",
        doc_id="doc-storage",
    )

    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1

    assert len(processor.lightrag.text_chunks.records) == 2
    assert (
        processor.lightrag.chunks_vdb.records
        == processor.lightrag.text_chunks.records
    )
    chunks = list(processor.lightrag.text_chunks.records.values())
    assert [chunk["content"] for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert [chunk["chunk_order_index"] for chunk in chunks] == [0, 1]
    assert all(chunk["full_doc_id"] == "doc-storage" for chunk in chunks)
    assert all(chunk["file_path"] == "sample_content_list_v2.json" for chunk in chunks)

    status = processor.lightrag.doc_status.records["doc-storage"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 2
    assert status["chunks_list"] == list(processor.lightrag.text_chunks.records.keys())
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "sample_content_list_v2.json"


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_recovers_mineru_v2_plaintext(
    monkeypatch,
):
    processor = make_processor(allow_embedding_only_ingestion=True)
    callback = RecordingCallback()
    processor.callback_manager.register(callback)
    captured = {}

    async def fake_insert_text_content_embedding_only(text_content, file_ref, doc_id):
        captured["text_content"] = text_content
        captured["file_ref"] = file_ref
        captured["doc_id"] = doc_id

    monkeypatch.setattr(
        processor,
        "_insert_text_content_embedding_only",
        fake_insert_text_content_embedding_only,
    )

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Document title"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Nested paragraph"},
                        {"text": "continued"},
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [{"type": "text", "content": "First item"}],
                        }
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>A</td></tr></table>"},
            },
            {"type": "image", "content": {"image_caption": ["Figure caption"]}},
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/sample_content_list_v2.json",
        doc_id="doc-mineru-v2",
    )

    assert captured["file_ref"] == "sample_content_list_v2.json"
    assert captured["doc_id"] == "doc-mineru-v2"
    assert "Document title" in captured["text_content"]
    assert "Nested paragraph continued" in captured["text_content"]
    assert "1. First item" in captured["text_content"]
    assert "<table><tr><td>A</td></tr></table>" in captured["text_content"]
    assert "Figure caption" in captured["text_content"]
    assert [event[0] for event in callback.events] == ["on_document_complete"]


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_document_complete(
    monkeypatch,
):
    import raganything.processor as processor_module

    processor = make_processor(allow_embedding_only_ingestion=False)
    callback = RecordingCallback()
    processor.callback_manager.register(callback)
    mark_calls = []
    process_multimodal = AsyncMock()
    insert_text = AsyncMock()

    async def fake_mark_multimodal_processing_complete(doc_id):
        mark_calls.append(doc_id)

    monkeypatch.setattr(processor_module, "insert_text_content", insert_text)
    monkeypatch.setattr(
        processor,
        "_mark_multimodal_processing_complete",
        fake_mark_multimodal_processing_complete,
    )
    monkeypatch.setattr(processor, "_process_multimodal_content", process_multimodal)

    await processor.insert_content_list(
        [
            {"type": "text", "text": "text to index"},
            {"type": "image", "img_path": "figure.png"},
        ],
        file_path="/tmp/with_image_content_list_v2.json",
        doc_id="doc-skip-mm",
        skip_multimodal_processing=True,
    )

    insert_text.assert_awaited_once()
    assert insert_text.await_args.kwargs["input"] == "text to index"
    assert (
        insert_text.await_args.kwargs["file_paths"]
        == "with_image_content_list_v2.json"
    )
    assert insert_text.await_args.kwargs["ids"] == "doc-skip-mm"
    assert mark_calls == ["doc-skip-mm"]
    process_multimodal.assert_not_awaited()
    assert [event[0] for event in callback.events] == [
        "on_text_insert_start",
        "on_text_insert_complete",
        "on_document_complete",
    ]
