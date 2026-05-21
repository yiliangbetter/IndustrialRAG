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


class FakeKVStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

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
        self.text_chunks = FakeKVStorage()
        self.chunks_vdb = FakeKVStorage()
        self.doc_status = FakeKVStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class RecordingDocumentCallback(ProcessingCallback):
    def __init__(self):
        self.completed = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.completed.append((file_path, doc_id))


@pytest.mark.asyncio
async def test_embedding_only_content_list_stores_recovered_mineru_text_and_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "content_format": "minerU",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Storage"},
                        {"type": "text", "content": "Overview"},
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "First paragraph."}
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
                        "item_content": [{"type": "text", "content": "Persist chunks"}],
                    },
                    {
                        "prefix": "2.",
                        "item_content": [{"type": "text", "content": "Mark status"}],
                    },
                ]
            },
        },
        {
            "type": "table",
            "content": {"html": "<table><tr><td>Total</td></tr></table>"},
        },
        {
            "type": "image",
            "content": {"image_caption": ["Figure 1 storage diagram"]},
        },
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/source/doc_content_list_v2.json",
        doc_id="doc-storage",
    )

    chunks = processor.lightrag.text_chunks.records
    assert list(chunks.values()) == [
        {
            "content": "Storage Overview",
            "tokens": 2,
            "full_doc_id": "doc-storage",
            "chunk_order_index": 0,
            "file_path": "doc_content_list_v2.json",
            "llm_cache_list": [],
        },
        {
            "content": "First paragraph.",
            "tokens": 2,
            "full_doc_id": "doc-storage",
            "chunk_order_index": 1,
            "file_path": "doc_content_list_v2.json",
            "llm_cache_list": [],
        },
        {
            "content": "1. Persist chunks\n2. Mark status",
            "tokens": 6,
            "full_doc_id": "doc-storage",
            "chunk_order_index": 2,
            "file_path": "doc_content_list_v2.json",
            "llm_cache_list": [],
        },
        {
            "content": "<table><tr><td>Total</td></tr></table>",
            "tokens": 1,
            "full_doc_id": "doc-storage",
            "chunk_order_index": 3,
            "file_path": "doc_content_list_v2.json",
            "llm_cache_list": [],
        },
        {
            "content": "Figure 1 storage diagram",
            "tokens": 4,
            "full_doc_id": "doc-storage",
            "chunk_order_index": 4,
            "file_path": "doc_content_list_v2.json",
            "llm_cache_list": [],
        },
    ]
    assert processor.lightrag.chunks_vdb.records == chunks
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    doc_status = processor.lightrag.doc_status.records["doc-storage"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 5
    assert doc_status["chunks_list"] == list(chunks.keys())
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "doc_content_list_v2.json"
    assert "Storage Overview" in doc_status["content"]
    assert "Figure 1 storage diagram" in doc_status["content"]
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
    assert callback.completed == [
        ("/tmp/source/doc_content_list_v2.json", "doc-storage")
    ]
