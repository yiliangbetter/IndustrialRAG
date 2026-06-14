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


class FakeDocStatusStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


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
        self.doc_status = FakeDocStatusStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.completed = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.completed.append((file_path, doc_id))


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_persists_failed_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing llm_model_func"
    assert doc_status["file_path"] == "sample.pdf"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_persists_chunks_and_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "display_content_stats": False,
            "use_full_path": False,
            "content_format": "minerU",
        },
    )()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()
    callback = RecordingCallback()
    processor.callback_manager.register(callback)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fail_if_multimodal_processed(*args, **kwargs):
        raise AssertionError("embedding-only ingestion should skip multimodal processors")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor._process_multimodal_content = fail_if_multimodal_processed

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Storage Title"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "First paragraph"}
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
                        "item_content": [{"type": "text", "content": "Key point"}],
                    }
                ]
            },
        },
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/uploaded/source_content_list_v2.json",
        doc_id="doc-embedding-only",
    )

    assert len(processor.lightrag.text_chunks.records) == 3
    assert processor.lightrag.text_chunks.records == processor.lightrag.chunks_vdb.records
    assert [
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    ] == ["Storage Title", "First paragraph", "1. Key point"]
    assert {
        chunk["file_path"] for chunk in processor.lightrag.text_chunks.records.values()
    } == {"source_content_list_v2.json"}
    assert [
        chunk["chunk_order_index"]
        for chunk in processor.lightrag.text_chunks.records.values()
    ] == [0, 1, 2]
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    doc_status = processor.lightrag.doc_status.records["doc-embedding-only"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 3
    assert doc_status["chunks_list"] == list(processor.lightrag.text_chunks.records)
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "source_content_list_v2.json"
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
    assert callback.completed == [
        ("/tmp/uploaded/source_content_list_v2.json", "doc-embedding-only")
    ]
