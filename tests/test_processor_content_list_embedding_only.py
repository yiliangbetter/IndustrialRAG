import pytest

from raganything.base import DocStatus
from raganything.callbacks import CallbackManager, ProcessingCallback
from raganything.config import RAGAnythingConfig
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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.document_complete_events = []

    def on_document_complete(self, file_path, doc_id="", **kwargs):
        self.document_complete_events.append((file_path, doc_id))


class DummyProcessor(ProcessorMixin):
    pass


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_ingests_mineru_v2_plaintext():
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = RAGAnythingConfig(
        allow_embedding_only_ingestion=True,
        display_content_stats=False,
        use_full_path=False,
    )
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()
    callback = RecordingCallback()
    processor.callback_manager.register(callback)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Quarterly report"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Revenue"},
                        {"type": "text", "content": "grew."},
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
                                {"type": "text", "content": "First control"}
                            ],
                        }
                    ]
                },
            },
        ]
    ]
    normalized_blocks = content_list[0]
    expected_doc_id = processor._generate_content_based_doc_id(normalized_blocks)

    await processor.insert_content_list(
        content_list,
        file_path="/docs/report_content_list_v2.json",
        display_stats=False,
    )

    assert set(processor.lightrag.text_chunks.records) == set(
        processor.lightrag.chunks_vdb.records
    )
    chunk_texts = {
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    }
    assert chunk_texts == {"Quarterly report", "Revenue grew.", "1. First control"}

    doc_status = processor.lightrag.doc_status.records[expected_doc_id]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 3
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "report_content_list_v2.json"
    assert "Revenue grew." in doc_status["content"]
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
    assert callback.document_complete_events == [
        ("/docs/report_content_list_v2.json", expected_doc_id)
    ]
