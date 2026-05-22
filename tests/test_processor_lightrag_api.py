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


class FakeUpsertStorage:
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
        self.text_chunks = FakeUpsertStorage()
        self.chunks_vdb = FakeUpsertStorage()
        self.doc_status = FakeDocStatusStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


def make_embedding_only_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "content_format": "mineru",
            "display_content_stats": False,
            "parser": "mineru",
            "use_full_path": False,
        },
    )()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = None

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


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
async def test_insert_content_list_embedding_only_recovers_mineru_v2_plaintext():
    processor = make_embedding_only_processor()
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
                        {"type": "text", "content": "Lockout"},
                        {"type": "text", "content": "procedure"},
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
                                {"type": "text", "content": "Shut down equipment"}
                            ],
                        }
                    ]
                },
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/safety_content_list_v2.json",
    )

    chunk_texts = [
        chunk["content"]
        for chunk in sorted(
            processor.lightrag.text_chunks.records.values(),
            key=lambda chunk: chunk["chunk_order_index"],
        )
    ]
    assert chunk_texts == [
        "Safety Manual",
        "Lockout procedure",
        "1. Shut down equipment",
    ]
    assert (
        processor.lightrag.chunks_vdb.records
        == processor.lightrag.text_chunks.records
    )

    assert len(processor.lightrag.doc_status.records) == 1
    doc_id, status = next(iter(processor.lightrag.doc_status.records.items()))
    assert doc_id.startswith("doc-")
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 3
    assert status["chunks_list"] == list(processor.lightrag.text_chunks.records.keys())
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "safety_content_list_v2.json"
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_embedding_only_empty_text_marks_multimodal_complete_without_upserts():
    processor = make_embedding_only_processor()
    completed_doc_ids = []

    async def fake_mark_multimodal_processing_complete(doc_id):
        completed_doc_ids.append(doc_id)

    processor._mark_multimodal_processing_complete = (
        fake_mark_multimodal_processing_complete
    )

    await processor._insert_text_content_embedding_only(
        text_content=" \n\t ",
        file_ref="empty.txt",
        doc_id="doc-empty",
    )

    assert completed_doc_ids == ["doc-empty"]
    assert processor.lightrag.text_chunks.records == {}
    assert processor.lightrag.chunks_vdb.records == {}
    assert processor.lightrag.doc_status.records == {}
    assert processor.lightrag.insert_done_calls == 0
