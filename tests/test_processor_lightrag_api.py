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
async def test_embedding_only_insert_persists_chunks_and_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "tokenizer": FakeTokenizer(),
            "text_chunks": FakeUpsertStorage(),
            "chunks_vdb": FakeUpsertStorage(),
            "doc_status": FakeDocStatusStorage(),
            "_insert_done_calls": 0,
        },
    )()

    async def fake_insert_done():
        processor.lightrag._insert_done_calls += 1

    processor.lightrag._insert_done = fake_insert_done

    await processor._insert_text_content_embedding_only(
        "first paragraph\n\nsecond paragraph",
        "doc.json",
        "doc-abc",
    )

    assert len(processor.lightrag.text_chunks.records) == 2
    assert processor.lightrag.chunks_vdb.records == processor.lightrag.text_chunks.records
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    doc_status = processor.lightrag.doc_status.records["doc-abc"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 2
    assert doc_status["chunks_list"] == list(processor.lightrag.text_chunks.records)
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "doc.json"
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag._insert_done_calls == 1


def test_mineru_v2_blocks_are_flattened_and_recovered_as_plaintext():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    content_list = [
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Nested"},
                        {"type": "text", "content": "paragraph"},
                    ]
                },
            }
        ],
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "1.",
                        "item_content": [{"type": "text", "content": "first item"}],
                    },
                    {"item_content": {"type": "text", "content": "second item"}},
                ]
            },
        },
        {"type": "table", "content": {"html": "<table><tr><td>A</td></tr></table>"}},
        {"type": "text", "text": "tail text"},
    ]

    normalized = processor._normalize_nested_content_list(content_list)
    plaintext = processor._plaintext_from_mineru_blocks(normalized)

    assert len(normalized) == 4
    assert "Nested paragraph" in plaintext
    assert "1. first item" in plaintext
    assert "second item" in plaintext
    assert "<table><tr><td>A</td></tr></table>" in plaintext
    assert "tail text" in plaintext


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_uses_mineru_plaintext_and_callback():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "display_content_stats": False,
            "use_full_path": False,
            "allow_embedding_only_ingestion": True,
            "content_format": "mineru",
        },
    )()
    events = []
    inserted = {}
    processor.callback_manager = type(
        "CallbackManager",
        (),
        {"dispatch": lambda self, event_name, **kwargs: events.append(event_name)},
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_insert_embedding_only(text_content, file_ref, doc_id):
        inserted["text_content"] = text_content
        inserted["file_ref"] = file_ref
        inserted["doc_id"] = doc_id

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor._insert_text_content_embedding_only = fake_insert_embedding_only

    await processor.insert_content_list(
        [
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Recovered text"}
                        ]
                    },
                }
            ]
        ],
        file_path="/tmp/mineru_content_list_v2.json",
        doc_id="doc-v2",
    )

    assert inserted == {
        "text_content": "Recovered text",
        "file_ref": "mineru_content_list_v2.json",
        "doc_id": "doc-v2",
    }
    assert "on_document_complete" in events
