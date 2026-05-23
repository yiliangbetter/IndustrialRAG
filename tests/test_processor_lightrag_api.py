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


class FakeEmbeddingOnlyLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeUpsertStorage()
        self.chunks_vdb = FakeUpsertStorage()
        self.doc_status = FakeDocStatusStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


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
async def test_embedding_only_content_list_recovers_nested_mineru_v2_text(tmp_path):
    class DummyProcessor(ProcessorMixin):
        async def _ensure_lightrag_initialized(self):
            return {"success": True}

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
    processor.lightrag = FakeEmbeddingOnlyLightRAG()

    source_path = tmp_path / "report_content_list_v2.json"
    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "System Overview"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pump status"},
                        {"type": "text", "content": "nominal"},
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
                                {"type": "text", "content": "Check pressure"}
                            ],
                        },
                        {"prefix": "2.", "item_content": "Log reading"},
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>Pressure</td></tr></table>"},
            },
            {"type": "image", "content": {"image_caption": ["Gauge photo"]}},
        ]
    ]

    await processor.insert_content_list(content_list, file_path=str(source_path))

    chunks = processor.lightrag.text_chunks.records
    assert processor.lightrag.chunks_vdb.records == chunks
    assert [
        chunk["content"]
        for chunk in sorted(chunks.values(), key=lambda data: data["chunk_order_index"])
    ] == [
        "System Overview",
        "Pump status nominal",
        "1. Check pressure\n2. Log reading",
        "<table><tr><td>Pressure</td></tr></table>",
        "Gauge photo",
    ]

    assert len(processor.lightrag.doc_status.records) == 1
    doc_id, status = next(iter(processor.lightrag.doc_status.records.items()))
    assert doc_id.startswith("doc-")
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 5
    assert status["chunks_list"] == list(chunks)
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "report_content_list_v2.json"
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
