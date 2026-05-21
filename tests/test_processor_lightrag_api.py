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
        self.upsert_calls = []

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upsert_calls.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeKeyValueStorage(FakeDocStatusStorage):
    pass


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
async def test_embedding_only_content_list_recovers_mineru_v2_text_and_status(tmp_path):
    class DummyProcessor(ProcessorMixin):
        pass

    class FakeLightRAG:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.text_chunks = FakeKeyValueStorage()
            self.chunks_vdb = FakeKeyValueStorage()
            self.doc_status = FakeDocStatusStorage()
            self.insert_done_calls = 0

        async def _insert_done(self):
            self.insert_done_calls += 1

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "display_content_stats": False,
            "allow_embedding_only_ingestion": True,
            "content_format": "minerU",
        },
    )()
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    source_path = tmp_path / "reports" / "sample_content_list_v2.json"
    source_path.parent.mkdir()
    source_path.write_text("[]")

    await processor.insert_content_list(
        [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "Risk Report"}]
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Revenue grew"},
                            {"type": "text", "content": "10%"},
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
                            },
                            {
                                "prefix": "2.",
                                "item_content": {
                                    "type": "text",
                                    "content": "Second control",
                                },
                            },
                        ]
                    },
                },
                {
                    "type": "table",
                    "content": {"html": "<table><tr><td>A</td></tr></table>"},
                },
                {"type": "image", "content": {"image_caption": ["Process diagram"]}},
            ]
        ],
        file_path=str(source_path),
        doc_id="doc-fixed",
    )

    chunk_contents = [
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    ]
    assert chunk_contents == [
        "Risk Report",
        "Revenue grew 10%",
        "1. First control\n2. Second control",
        "<table><tr><td>A</td></tr></table>",
        "Process diagram",
    ]
    assert (
        processor.lightrag.chunks_vdb.records == processor.lightrag.text_chunks.records
    )
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1

    doc_status = processor.lightrag.doc_status.records["doc-fixed"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["file_path"] == "sample_content_list_v2.json"
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 5
    assert doc_status["chunks_list"] == list(processor.lightrag.text_chunks.records)
