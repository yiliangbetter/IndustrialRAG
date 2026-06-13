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


class FakeLightRAG:
    def __init__(self):
        self.text_chunks = FakeDocStatusStorage()
        self.chunks_vdb = FakeDocStatusStorage()
        self.doc_status = FakeDocStatusStorage()
        self.insert_done_calls = 0
        self.tokenizer = type(
            "Tokenizer",
            (),
            {"encode": staticmethod(lambda text: text.split())},
        )()

    async def _insert_done(self):
        self.insert_done_calls += 1


class FakeCallbackManager:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


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
async def test_insert_content_list_embedding_only_indexes_mineru_v2_plaintext():
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
    processor.callback_manager = FakeCallbackManager()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Intro"},
                        {"content": [{"type": "text", "content": "Overview"}]},
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "First sentence"},
                        {"content": [{"type": "text", "content": "second sentence"}]},
                    ]
                },
            },
        ],
        "ignored top-level noise",
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "1.",
                        "item_content": [{"type": "text", "content": "First item"}],
                    },
                    {"item_content": {"text": "Second item"}},
                ]
            },
        },
        {
            "type": "table",
            "content": {"html": " <table><tr><td>A</td></tr></table> "},
        },
        {"type": "image", "content": {"image_caption": ["figure caption"]}},
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/source_content_list_v2.json",
        doc_id="doc-mineru-v2",
    )

    chunks = sorted(
        processor.lightrag.text_chunks.records.values(),
        key=lambda chunk: chunk["chunk_order_index"],
    )
    assert [chunk["content"] for chunk in chunks] == [
        "Intro Overview",
        "First sentence second sentence",
        "1. First item\nSecond item",
        "<table><tr><td>A</td></tr></table>",
        "figure caption",
    ]
    assert (
        processor.lightrag.chunks_vdb.records
        == processor.lightrag.text_chunks.records
    )
    assert all(chunk["full_doc_id"] == "doc-mineru-v2" for chunk in chunks)
    assert all(chunk["file_path"] == "source_content_list_v2.json" for chunk in chunks)
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    doc_status = processor.lightrag.doc_status.records["doc-mineru-v2"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 5
    assert doc_status["chunks_list"] == list(processor.lightrag.text_chunks.records)
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "source_content_list_v2.json"
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
    assert processor.callback_manager.events[0][0] == "on_document_complete"
    assert processor.callback_manager.events[0][1]["doc_id"] == "doc-mineru-v2"


@pytest.mark.asyncio
async def test_embedding_only_empty_text_marks_multimodal_complete_without_chunks():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.lightrag = FakeLightRAG()
    completed_doc_ids = []

    async def fake_mark_multimodal_processing_complete(doc_id):
        completed_doc_ids.append(doc_id)

    processor._mark_multimodal_processing_complete = (
        fake_mark_multimodal_processing_complete
    )

    await processor._insert_text_content_embedding_only(
        text_content=" \n ",
        file_ref="empty.pdf",
        doc_id="doc-empty",
    )

    assert completed_doc_ids == ["doc-empty"]
    assert processor.lightrag.text_chunks.records == {}
    assert processor.lightrag.chunks_vdb.records == {}
    assert processor.lightrag.doc_status.records == {}
    assert processor.lightrag.insert_done_calls == 0
