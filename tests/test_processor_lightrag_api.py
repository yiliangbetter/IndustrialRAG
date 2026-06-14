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


class FakeLightRAGForContentList:
    def __init__(self):
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeDocStatusStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0
        self.ainsert = AsyncMock()

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
async def test_insert_content_list_embedding_only_recovers_mineru_v2_text():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAGForContentList()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "display_content_stats": False,
            "use_full_path": False,
            "content_format": "mineru",
        },
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Safety Report"}
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pressure rising"},
                        {"type": "text", "content": "shutdown required"},
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
                                {"type": "text", "content": "Open relief valve"}
                            ],
                        }
                    ]
                },
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/uploads/doc_content_list_v2.json",
        doc_id="doc-risk",
    )

    chunks = sorted(
        processor.lightrag.text_chunks.records.values(),
        key=lambda chunk: chunk["chunk_order_index"],
    )
    assert [chunk["content"] for chunk in chunks] == [
        "Safety Report",
        "Pressure rising shutdown required",
        "1. Open relief valve",
    ]
    assert processor.lightrag.chunks_vdb.records == (
        processor.lightrag.text_chunks.records
    )
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    status = processor.lightrag.doc_status.records["doc-risk"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["file_path"] == "doc_content_list_v2.json"
    assert status["chunks_count"] == 3
    assert status["chunks_list"] == list(processor.lightrag.text_chunks.records.keys())
    assert status["multimodal_processed"] is True
    processor.lightrag.ainsert.assert_not_awaited()
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_recovered_text_complete(
    monkeypatch,
):
    import raganything.processor as processor_module

    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAGForContentList()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": False,
            "display_content_stats": False,
            "use_full_path": False,
            "content_format": "mineru",
        },
    )()
    processor.lightrag.doc_status.records["doc-mixed"] = {
        "status": DocStatus.PROCESSED,
        "chunks_list": ["text-chunk"],
        "chunks_count": 1,
    }
    processor._process_multimodal_content = AsyncMock()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    captured_insert = {}

    async def fake_insert_text_content(lightrag, **kwargs):
        captured_insert.update(kwargs)

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    content_list = [
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pump trip"},
                    ]
                },
            },
            {
                "type": "image",
                "content": {"image_caption": ["High pressure alarm"]},
                "img_path": "/tmp/alarm.png",
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/uploads/risk.json",
        doc_id="doc-mixed",
        skip_multimodal_processing=True,
    )

    assert captured_insert == {
        "input": "Pump trip\n\nHigh pressure alarm",
        "file_paths": "risk.json",
        "split_by_character": None,
        "split_by_character_only": False,
        "ids": "doc-mixed",
    }
    processor._process_multimodal_content.assert_not_awaited()
    assert (
        processor.lightrag.doc_status.records["doc-mixed"]["multimodal_processed"]
        is True
    )
    assert processor.lightrag.doc_status.index_done_calls == 1
