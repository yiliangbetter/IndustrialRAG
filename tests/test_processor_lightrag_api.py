import asyncio
from types import SimpleNamespace

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


class FakeChunkStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeEmbeddingOnlyLightRAG:
    def __init__(self):
        self.text_chunks = FakeChunkStorage()
        self.chunks_vdb = FakeChunkStorage()
        self.doc_status = FakeDocStatusStorage()
        self.insert_done_calls = 0
        self.tokenizer = SimpleNamespace(encode=lambda text: text.split())

    async def _insert_done(self):
        self.insert_done_calls += 1


def _install_pipeline_status(monkeypatch):
    pipeline_status = {"history_messages": []}
    pipeline_status_lock = asyncio.Lock()

    async def fake_get_namespace_data(namespace):
        assert namespace == "pipeline_status"
        return pipeline_status

    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_namespace_data",
        fake_get_namespace_data,
    )
    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_pipeline_status_lock",
        lambda: pipeline_status_lock,
    )


def _make_lightrag_api_processor(monkeypatch, content_list, ainsert_side_effect=None):
    class DummyProcessor(ProcessorMixin):
        pass

    class FakeLightRAG:
        def __init__(self):
            self.doc_status = FakeDocStatusStorage()
            self.ainsert_calls = []

        async def ainsert(self, **kwargs):
            self.ainsert_calls.append(kwargs)
            if ainsert_side_effect is not None:
                raise ainsert_side_effect

    _install_pipeline_status(monkeypatch)

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        use_full_path=False,
        parser="mineru",
        parser_output_dir=None,
        parse_method="auto",
        display_content_stats=False,
        content_format="mineru",
    )
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(
        file_path, output_dir, parse_method, display_stats, **kwargs
    ):
        return content_list, "doc-content"

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    return processor


def _make_embedding_only_processor(content_list):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        use_full_path=False,
        parser="mineru",
        parser_output_dir=None,
        parse_method="auto",
        display_content_stats=False,
        allow_embedding_only_ingestion=True,
        content_format="mineru",
    )
    processor.lightrag = FakeEmbeddingOnlyLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(
        file_path, output_dir, parse_method, display_stats, **kwargs
    ):
        return content_list, "doc-content"

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
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
async def test_lightrag_api_success_marks_pre_doc_processed(monkeypatch):
    processor = _make_lightrag_api_processor(
        monkeypatch,
        [{"type": "text", "text": "hello world"}],
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert processor.lightrag.ainsert_calls
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["error_msg"] == ""
    assert doc_status["content"] == "hello world"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_insert_failure_persists_failed_status(monkeypatch):
    processor = _make_lightrag_api_processor(
        monkeypatch,
        [{"type": "text", "text": "hello world"}],
        ainsert_side_effect=RuntimeError("storage down"),
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "storage down"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_empty_extraction_fails_instead_of_succeeding(monkeypatch):
    processor = _make_lightrag_api_processor(
        monkeypatch,
        [{"type": "image", "content": {}}],
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    assert processor.lightrag.ainsert_calls == []
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert "No text content extracted" in doc_status["error_msg"]


@pytest.mark.asyncio
async def test_embedding_only_empty_content_list_raises():
    processor = _make_embedding_only_processor([])

    with pytest.raises(ValueError, match="No text content extracted"):
        await processor.insert_content_list([], file_path="empty_content_list_v2.json")


@pytest.mark.asyncio
async def test_embedding_only_document_recovers_mineru_v2_text():
    processor = _make_embedding_only_processor(
        [
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Recovered paragraph"}
                        ]
                    },
                }
            ]
        ]
    )

    await processor.process_document_complete("sample.pdf")

    assert processor.lightrag.insert_done_calls == 1
    assert [
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    ] == ["Recovered paragraph"]
    doc_status = processor.lightrag.doc_status.records["doc-content"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 1
