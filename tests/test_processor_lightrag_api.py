import asyncio
import sys
import types

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
    def __init__(self, ainsert_error: Exception | None = None):
        self.doc_status = FakeDocStatusStorage()
        self.ainsert_error = ainsert_error
        self.ainsert_calls = []

    async def ainsert(self, **kwargs):
        self.ainsert_calls.append(kwargs)
        if self.ainsert_error is not None:
            raise self.ainsert_error


def install_fake_shared_storage(monkeypatch):
    pipeline_status = {"history_messages": []}
    pipeline_status_lock = asyncio.Lock()

    async def get_namespace_data(namespace):
        assert namespace == "pipeline_status"
        return pipeline_status

    def get_pipeline_status_lock():
        return pipeline_status_lock

    lightrag_module = types.ModuleType("lightrag")
    kg_module = types.ModuleType("lightrag.kg")
    shared_storage_module = types.ModuleType("lightrag.kg.shared_storage")
    shared_storage_module.get_namespace_data = get_namespace_data
    shared_storage_module.get_pipeline_status_lock = get_pipeline_status_lock

    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setitem(sys.modules, "lightrag.kg", kg_module)
    monkeypatch.setitem(
        sys.modules, "lightrag.kg.shared_storage", shared_storage_module
    )


def make_processor(lightrag):
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
            "parser_output_dir": "./output",
            "parse_method": "auto",
            "display_content_stats": False,
            "content_format": "mineru",
        },
    )()
    processor.lightrag = lightrag
    return processor


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_persists_failed_doc_status():
    processor = make_processor(FakeLightRAG())

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
async def test_lightrag_api_ainsert_failure_marks_document_failed(monkeypatch):
    install_fake_shared_storage(monkeypatch)
    processor = make_processor(FakeLightRAG(RuntimeError("storage write failed")))

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return [{"type": "text", "text": "important document text"}], "doc-content"

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "storage write failed"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_recovers_mineru_v2_paragraph_text(monkeypatch):
    install_fake_shared_storage(monkeypatch)
    processor = make_processor(FakeLightRAG())

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Recovered MinerU v2 text"}
                    ]
                },
            }
        ], "doc-content"

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert processor.lightrag.ainsert_calls
    assert processor.lightrag.ainsert_calls[0]["input"] == "Recovered MinerU v2 text"
