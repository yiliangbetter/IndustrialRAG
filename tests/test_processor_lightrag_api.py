"""Regression tests for ProcessorMixin.process_document_complete_lightrag_api.

These assert current main-branch contracts (status transitions, parse failure
persistence, insert forwarding, and pipeline_status scan re-enable). They do
not depend on open critical-bug production fixes that change empty-extraction
or insert soft-fail semantics.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import raganything.processor as processor_module
from raganything.base import DocStatus
from raganything.parser import MineruExecutionError
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


def _install_pipeline_status(monkeypatch):
    pipeline_status = {"history_messages": [], "scan_disabled": False}
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
    return pipeline_status


def _make_lightrag_api_processor(monkeypatch, content_list):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        use_full_path=False,
        parser="mineru",
        parser_output_dir="output",
        parse_method="auto",
        display_content_stats=False,
        content_format="minerU",
    )
    processor.lightrag = SimpleNamespace(doc_status=FakeDocStatusStorage())
    processor.content_source_calls = []

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return content_list, "doc-content"

    def fake_set_content_source_for_context(content_list_arg, content_format):
        processor.content_source_calls.append((content_list_arg, content_format))

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    processor.set_content_source_for_context = fake_set_content_source_for_context
    pipeline_status = _install_pipeline_status(monkeypatch)
    return processor, pipeline_status


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
async def test_lightrag_api_init_failure_without_doc_status_returns_false():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(use_full_path=False, parser="mineru")
    processor.lightrag = None

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "not constructed"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("orphan.pdf")
    assert result is False


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_updates_existing_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    storage = FakeDocStatusStorage()
    storage.records["doc-pre-existing.pdf"] = {
        "status": DocStatus.READY,
        "content": "keep-me",
        "error_msg": "",
        "content_summary": "summary",
        "multimodal_content": [{"type": "image"}],
        "scheme_name": "scheme-a",
        "content_length": 12,
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
        "file_path": "existing.pdf",
        "extra_field": "preserved",
    }

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(use_full_path=False, parser="mineru")
    processor.lightrag = SimpleNamespace(doc_status=storage)

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "llm missing"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("existing.pdf")

    assert result is False
    doc_status = storage.records["doc-pre-existing.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "llm missing"
    assert doc_status["content"] == "keep-me"
    assert doc_status["extra_field"] == "preserved"
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_parse_failure_persists_failed_doc_status(monkeypatch):
    processor, pipeline_status = _make_lightrag_api_processor(
        monkeypatch, [{"type": "text", "text": "x"}]
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("parse exploded")

    processor.parse_document = boom

    result = await processor.process_document_complete_lightrag_api("broken.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-broken.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert "parse exploded" in doc_status["error_msg"]
    # finally block must re-enable scanning even after parse failure
    assert pipeline_status["scan_disabled"] is False


@pytest.mark.asyncio
async def test_lightrag_api_mineru_execution_error_joins_list_messages(monkeypatch):
    processor, _pipeline_status = _make_lightrag_api_processor(
        monkeypatch, [{"type": "text", "text": "x"}]
    )

    async def boom(*args, **kwargs):
        raise MineruExecutionError(2, ["stderr line 1", "stderr line 2"])

    processor.parse_document = boom

    result = await processor.process_document_complete_lightrag_api("mineru-fail.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-mineru-fail.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "stderr line 1\nstderr line 2"


@pytest.mark.asyncio
async def test_lightrag_api_success_inserts_and_reenables_pipeline(monkeypatch):
    content_list = [
        {"type": "text", "text": "indexed body", "page_idx": 0},
        {
            "type": "image",
            "img_path": "/tmp/fig.png",
            "image_caption": ["Fig 1"],
            "page_idx": 1,
        },
    ]
    processor, pipeline_status = _make_lightrag_api_processor(monkeypatch, content_list)
    insert_mock = AsyncMock()
    monkeypatch.setattr(
        processor_module, "insert_text_content_with_multimodal_content", insert_mock
    )

    result = await processor.process_document_complete_lightrag_api(
        "/data/ok.pdf",
        doc_id="doc-explicit",
        split_by_character="\n",
        split_by_character_only=True,
        scheme_name="scheme-x",
    )

    assert result is True
    insert_mock.assert_awaited_once()
    kwargs = insert_mock.await_args.kwargs
    assert kwargs["input"] == "indexed body"
    assert kwargs["ids"] == "doc-explicit"
    assert kwargs["file_paths"] == "ok.pdf"
    assert kwargs["split_by_character"] == "\n"
    assert kwargs["split_by_character_only"] is True
    assert kwargs["scheme_name"] == "scheme-x"
    assert len(kwargs["multimodal_content"]) == 1
    assert kwargs["multimodal_content"][0]["type"] == "image"
    assert processor.content_source_calls == [(content_list, "minerU")]
    assert pipeline_status["scan_disabled"] is False
    assert any(
        "processing completed" in msg for msg in pipeline_status["history_messages"]
    )


@pytest.mark.asyncio
async def test_lightrag_api_parser_kwarg_updates_config(monkeypatch):
    processor, _pipeline_status = _make_lightrag_api_processor(
        monkeypatch, [{"type": "text", "text": "body", "page_idx": 0}]
    )
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        AsyncMock(),
    )

    result = await processor.process_document_complete_lightrag_api(
        "switch.pdf", parser="docling"
    )

    assert result is True
    assert processor.config.parser == "docling"
