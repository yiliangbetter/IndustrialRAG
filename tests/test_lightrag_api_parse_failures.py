"""Regression tests for process_document_complete_lightrag_api parse/status.

Only the init-failure branch is covered on main. Parse errors must persist
FAILED on doc-pre-* and return False so scanners do not treat the file as
still HANDLING or successfully indexed.
"""

import pytest

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


class FakeLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _patch_pipeline_status(monkeypatch):
    from lightrag.kg import shared_storage

    pipeline = {"history_messages": []}

    async def fake_get_namespace_data(name):
        return pipeline

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_pipeline_status_lock", lambda: FakeLock())
    return pipeline


def _make_processor():
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
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    return processor


@pytest.mark.asyncio
async def test_mineru_execution_error_persists_joined_failed_status(monkeypatch):
    _patch_pipeline_status(monkeypatch)
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        raise MineruExecutionError(1, ["cuda oom", "page 3 failed"])

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse

    result = await processor.process_document_complete_lightrag_api("plant/manual.pdf")

    assert result is False
    status = processor.lightrag.doc_status.records["doc-pre-manual.pdf"]
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "cuda oom\npage 3 failed"
    assert status["file_path"] == "manual.pdf"


@pytest.mark.asyncio
async def test_generic_parse_error_persists_failed_status(monkeypatch):
    _patch_pipeline_status(monkeypatch)
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        raise ValueError("empty extract")

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "empty extract"


@pytest.mark.asyncio
async def test_successful_parse_forwards_text_and_multimodal_to_ainsert(monkeypatch):
    import raganything.processor as processor_module

    _patch_pipeline_status(monkeypatch)
    processor = _make_processor()
    captured = {}

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return (
            [
                {"type": "text", "text": "intro", "page_idx": 0},
                {"type": "table", "table_body": "a|b", "page_idx": 1},
            ],
            "doc-content",
        )

    async def fake_insert(lightrag, input=None, **kwargs):
        captured["input"] = input
        captured.update(kwargs)

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert,
    )

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf", doc_id="doc-content", scheme_name="plant-a"
    )

    assert result is True
    assert captured["input"] == "intro"
    assert captured["multimodal_content"] == [
        {"type": "table", "table_body": "a|b", "page_idx": 1}
    ]
    assert captured["file_paths"] == "sample.pdf"
    assert captured["ids"] == "doc-content"
    assert captured["scheme_name"] == "plant-a"


@pytest.mark.asyncio
async def test_post_parse_exception_marks_failed_and_returns_false(monkeypatch):
    import raganything.processor as processor_module

    pipeline = _patch_pipeline_status(monkeypatch)
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "intro", "page_idx": 0}], "doc-content")

    async def fake_insert(lightrag, input=None, **kwargs):
        raise RuntimeError("vector upsert failed")

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "vector upsert failed"
    assert processor.lightrag.doc_status.index_done_calls >= 1
    assert pipeline.get("scan_disabled") is False
