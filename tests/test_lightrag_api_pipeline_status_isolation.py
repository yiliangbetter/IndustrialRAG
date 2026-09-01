"""LightRAG API must re-enable pipeline scan even when status updates fail.

Inner parse errors return False without hitting the outer except, so only
the finally block clears scan_disabled. If that update raises, the original
True/False result must still be delivered — otherwise a status-store blip
looks like ingest failure (or leaves scanners blocked).
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


class FailAfterLock:
    """Succeeds `succeed_times` acquires, then raises — models a flaky status store."""

    def __init__(self, succeed_times: int = 1):
        self.succeed_times = succeed_times
        self.acquires = 0

    async def __aenter__(self):
        self.acquires += 1
        if self.acquires > self.succeed_times:
            raise RuntimeError("pipeline lock failed")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


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
            "content_format": "minerU",
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    return processor


def _patch_pipeline_status(monkeypatch, pipeline=None, lock=None):
    from lightrag.kg import shared_storage

    if pipeline is None:
        pipeline = {"scan_disabled": False, "history_messages": []}
    if lock is None:
        lock = FakeLock()

    async def fake_get_namespace_data(name):
        assert name == "pipeline_status"
        return pipeline

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_pipeline_status_lock", lambda: lock)
    return pipeline


@pytest.mark.asyncio
async def test_parse_error_re_enables_pipeline_scan_via_finally(monkeypatch):
    pytest.importorskip("lightrag")

    pipeline = _patch_pipeline_status(monkeypatch)
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        raise MineruExecutionError(1, ["cuda oom"])

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse

    result = await processor.process_document_complete_lightrag_api("plant/manual.pdf")

    assert result is False
    assert pipeline["scan_disabled"] is False
    assert any("Now is allowed to scan" in msg for msg in pipeline["history_messages"])
    status = processor.lightrag.doc_status.records["doc-pre-manual.pdf"]
    assert status["status"] == DocStatus.FAILED


@pytest.mark.asyncio
async def test_finally_pipeline_error_does_not_mask_successful_return(
    monkeypatch, tmp_path
):
    pytest.importorskip("lightrag")

    import raganything.processor as processor_module

    pipeline = {"scan_disabled": False, "history_messages": []}
    _patch_pipeline_status(
        monkeypatch, pipeline=pipeline, lock=FailAfterLock(succeed_times=1)
    )
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "intro", "page_idx": 0}], "doc-ok")

    async def fake_insert(*args, **kwargs):
        return None

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module, "insert_text_content_with_multimodal_content", fake_insert
    )

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = await processor.process_document_complete_lightrag_api(str(pdf))

    assert result is True


@pytest.mark.asyncio
async def test_finally_pipeline_error_does_not_mask_parse_failure(monkeypatch):
    pytest.importorskip("lightrag")

    _patch_pipeline_status(monkeypatch, lock=FailAfterLock(succeed_times=1))
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
async def test_outer_except_pipeline_error_still_returns_false(monkeypatch):
    pytest.importorskip("lightrag")

    import raganything.processor as processor_module

    pipeline = {"scan_disabled": False, "history_messages": []}
    _patch_pipeline_status(
        monkeypatch, pipeline=pipeline, lock=FailAfterLock(succeed_times=1)
    )
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "intro", "page_idx": 0}], "doc-ok")

    async def fake_insert(*args, **kwargs):
        raise RuntimeError("vector upsert failed")

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module, "insert_text_content_with_multimodal_content", fake_insert
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "vector upsert failed"


@pytest.mark.asyncio
async def test_lightrag_api_sets_content_source_for_multimodal_items(monkeypatch):
    pytest.importorskip("lightrag")

    import raganything.processor as processor_module

    _patch_pipeline_status(monkeypatch)
    processor = _make_processor()
    captured = {}

    def fake_set_source(content_list, content_format="auto"):
        captured["content_list"] = content_list
        captured["content_format"] = content_format

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return (
            [
                {"type": "text", "text": "intro", "page_idx": 0},
                {"type": "image", "img_path": "/abs/fig.png", "page_idx": 1},
            ],
            "doc-mm",
        )

    async def fake_insert(*args, **kwargs):
        return None

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    processor.set_content_source_for_context = fake_set_source
    monkeypatch.setattr(
        processor_module, "insert_text_content_with_multimodal_content", fake_insert
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert captured["content_format"] == "minerU"
    assert any(item.get("type") == "image" for item in captured["content_list"])
