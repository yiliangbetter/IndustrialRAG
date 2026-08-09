"""Outer-except path for process_document_complete_lightrag_api.

Parse failures are handled by an inner try/except (#103). Post-parse failures
(e.g. content separation) enter the outer except, which must persist FAILED
status, call index_done_callback, record a pipeline error message, and still
re-enable scanning in finally. Kept as a separate module to avoid conflicting
with open expansions of tests/test_processor_lightrag_api.py.
"""

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


@pytest.mark.asyncio
async def test_lightrag_api_outer_except_persists_failed_and_logs_pipeline(
    monkeypatch,
):
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

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return (
            [{"type": "text", "text": "hello"}, {"type": "image", "img_path": "/a.png"}],
            "doc-content",
        )

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    pipeline_status = _install_pipeline_status(monkeypatch)

    def boom_separate_content(content_list):
        raise RuntimeError("content separation exploded")

    monkeypatch.setattr(
        "raganything.processor.separate_content",
        boom_separate_content,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "content separation exploded"
    assert processor.lightrag.doc_status.index_done_calls >= 1

    # Outer except records the failure, then finally re-enables scan and appends
    # a "completed" history line (current main-branch ordering).
    assert pipeline_status["scan_disabled"] is False
    history = pipeline_status["history_messages"]
    assert any(
        "RAGAnything processing failed for sample.pdf: content separation exploded"
        in msg
        for msg in history
    )
    assert any("Now is allowed to scan" in msg for msg in history)
    assert any(
        "RAGAnything processing completed for sample.pdf" in msg for msg in history
    )
    assert (
        pipeline_status["latest_message"]
        == "RAGAnything processing completed for sample.pdf"
    )


@pytest.mark.asyncio
async def test_lightrag_api_outer_except_survives_pipeline_update_failure(monkeypatch):
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

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "hello"}], "doc-content")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    class FailAfterFirstLock:
        """Allow the initial scan_disabled=True acquire; fail later updates."""

        def __init__(self):
            self.acquires = 0
            self._inner = asyncio.Lock()

        async def __aenter__(self):
            self.acquires += 1
            if self.acquires > 1:
                raise RuntimeError("pipeline lock broken")
            await self._inner.acquire()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            if self.acquires == 1:
                self._inner.release()
            return False

    pipeline_status = {"history_messages": [], "scan_disabled": False}
    lock = FailAfterFirstLock()

    async def fake_get_namespace_data(namespace):
        return pipeline_status

    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_namespace_data",
        fake_get_namespace_data,
    )
    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_pipeline_status_lock",
        lambda: lock,
    )
    monkeypatch.setattr(
        "raganything.processor.separate_content",
        lambda _content_list: (_ for _ in ()).throw(RuntimeError("post-parse boom")),
    )

    result = await processor.process_document_complete_lightrag_api("broken.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-broken.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "post-parse boom"
    # Outer-except / finally pipeline updates failed, but doc failure persisted.
    assert processor.lightrag.doc_status.index_done_calls >= 1
    assert lock.acquires >= 2
