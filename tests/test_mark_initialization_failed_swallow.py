"""LightRAG API init-failure status persistence exception swallow.

#103 covers happy-path FAILED upsert, missing doc_status, and existing-status
merge. This module covers the remaining operational edge: upsert/index_done
errors inside mark_initialization_failed must be logged and swallowed so the
API still returns False instead of crashing the caller.
"""

from types import SimpleNamespace

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class BoomDocStatusStorage:
    def __init__(self, existing=None, fail_on="upsert"):
        self.records = dict(existing or {})
        self.fail_on = fail_on
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        if self.fail_on == "upsert":
            raise RuntimeError("doc_status upsert exploded")
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1
        if self.fail_on == "index_done":
            raise RuntimeError("index_done exploded")


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_swallows_upsert_error():
    storage = BoomDocStatusStorage(fail_on="upsert")
    processor = ProcessorMixin.__new__(ProcessorMixin)
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(use_full_path=False, parser="mineru")
    processor.lightrag = SimpleNamespace(doc_status=storage)

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    assert storage.records == {}
    assert any(
        "Failed to persist initialization failure status for sample.pdf" in msg
        for msg in processor.logger.errors
    )


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_swallows_index_done_error():
    storage = BoomDocStatusStorage(fail_on="index_done")
    processor = ProcessorMixin.__new__(ProcessorMixin)
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(use_full_path=False, parser="mineru")
    processor.lightrag = SimpleNamespace(doc_status=storage)

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing embedding_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = storage.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing embedding_func"
    assert storage.index_done_calls == 1
    assert any(
        "Failed to persist initialization failure status for sample.pdf" in msg
        for msg in processor.logger.errors
    )
