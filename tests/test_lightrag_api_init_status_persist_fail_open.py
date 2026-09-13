"""LightRAG API init failure must not raise if status persist fails.

``process_document_complete_lightrag_api`` already returns ``False`` when
LightRAG init fails. If writing that FAILED ``doc_status`` row (or flushing
it) raises, the exception is logged and swallowed so the API caller still
gets ``False`` instead of a 500. A regression here turns an init problem
into an ingest crash after the document was already skipped.

Distinct from ``tests/test_processor_lightrag_api.py`` (happy-path persist)
and from #150 (init failure when ``doc_status`` is missing).
"""

from __future__ import annotations

import pytest

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


class RaisingDocStatusStorage:
    def __init__(self):
        self.upsert_calls = 0

    async def get_by_id(self, key):
        return None

    async def upsert(self, data):
        self.upsert_calls += 1
        raise RuntimeError("doc_status unavailable")

    async def index_done_callback(self):
        raise AssertionError("index flush must not run after upsert failure")


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_swallows_status_persist_error():
    processor = ProcessorMixin()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
        },
    )()
    storage = RaisingDocStatusStorage()
    processor.lightrag = type("FakeLightRAG", (), {"doc_status": storage})()

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    assert storage.upsert_calls == 1
