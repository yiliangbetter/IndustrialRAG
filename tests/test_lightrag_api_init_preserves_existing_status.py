"""Init failure must merge FAILED onto an existing LightRAG doc_status row.

Re-ingest of a previously READY/HANDLING document must keep scheme_name,
citation path, and prior summaries. Wiping those fields would detach the
row from its tenant scheme and break citation identity.

Distinct from tests/test_processor_lightrag_api.py (new FAILED row) and
#176 (upsert raise is fail-open).
"""

from __future__ import annotations

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
    def __init__(self, records):
        self.records = records
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


@pytest.mark.asyncio
async def test_init_failure_preserves_existing_scheme_and_citation():
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
    existing = {
        "status": DocStatus.READY,
        "content": "prior-body",
        "error_msg": "",
        "content_summary": "keep-summary",
        "multimodal_content": [{"type": "image"}],
        "scheme_name": "plant-a",
        "content_length": 11,
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
        "file_path": "manuals/sample.pdf",
    }
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage({"doc-pre-sample.pdf": dict(existing)})},
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf", scheme_name="plant-b"
    )

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing llm_model_func"
    assert doc_status["scheme_name"] == "plant-a"
    assert doc_status["file_path"] == "manuals/sample.pdf"
    assert doc_status["content_summary"] == "keep-summary"
    assert doc_status["content"] == "prior-body"
    assert doc_status["created_at"] == "2020-01-01T00:00:00+00:00"
    assert doc_status["updated_at"] != "2020-01-01T00:00:00+00:00"
    assert processor.lightrag.doc_status.index_done_calls == 1
