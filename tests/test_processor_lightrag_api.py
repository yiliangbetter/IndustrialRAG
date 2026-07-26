import hashlib
from pathlib import Path

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
    abs_path = str(Path("sample.pdf").resolve())
    doc_pre_id = f"doc-pre-{hashlib.md5(abs_path.encode()).hexdigest()}"
    doc_status = processor.lightrag.doc_status.records[doc_pre_id]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing llm_model_func"
    assert doc_status["file_path"] == "sample.pdf"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_doc_pre_id_unique_for_same_basename():
    """Same basename in different dirs must not share doc-pre status keys."""

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

    path_a = "/tmp/dir_a/report.pdf"
    path_b = "/tmp/dir_b/report.pdf"
    await processor.process_document_complete_lightrag_api(path_a)
    await processor.process_document_complete_lightrag_api(path_b)

    key_a = f"doc-pre-{hashlib.md5(str(Path(path_a).resolve()).encode()).hexdigest()}"
    key_b = f"doc-pre-{hashlib.md5(str(Path(path_b).resolve()).encode()).hexdigest()}"
    assert key_a != key_b
    assert key_a in processor.lightrag.doc_status.records
    assert key_b in processor.lightrag.doc_status.records
    assert processor.lightrag.doc_status.records[key_a]["file_path"] == "report.pdf"
    assert processor.lightrag.doc_status.records[key_b]["file_path"] == "report.pdf"
