"""LightRAG API ingest must preserve existing doc status and restore pipeline scan.

The existing callback test only covers init failure on a *new* document. These
cases lock the merge/fail-closed branches that would otherwise drop stored
fields or leave scan disabled after a successful parse.
"""

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
    def __init__(self, records=None, get_error=None):
        self.records = dict(records or {})
        self.index_done_calls = 0
        self.get_error = get_error

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _processor(*, use_full_path=False, parser="mineru", storage=None, lightrag=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = types.SimpleNamespace(
        use_full_path=use_full_path,
        parser=parser,
        parser_output_dir="./output",
        parse_method="auto",
        display_content_stats=False,
        content_format="minerU",
    )
    if lightrag is not None:
        processor.lightrag = lightrag
    else:
        processor.lightrag = types.SimpleNamespace(
            doc_status=storage if storage is not None else FakeDocStatusStorage()
        )
    return processor


@pytest.mark.asyncio
async def test_init_failure_preserves_existing_doc_status_fields():
    storage = FakeDocStatusStorage(
        {
            "doc-pre-sample.pdf": {
                "status": DocStatus.READY,
                "content": "already parsed",
                "error_msg": "",
                "content_summary": "summary",
                "multimodal_content": [{"type": "image"}],
                "scheme_name": "plant",
                "content_length": 42,
                "created_at": "2020-01-01T00:00:00+00:00",
                "updated_at": "old-timestamp",
                "file_path": "sample.pdf",
                "custom_field": "keep-me",
            }
        }
    )
    processor = _processor(storage=storage)

    async def fake_ensure():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    saved = storage.records["doc-pre-sample.pdf"]
    assert saved["status"] == DocStatus.FAILED
    assert saved["error_msg"] == "missing llm_model_func"
    assert saved["content"] == "already parsed"
    assert saved["content_summary"] == "summary"
    assert saved["scheme_name"] == "plant"
    assert saved["custom_field"] == "keep-me"
    assert saved["created_at"] == "2020-01-01T00:00:00+00:00"
    assert saved["updated_at"] != "old-timestamp"
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_init_failure_without_doc_status_returns_false_without_raising():
    processor = _processor(lightrag=types.SimpleNamespace(doc_status=None))

    async def fake_ensure():
        return {"success": False, "error": "not ready"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api("sample.pdf")
    assert result is False


@pytest.mark.asyncio
async def test_init_failure_storage_error_is_swallowed_and_returns_false():
    storage = FakeDocStatusStorage(get_error=RuntimeError("kv down"))
    processor = _processor(storage=storage)

    async def fake_ensure():
        return {"success": False, "error": "missing embedding"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api("sample.pdf")
    assert result is False
    assert storage.records == {}
    assert storage.index_done_calls == 0


@pytest.mark.asyncio
async def test_parser_kwarg_updates_config_before_init():
    processor = _processor(parser="mineru")
    seen = []

    async def fake_ensure():
        seen.append(processor.config.parser)
        return {"success": False, "error": "stop"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf", parser="docling"
    )

    assert result is False
    assert seen == ["docling"]
    assert processor.config.parser == "docling"


@pytest.mark.asyncio
async def test_doc_pre_id_uses_full_path_when_configured():
    storage = FakeDocStatusStorage()
    processor = _processor(use_full_path=True, storage=storage)

    async def fake_ensure():
        return {"success": False, "error": "missing llm"}

    processor._ensure_lightrag_initialized = fake_ensure

    file_path = "/abs/manuals/pump.pdf"
    result = await processor.process_document_complete_lightrag_api(file_path)

    assert result is False
    assert f"doc-pre-{file_path}" in storage.records
    assert storage.records[f"doc-pre-{file_path}"]["file_path"] == file_path


@pytest.mark.asyncio
async def test_successful_parse_re_enables_pipeline_scan(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    storage = FakeDocStatusStorage()
    processor = _processor(storage=storage)
    pipeline_status = {"scan_disabled": False, "history_messages": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "hello"}], "doc-abc")

    async def fake_insert(*args, **kwargs):
        return None

    async def fake_get_namespace_data(name):
        assert name == "pipeline_status"
        return pipeline_status

    def fake_get_pipeline_status_lock():
        return DummyLock()

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse_document

    import raganything.processor as processor_module

    monkeypatch.setattr(
        processor_module, "insert_text_content_with_multimodal_content", fake_insert
    )

    try:
        import lightrag.kg.shared_storage as shared_storage
    except ImportError:
        shared_storage = types.ModuleType("lightrag.kg.shared_storage")
        monkeypatch.setitem(
            __import__("sys").modules, "lightrag.kg.shared_storage", shared_storage
        )

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = await processor.process_document_complete_lightrag_api(str(pdf))

    assert result is True
    assert pipeline_status["scan_disabled"] is False
    assert any(
        "Now is allowed to scan" in msg for msg in pipeline_status["history_messages"]
    )
