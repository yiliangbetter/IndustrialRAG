"""use_full_path must reach insert citations and LightRAG API pre-ids.

_get_file_reference itself is covered by #141. These tests lock the ingest
wiring: if process_document_complete / insert_content_list / the API path
ignore the flag, chunk citations and doc-pre-* status keys silently drift
from the configured identity.
"""

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


class DummyProcessor(ProcessorMixin):
    pass


def _processor(*, use_full_path: bool):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
            "display_content_stats": False,
            "parse_method": "auto",
            "parser": "mineru",
            "parser_output_dir": "./output",
            "use_full_path": use_full_path,
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    processor.mm_calls = []
    processor.mark_calls = []

    async def fake_ensure():
        return {"success": True}

    async def fake_mm(items, file_path, doc_id):
        processor.mm_calls.append((list(items), file_path, doc_id))

    async def fake_mark(doc_id):
        processor.mark_calls.append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    return processor


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


@pytest.mark.asyncio
async def test_process_document_complete_cites_full_path_when_configured(
    monkeypatch,
):
    processor = _processor(use_full_path=True)
    insert_kwargs = []

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "intro"}], "doc-from-content")

    async def fake_insert(lightrag, **kwargs):
        insert_kwargs.append(kwargs)

    processor.parse_document = fake_parse
    import raganything.processor as processor_module

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.process_document_complete("/data/manuals/pump.pdf")

    assert insert_kwargs[0]["file_paths"] == "/data/manuals/pump.pdf"
    assert insert_kwargs[0]["ids"] == "doc-from-content"
    assert processor.mark_calls == ["doc-from-content"]
    assert processor.mm_calls == []


@pytest.mark.asyncio
async def test_insert_content_list_cites_full_path_when_configured(monkeypatch):
    processor = _processor(use_full_path=True)
    insert_kwargs = []

    async def fake_insert(lightrag, **kwargs):
        insert_kwargs.append(kwargs)

    import raganything.processor as processor_module

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [{"type": "text", "text": "body"}],
        file_path="/data/manuals/pump.pdf",
        doc_id="doc-list",
    )

    assert insert_kwargs[0]["file_paths"] == "/data/manuals/pump.pdf"
    assert insert_kwargs[0]["ids"] == "doc-list"
    assert processor.mark_calls == ["doc-list"]


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_uses_full_path_pre_id(monkeypatch):
    """doc-pre-* keys must include the configured file reference, not basename."""
    _patch_pipeline_status(monkeypatch)
    processor = _processor(use_full_path=True)

    async def fake_ensure():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api(
        "/data/manuals/pump.pdf"
    )

    assert result is False
    status = processor.lightrag.doc_status.records["doc-pre-/data/manuals/pump.pdf"]
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "missing llm_model_func"
    assert status["file_path"] == "/data/manuals/pump.pdf"
    assert "doc-pre-pump.pdf" not in processor.lightrag.doc_status.records
    assert processor.lightrag.doc_status.index_done_calls == 1
