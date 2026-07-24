import asyncio

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


def _patch_pipeline_status(monkeypatch):
    from lightrag.kg import shared_storage

    pipeline_status = {"history_messages": []}
    pipeline_status_lock = asyncio.Lock()

    async def fake_get_namespace_data(namespace):
        assert namespace == "pipeline_status"
        return pipeline_status

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_pipeline_status_lock", lambda: pipeline_status_lock
    )


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
async def test_lightrag_api_insert_failure_persists_failed_doc_status(
    monkeypatch, tmp_path
):
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
            "parser_output_dir": str(tmp_path),
            "parse_method": "auto",
            "display_content_stats": False,
        },
    )()

    class FakeLightRAG:
        def __init__(self):
            self.doc_status = FakeDocStatusStorage()

        async def ainsert(self, **kwargs):
            raise RuntimeError("vector storage unavailable")

    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "hello world"}], "doc-123")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    _patch_pipeline_status(monkeypatch)

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "vector storage unavailable"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_recovers_mineru_v2_plaintext_when_no_type_text(
    monkeypatch, tmp_path
):
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
            "parser_output_dir": str(tmp_path),
            "parse_method": "auto",
            "display_content_stats": False,
            "content_format": "minerU",
        },
    )()

    class FakeLightRAG:
        def __init__(self):
            self.doc_status = FakeDocStatusStorage()
            self.inserted_inputs = []

        async def ainsert(self, **kwargs):
            self.inserted_inputs.append(kwargs.get("input"))

    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        # MinerU v2: prose lives in paragraph/title, not type=text.
        return (
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "Safety"}]
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Wear protective gloves."}
                        ]
                    },
                },
            ],
            "doc-content-v2",
        )

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    _patch_pipeline_status(monkeypatch)

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert len(processor.lightrag.inserted_inputs) == 1
    inserted = processor.lightrag.inserted_inputs[0]
    assert "Safety" in inserted
    assert "Wear protective gloves." in inserted
