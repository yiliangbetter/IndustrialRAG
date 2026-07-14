import pytest

import raganything.processor as processor_module
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
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing llm_model_func"
    assert doc_status["file_path"] == "sample.pdf"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_content_list_skip_multimodal_marks_document_complete(monkeypatch):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.callback_manager = None
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    insert_calls = []

    async def fake_insert_text_content(lightrag, **kwargs):
        insert_calls.append(kwargs)
        await lightrag.doc_status.upsert(
            {
                kwargs["ids"]: {
                    "status": DocStatus.PROCESSED,
                    "multimodal_processed": False,
                }
            }
        )

    async def fail_multimodal_processing(*args, **kwargs):
        raise AssertionError("multimodal processing must be skipped")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor._process_multimodal_content = fail_multimodal_processing
    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.insert_content_list(
        [
            {"type": "text", "text": "Index this paragraph."},
            {"type": "image", "img_path": "/tmp/diagram.png"},
        ],
        file_path="/tmp/report.json",
        doc_id="doc-graph",
        skip_multimodal_processing=True,
    )

    assert insert_calls == [
        {
            "input": "Index this paragraph.",
            "file_paths": "report.json",
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": "doc-graph",
        }
    ]
    doc_status = processor.lightrag.doc_status.records["doc-graph"]
    assert doc_status["multimodal_processed"] is True
    assert processor.lightrag.doc_status.index_done_calls == 1
