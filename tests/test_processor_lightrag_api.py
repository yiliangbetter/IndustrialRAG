import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin
from raganything.utils import insert_text_content_with_multimodal_content


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


class FakePipelineLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_processor():
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
            "allow_embedding_only_ingestion": False,
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    return processor


def patch_pipeline_status(monkeypatch):
    import lightrag.kg.shared_storage as shared_storage

    pipeline_status = {"history_messages": []}

    async def fake_get_namespace_data(namespace):
        assert namespace == "pipeline_status"
        return pipeline_status

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_pipeline_status_lock", lambda: FakePipelineLock()
    )
    return pipeline_status


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_persists_failed_doc_status():
    processor = make_processor()

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
async def test_lightrag_api_parse_failure_persists_failed_doc_status(monkeypatch):
    processor = make_processor()
    patch_pipeline_status(monkeypatch)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        raise RuntimeError("parse exploded")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "parse exploded"
    assert processor.lightrag.doc_status.index_done_calls == 2


@pytest.mark.asyncio
async def test_lightrag_api_recovers_mineru_v2_text_and_marks_processed(
    monkeypatch,
):
    import raganything.processor as processor_module

    processor = make_processor()
    patch_pipeline_status(monkeypatch)
    insert_call = {}

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return (
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Recovered paragraph"}
                        ]
                    },
                }
            ],
            "doc-original",
        )

    async def fake_insert_text_content_with_multimodal_content(lightrag, **kwargs):
        insert_call.update(kwargs)

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert_text_content_with_multimodal_content,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert insert_call["input"] == "Recovered paragraph"
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["content"] == "Recovered paragraph"
    assert doc_status["content_length"] == len("Recovered paragraph")
    assert processor.lightrag.doc_status.index_done_calls == 2


@pytest.mark.asyncio
async def test_process_document_complete_embedding_only_normalizes_nested_v2_blocks():
    processor = make_processor()
    processor.config.allow_embedding_only_ingestion = True
    inserted = {}

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return (
            [
                [
                    {
                        "type": "paragraph",
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "Nested paragraph"}
                            ]
                        },
                    }
                ]
            ],
            "doc-empty-hash",
        )

    async def fake_insert_text_content_embedding_only(text_content, file_ref, doc_id):
        inserted["text_content"] = text_content
        inserted["file_ref"] = file_ref
        inserted["doc_id"] = doc_id

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    processor._insert_text_content_embedding_only = (
        fake_insert_text_content_embedding_only
    )

    await processor.process_document_complete("sample.pdf")

    assert inserted["text_content"] == "Nested paragraph"
    assert inserted["file_ref"] == "sample.pdf"
    assert inserted["doc_id"] != "doc-empty-hash"


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_content_propagates_errors():
    class BadLightRAG:
        async def ainsert(self, **kwargs):
            raise RuntimeError("insert failed")

    with pytest.raises(RuntimeError, match="insert failed"):
        await insert_text_content_with_multimodal_content(
            BadLightRAG(), input="hello", multimodal_content=[]
        )
