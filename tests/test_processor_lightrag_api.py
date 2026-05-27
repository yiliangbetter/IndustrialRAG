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

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeLightRAG:
    def __init__(self):
        self.doc_status = FakeDocStatusStorage()


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeEmbeddingLightRAG:
    def __init__(self):
        self.tokenizer = type(
            "Tokenizer",
            (),
            {"encode": staticmethod(lambda text: text.split())},
        )()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


def _configure_lightrag_api_processor(processor):
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
            "parser_output_dir": None,
            "parse_method": None,
            "display_content_stats": False,
            "content_format": "mineru",
        },
    )()
    processor.lightrag = FakeLightRAG()


def _configure_embedding_processor(processor):
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
            "display_content_stats": False,
            "allow_embedding_only_ingestion": True,
            "content_format": "mineru",
        },
    )()
    processor.lightrag = FakeEmbeddingLightRAG()


def _patch_pipeline_status(monkeypatch):
    import lightrag.kg.shared_storage as shared_storage

    pipeline_status = {"history_messages": []}

    async def fake_get_namespace_data(namespace):
        assert namespace == "pipeline_status"
        return pipeline_status

    monkeypatch.setattr(
        shared_storage, "get_namespace_data", fake_get_namespace_data
    )
    monkeypatch.setattr(
        shared_storage, "get_pipeline_status_lock", lambda: FakePipelineLock()
    )
    return pipeline_status


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
async def test_lightrag_api_recovers_mineru_v2_text(monkeypatch):
    import raganything.processor as processor_module

    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    _configure_lightrag_api_processor(processor)
    _patch_pipeline_status(monkeypatch)

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
            "doc-recovered",
        )

    inserted = {}

    async def fake_insert_text_content_with_multimodal_content(*args, **kwargs):
        inserted.update(kwargs)

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert_text_content_with_multimodal_content,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert inserted["input"] == "Recovered paragraph"
    assert processor.lightrag.doc_status.index_done_calls >= 2


@pytest.mark.asyncio
async def test_lightrag_api_insert_failure_marks_failed(monkeypatch):
    import raganything.processor as processor_module

    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    _configure_lightrag_api_processor(processor)
    _patch_pipeline_status(monkeypatch)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "hello"}], "doc-fails")

    async def fake_insert_text_content_with_multimodal_content(*args, **kwargs):
        raise RuntimeError("storage unavailable")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert_text_content_with_multimodal_content,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "storage unavailable"


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_content_propagates_errors():
    class FailingLightRAG:
        async def ainsert(self, **kwargs):
            raise RuntimeError("insert failed")

    with pytest.raises(RuntimeError, match="insert failed"):
        await insert_text_content_with_multimodal_content(
            FailingLightRAG(),
            input="hello",
            multimodal_content=[],
            file_paths="sample.pdf",
        )


@pytest.mark.asyncio
async def test_embedding_only_content_list_recovers_flat_mineru_fields():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    _configure_embedding_processor(processor)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    await processor.insert_content_list(
        [
            {
                "type": "image",
                "img_path": "/tmp/figure.png",
                "image_caption": ["Figure 1 architecture"],
            },
            {
                "type": "table",
                "table_caption": ["Results"],
                "table_body": "| A | B |",
            },
        ],
        file_path="doc_content_list_v2.json",
        doc_id="doc-flat",
    )

    chunk_texts = [
        record["content"] for record in processor.lightrag.text_chunks.records.values()
    ]
    assert chunk_texts == ["Figure 1 architecture", "Results\n| A | B |"]
    assert (
        processor.lightrag.doc_status.records["doc-flat"]["status"]
        == DocStatus.PROCESSED
    )


@pytest.mark.asyncio
async def test_embedding_only_empty_content_list_fails_instead_of_succeeding():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    _configure_embedding_processor(processor)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    with pytest.raises(ValueError, match="No text content"):
        await processor.insert_content_list(
            [{"type": "image", "img_path": "/tmp/figure.png"}],
            file_path="empty_content_list_v2.json",
            doc_id="doc-empty",
        )

    assert processor.lightrag.text_chunks.records == {}
    assert processor.lightrag.doc_status.records == {}
