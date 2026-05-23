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


class FakeKVStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self):
        self.doc_status = FakeDocStatusStorage()
        self.text_chunks = FakeKVStorage()
        self.chunks_vdb = FakeKVStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class AsyncNullLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_processor(*, allow_embedding_only_ingestion=False):
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
            "parser_output_dir": "",
            "parse_method": "auto",
            "display_content_stats": False,
            "content_format": "mineru",
            "allow_embedding_only_ingestion": allow_embedding_only_ingestion,
        },
    )()
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.fixture
def fake_pipeline_status(monkeypatch):
    import lightrag.kg.shared_storage as shared_storage

    status = {"history_messages": []}

    async def fake_get_namespace_data(name):
        assert name == "pipeline_status"
        return status

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_pipeline_status_lock", lambda: AsyncNullLock()
    )
    return status


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
async def test_lightrag_api_inserts_mineru_v2_paragraph_text(
    monkeypatch, fake_pipeline_status
):
    processor = make_processor()

    async def fake_parse_document(*args, **kwargs):
        return (
            [
                [
                    {
                        "type": "paragraph",
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "hello from v2"}
                            ]
                        },
                    }
                ]
            ],
            "doc-empty",
        )

    insert_calls = []

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        "raganything.processor.insert_text_content_with_multimodal_content",
        fake_insert,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert insert_calls[0]["input"] == "hello from v2"
    assert insert_calls[0]["ids"] != "doc-empty"
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["content"] == "hello from v2"
    assert doc_status["content_length"] == len("hello from v2")
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_insert_failure_returns_false_and_persists_failed_status(
    monkeypatch, fake_pipeline_status
):
    processor = make_processor()

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "hello"}], "doc-hello")

    async def fake_insert(*args, **kwargs):
        raise RuntimeError("storage down")

    processor.parse_document = fake_parse_document
    monkeypatch.setattr(
        "raganything.processor.insert_text_content_with_multimodal_content",
        fake_insert,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "storage down"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_parse_failure_persists_failed_status(fake_pipeline_status):
    processor = make_processor()

    async def fake_parse_document(*args, **kwargs):
        raise RuntimeError("parse failed")

    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "parse failed"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_process_document_embedding_only_recovers_mineru_v2_paragraph_text():
    processor = make_processor(allow_embedding_only_ingestion=True)

    async def fake_parse_document(*args, **kwargs):
        return (
            [
                [
                    {
                        "type": "paragraph",
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "embedded v2 text"}
                            ]
                        },
                    }
                ]
            ],
            "doc-empty",
        )

    processor.parse_document = fake_parse_document

    await processor.process_document_complete("sample.pdf")

    stored_chunks = processor.lightrag.text_chunks.records
    assert len(stored_chunks) == 1
    assert next(iter(stored_chunks.values()))["content"] == "embedded v2 text"
    doc_status = next(iter(processor.lightrag.doc_status.records.values()))
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 1
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_multimodal_insert_helper_propagates_lightrag_failures():
    class FailingLightRAG:
        async def ainsert(self, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await insert_text_content_with_multimodal_content(
            FailingLightRAG(),
            input="hello",
            multimodal_content=[],
        )
