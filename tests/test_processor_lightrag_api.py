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


class FakeKVStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

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
async def test_embedding_only_insert_persists_all_storage_callbacks():
    class DummyProcessor(ProcessorMixin):
        pass

    class FakeTokenizer:
        def encode(self, text):
            return text.split()

    class FakeLightRAG:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.text_chunks = FakeKVStorage()
            self.chunks_vdb = FakeKVStorage()
            self.doc_status = FakeKVStorage()
            self.insert_done_calls = 0

        async def _insert_done(self):
            self.insert_done_calls += 1

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()

    await processor._insert_text_content_embedding_only(
        "first paragraph\n\nsecond paragraph", "sample.pdf", "doc-123"
    )

    assert len(processor.lightrag.text_chunks.records) == 2
    assert processor.lightrag.text_chunks.records == processor.lightrag.chunks_vdb.records
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    status = processor.lightrag.doc_status.records["doc-123"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 2
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "sample.pdf"
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_store_cached_result_flushes_parse_cache(tmp_path):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parse_method": "auto",
        },
    )()
    processor.parse_cache = FakeKVStorage()

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "hello"}],
        "doc-123",
        file_path,
        lang="en",
        ignored="value",
    )

    cached = processor.parse_cache.records["cache-key"]
    assert cached["doc_id"] == "doc-123"
    assert cached["content_list"] == [{"type": "text", "text": "hello"}]
    assert cached["parse_config"] == {
        "parser": "mineru",
        "parse_method": "auto",
        "lang": "en",
    }
    assert processor.parse_cache.index_done_calls == 1
