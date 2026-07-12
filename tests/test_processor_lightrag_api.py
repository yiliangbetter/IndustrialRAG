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
        self.upsert_calls = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upsert_calls.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self, insert_error=None):
        self.doc_status = FakeDocStatusStorage()
        self.text_chunks = FakeDocStatusStorage()
        self.chunks_vdb = FakeDocStatusStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_error = insert_error
        self.ainsert_calls = []
        self.insert_done_calls = 0

    async def ainsert(self, **kwargs):
        self.ainsert_calls.append(kwargs)
        if self.insert_error is not None:
            raise self.insert_error

    async def _insert_done(self):
        self.insert_done_calls += 1


def make_config(tmp_path, *, allow_embedding_only_ingestion=False):
    return type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
            "parse_method": "auto",
            "parser_output_dir": str(tmp_path),
            "display_content_stats": False,
            "content_format": "minerU",
            "allow_embedding_only_ingestion": allow_embedding_only_ingestion,
        },
    )()


def install_fake_pipeline_status(monkeypatch):
    import lightrag.kg.shared_storage as shared_storage

    async def fake_get_namespace_data(_namespace):
        return {"history_messages": []}

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_pipeline_status_lock", lambda: FakeAsyncLock()
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
async def test_embedding_only_process_document_recovers_mineru_v2_text(tmp_path):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = make_config(tmp_path, allow_embedding_only_ingestion=True)
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*_args, **_kwargs):
        return (
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "Safety Manual"}]
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Do not disable power guards."}
                        ]
                    },
                },
            ],
            "doc-from-empty-legacy-text",
        )

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    await processor.process_document_complete(str(tmp_path / "manual.pdf"))

    indexed_text = "\n".join(
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    )
    assert "Safety Manual" in indexed_text
    assert "Do not disable power guards." in indexed_text
    doc_status = next(iter(processor.lightrag.doc_status.records.values()))
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 2


@pytest.mark.asyncio
async def test_lightrag_api_insert_failure_returns_false_and_persists_failed_status(
    monkeypatch, tmp_path
):
    install_fake_pipeline_status(monkeypatch)

    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = make_config(tmp_path)
    processor.lightrag = FakeLightRAG(insert_error=RuntimeError("insert failed"))

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*_args, **_kwargs):
        return ([{"type": "text", "text": "important content"}], "doc-abc")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "insert failed"
    assert processor.lightrag.doc_status.index_done_calls >= 1


@pytest.mark.asyncio
async def test_lightrag_api_success_recovers_text_and_marks_pre_status_processed(
    monkeypatch, tmp_path
):
    install_fake_pipeline_status(monkeypatch)

    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = make_config(tmp_path)
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*_args, **_kwargs):
        return (
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Recovered API text."}
                        ]
                    },
                }
            ],
            "doc-from-empty-legacy-text",
        )

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert processor.lightrag.ainsert_calls[0]["input"] == "Recovered API text."
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["error_msg"] == ""
    assert doc_status["content_length"] == len("Recovered API text.")
