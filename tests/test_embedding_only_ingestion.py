import pytest

from raganything.base import DocStatus
from raganything.callbacks import CallbackManager, ProcessingCallback
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


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class RecordingDocumentCallback(ProcessingCallback):
    def __init__(self):
        self.document_completes = []
        self.document_errors = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.document_completes.append((file_path, doc_id))

    def on_document_error(self, file_path, doc_id=None, stage="", error="", **kwargs):
        self.document_errors.append((file_path, doc_id, stage, str(error)))


class DummyProcessor(ProcessorMixin):
    pass


def make_embedding_only_processor(tmp_path):
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "content_format": "minerU",
            "display_content_stats": False,
            "parse_method": "auto",
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "use_full_path": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()
    processor.parse_cache = None

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_emits_document_complete(tmp_path):
    processor = make_embedding_only_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    await processor.insert_content_list(
        [
            {"type": "text", "text": "First paragraph"},
            {"type": "text", "text": "Second paragraph"},
        ],
        file_path="/tmp/source.md",
        doc_id="doc-1",
    )

    assert callback.document_completes == [("/tmp/source.md", "doc-1")]
    assert callback.document_errors == []

    doc_status = processor.lightrag.doc_status.records["doc-1"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 2
    assert len(doc_status["chunks_list"]) == 2
    assert doc_status["file_path"] == "source.md"
    assert processor.lightrag.insert_done_calls == 1
    assert set(processor.lightrag.text_chunks.records) == set(
        processor.lightrag.chunks_vdb.records
    )


@pytest.mark.asyncio
async def test_process_document_complete_embedding_only_emits_document_complete(tmp_path):
    processor = make_embedding_only_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    async def fake_parse_document(*args, **kwargs):
        return [{"type": "text", "text": "Parsed document"}], "doc-from-content"

    processor.parse_document = fake_parse_document

    await processor.process_document_complete("report.pdf")

    assert callback.document_completes == [("report.pdf", "doc-from-content")]
    assert callback.document_errors == []
    assert processor.lightrag.doc_status.records["doc-from-content"][
        "status"
    ] == DocStatus.PROCESSED


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_recovers_mineru_v2_plaintext(tmp_path):
    processor = make_embedding_only_processor(tmp_path)

    await processor.insert_content_list(
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Executive Summary"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Alpha"},
                        {"type": "text", "content": "Beta"},
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [{"type": "text", "content": "First"}],
                        },
                        {"prefix": "2.", "item_content": "Second"},
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>Total</td></tr></table>"},
            },
            {"type": "image", "content": {"image_caption": ["Pump diagram"]}},
        ],
        file_path="output/data_upload_test_v3/doc_content_list_v2.json",
        doc_id="doc-mineru-v2",
    )

    doc_status = processor.lightrag.doc_status.records["doc-mineru-v2"]
    chunk_text = "\n".join(
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    )

    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 5
    assert "Executive Summary" in chunk_text
    assert "Alpha Beta" in chunk_text
    assert "1. First\n2. Second" in chunk_text
    assert "<table><tr><td>Total</td></tr></table>" in chunk_text
    assert "Pump diagram" in chunk_text
