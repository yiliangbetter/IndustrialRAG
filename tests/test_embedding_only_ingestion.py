"""Regression tests for embedding-only ingestion paths.

Covers the parsed-document workflow (`process_document_complete`) and MinerU v2
plaintext recovery used when content lists lack flat `type=text` blocks. These
paths write directly to LightRAG storages and emit completion callbacks; silent
regressions would mark documents processed with missing or empty vectors.
"""

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
async def test_process_document_complete_embedding_only_persists_and_completes(
    tmp_path,
):
    """Parsed-document embedding-only path must store chunks and emit complete."""
    processor = make_embedding_only_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    async def fake_parse_document(*args, **kwargs):
        return (
            [
                {"type": "text", "text": "First section"},
                {"type": "text", "text": "Second section"},
            ],
            "doc-from-content",
        )

    processor.parse_document = fake_parse_document

    await processor.process_document_complete("report.pdf")

    assert callback.document_completes == [("report.pdf", "doc-from-content")]
    assert callback.document_errors == []

    doc_status = processor.lightrag.doc_status.records["doc-from-content"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 2
    assert len(doc_status["chunks_list"]) == 2
    assert doc_status["file_path"] == "report.pdf"

    assert set(processor.lightrag.text_chunks.records) == set(
        processor.lightrag.chunks_vdb.records
    )
    assert len(processor.lightrag.text_chunks.records) == 2
    chunk_text = "\n".join(
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    )
    assert "First section" in chunk_text
    assert "Second section" in chunk_text

    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_recovers_mineru_v2_plaintext(
    tmp_path,
):
    """MinerU v2 nested blocks must become searchable chunks without flat text."""
    processor = make_embedding_only_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    await processor.insert_content_list(
        [
            # Nested wrapper as exported by MinerU *_content_list_v2.json
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [
                            {"type": "text", "content": "Executive Summary"}
                        ]
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
            ]
        ],
        file_path="output/data_upload_test_v3/doc_content_list_v2.json",
        doc_id="doc-mineru-v2",
    )

    assert callback.document_completes == [
        ("output/data_upload_test_v3/doc_content_list_v2.json", "doc-mineru-v2")
    ]
    assert callback.document_errors == []

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
    assert processor.lightrag.insert_done_calls == 1
