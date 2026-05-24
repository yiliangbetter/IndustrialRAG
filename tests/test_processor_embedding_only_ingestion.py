from types import SimpleNamespace

import pytest

from raganything.callbacks import CallbackManager
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


class FakeStorage:
    def __init__(self):
        self.upserts = []
        self.index_done_calls = 0

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self):
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class DummyEmbeddingOnlyProcessor(ProcessorMixin):
    pass


def _make_processor(tmp_path):
    processor = DummyEmbeddingOnlyProcessor()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=True,
        content_format="mineru",
        display_content_stats=False,
        parse_method="auto",
        parser_output_dir=str(tmp_path),
        use_full_path=False,
    )
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()
    processor.callback_manager.enable_event_log(True)
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_recovers_mineru_text_and_completes(
    tmp_path,
):
    processor = _make_processor(tmp_path)
    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Risk Controls"}
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Validate every upload."}
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [
                                {"type": "text", "content": "Reject empty files"}
                            ],
                        }
                    ]
                },
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/source_content_list_v2.json",
        doc_id="doc-explicit",
    )

    chunk_upsert = processor.lightrag.text_chunks.upserts[0]
    chunk_contents = [chunk["content"] for chunk in chunk_upsert.values()]
    assert chunk_contents == [
        "Risk Controls",
        "Validate every upload.",
        "1. Reject empty files",
    ]
    assert processor.lightrag.chunks_vdb.upserts[0] == chunk_upsert
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1

    status = processor.lightrag.doc_status.upserts[0]["doc-explicit"]
    assert status["chunks_count"] == 3
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "source_content_list_v2.json"
    assert "Validate every upload." in status["content"]

    complete_events = [
        event
        for event in processor.callback_manager.event_log
        if event.event_type == "on_document_complete"
    ]
    assert len(complete_events) == 1
    assert complete_events[0].file_path == "/tmp/source_content_list_v2.json"
    assert complete_events[0].doc_id == "doc-explicit"


@pytest.mark.asyncio
async def test_process_document_embedding_only_emits_completion_after_storage(tmp_path):
    processor = _make_processor(tmp_path)
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return ([{"type": "text", "text": "parsed body"}], "doc-content")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document

    await processor.process_document_complete(
        str(input_path),
        doc_id="doc-override",
        file_name="display-name.pdf",
    )

    status = processor.lightrag.doc_status.upserts[0]["doc-override"]
    assert status["status"].value == "processed"
    assert status["chunks_count"] == 1
    assert status["file_path"] == "display-name.pdf"
    assert processor.lightrag.insert_done_calls == 1

    event_log = processor.callback_manager.event_log
    assert event_log[-1].event_type == "on_document_complete"
    assert event_log[-1].file_path == str(input_path)
    assert event_log[-1].doc_id == "doc-override"
