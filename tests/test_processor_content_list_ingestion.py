from types import SimpleNamespace

import pytest

from raganything.base import DocStatus
from raganything.callbacks import CallbackManager, ProcessingCallback
import raganything.processor as processor_module
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


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_text_insert_start(self, file_path, text_length=0, **kwargs):
        self.events.append(("text_insert_start", file_path, text_length))

    def on_text_insert_complete(self, file_path, **kwargs):
        self.events.append(("text_insert_complete", file_path))

    def on_multimodal_start(self, file_path, item_count=0, **kwargs):
        self.events.append(("multimodal_start", file_path, item_count))

    def on_document_complete(self, file_path, doc_id="", **kwargs):
        self.events.append(("document_complete", file_path, doc_id))


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(*, allow_embedding_only=False):
    processor = DummyProcessor()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=allow_embedding_only,
        content_format="minerU",
        display_content_stats=False,
        parser="mineru",
        parser_output_dir="/tmp",
        parse_method="auto",
        use_full_path=False,
    )
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = CallbackManager()

    async def ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = ensure_lightrag_initialized
    return processor


def test_mineru_v2_blocks_normalize_and_recover_plaintext():
    processor = make_processor()
    content_list = [
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pump"},
                        {"content": [{"type": "text", "content": "failure"}]},
                    ]
                },
            }
        ],
        {
            "type": "title",
            "content": {"title_content": {"text": "Mitigation plan"}},
        },
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "1.",
                        "item_content": [{"type": "text", "content": "Inspect seals"}],
                    },
                    {"item_content": {"text": "Replace worn impeller"}},
                ]
            },
        },
        {"type": "table", "content": {"html": "<table><tr><td>Risk</td></tr></table>"}},
        {"type": "image", "content": {"image_caption": ["Figure 1", "Pump curve"]}},
        ["ignored", {"type": "text", "text": "Trailing note"}],
        "not-a-block",
    ]

    normalized = processor._normalize_nested_content_list(content_list)
    plaintext = processor._plaintext_from_mineru_blocks(normalized)

    assert [item["type"] for item in normalized] == [
        "paragraph",
        "title",
        "list",
        "table",
        "image",
        "text",
    ]
    assert plaintext == (
        "Pump failure\n\n"
        "Mitigation plan\n\n"
        "1. Inspect seals\n"
        "Replace worn impeller\n\n"
        "<table><tr><td>Risk</td></tr></table>\n\n"
        "Figure 1 Pump curve\n\n"
        "Trailing note"
    )


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_stores_normalized_mineru_v2_text():
    processor = make_processor(allow_embedding_only=True)
    callback = RecordingCallback()
    processor.callback_manager.register(callback)
    content_list = [
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pump failure analysis"}
                    ]
                },
            }
        ],
        {
            "type": "title",
            "content": {"title_content": [{"type": "text", "content": "Mitigation"}]},
        },
    ]
    empty_doc_id = processor._generate_content_based_doc_id([])

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/uploads/report_content_list_v2.json",
    )

    assert len(processor.lightrag.doc_status.records) == 1
    doc_id, doc_status = next(iter(processor.lightrag.doc_status.records.items()))
    assert doc_id != empty_doc_id
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 2
    assert doc_status["chunks_list"] == list(processor.lightrag.text_chunks.records)
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "report_content_list_v2.json"
    assert [chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()] == [
        "Pump failure analysis",
        "Mitigation",
    ]
    assert processor.lightrag.chunks_vdb.records == processor.lightrag.text_chunks.records
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
    assert callback.events == [
        ("document_complete", "/tmp/uploads/report_content_list_v2.json", doc_id)
    ]


@pytest.mark.asyncio
async def test_process_document_complete_embedding_only_dispatches_completion():
    processor = make_processor(allow_embedding_only=True)
    callback = RecordingCallback()
    processor.callback_manager.register(callback)
    inserted = {}

    async def parse_document(*args, **kwargs):
        return (
            [
                {"type": "text", "text": "parsed text"},
                {"type": "image", "img_path": "/tmp/chart.png"},
            ],
            "doc-parsed",
        )

    async def insert_embedding_only(text_content, file_ref, doc_id):
        inserted.update(
            {"text_content": text_content, "file_ref": file_ref, "doc_id": doc_id}
        )

    async def fail_if_multimodal_processed(*args, **kwargs):
        raise AssertionError("embedding-only path must skip multimodal processing")

    processor.parse_document = parse_document
    processor._insert_text_content_embedding_only = insert_embedding_only
    processor._process_multimodal_content = fail_if_multimodal_processed

    await processor.process_document_complete("/tmp/uploads/manual.pdf")

    assert inserted == {
        "text_content": "parsed text",
        "file_ref": "manual.pdf",
        "doc_id": "doc-parsed",
    }
    assert callback.events == [
        ("document_complete", "/tmp/uploads/manual.pdf", "doc-parsed")
    ]


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_complete_without_processors(
    monkeypatch,
):
    processor = make_processor()
    callback = RecordingCallback()
    processor.callback_manager.register(callback)
    marked_complete = []
    insert_calls = []

    async def fake_insert_text_content(*args, **kwargs):
        insert_calls.append(kwargs)

    async def mark_multimodal_complete(doc_id):
        marked_complete.append(doc_id)

    async def fail_if_multimodal_processed(*args, **kwargs):
        raise AssertionError("skip_multimodal_processing should bypass processors")

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )
    processor._mark_multimodal_processing_complete = mark_multimodal_complete
    processor._process_multimodal_content = fail_if_multimodal_processed

    await processor.insert_content_list(
        [
            {"type": "text", "text": "pump data"},
            {"type": "image", "img_path": "/tmp/pump.png"},
        ],
        file_path="/tmp/uploads/pump_report.json",
        doc_id="doc-skip",
        skip_multimodal_processing=True,
    )

    assert insert_calls == [
        {
            "input": "pump data",
            "file_paths": "pump_report.json",
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": "doc-skip",
        }
    ]
    assert marked_complete == ["doc-skip"]
    assert callback.events == [
        ("text_insert_start", "pump_report.json", len("pump data")),
        ("text_insert_complete", "pump_report.json"),
        ("document_complete", "/tmp/uploads/pump_report.json", "doc-skip"),
    ]
