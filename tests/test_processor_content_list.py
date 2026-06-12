from types import SimpleNamespace

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


class RecordingStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class RecordingCallbackManager:
    def __init__(self):
        self.events = []

    def dispatch(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(*, allow_embedding_only_ingestion=False):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=allow_embedding_only_ingestion,
        content_format="minerU",
        display_content_stats=False,
        use_full_path=False,
    )
    processor.callback_manager = RecordingCallbackManager()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_flattens_mineru_blocks_and_persists_status():
    processor = make_processor(allow_embedding_only_ingestion=True)

    text_chunks = RecordingStorage()
    chunks_vdb = RecordingStorage()
    doc_status = RecordingStorage()
    insert_done_calls = 0

    async def fake_insert_done():
        nonlocal insert_done_calls
        insert_done_calls += 1

    processor.lightrag = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: text.split()),
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        doc_status=doc_status,
        _insert_done=fake_insert_done,
    )

    async def fail_if_multimodal_processed(*args, **kwargs):
        raise AssertionError("embedding-only ingestion must not process multimodal items")

    processor._process_multimodal_content = fail_if_multimodal_processed

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Risk Controls"},
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Validate limits"},
                        {"type": "text", "content": "before approval."},
                    ]
                },
            },
        ],
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "1.",
                        "item_content": [{"type": "text", "content": "Check owner"}],
                    },
                    {
                        "prefix": "2.",
                        "item_content": [{"type": "text", "content": "Check amount"}],
                    },
                ]
            },
        },
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/uploaded/report_content_list_v2.json",
        doc_id="doc-risk-controls",
    )

    assert text_chunks.records == chunks_vdb.records
    assert len(text_chunks.records) == 3
    chunk_values = list(text_chunks.records.values())
    assert [chunk["content"] for chunk in chunk_values] == [
        "Risk Controls",
        "Validate limits before approval.",
        "1. Check owner\n2. Check amount",
    ]
    assert all(
        chunk["file_path"] == "report_content_list_v2.json" for chunk in chunk_values
    )
    assert text_chunks.index_done_calls == 1
    assert chunks_vdb.index_done_calls == 1
    assert insert_done_calls == 1

    status = doc_status.records["doc-risk-controls"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 3
    assert status["chunks_list"] == list(text_chunks.records.keys())
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "report_content_list_v2.json"
    assert doc_status.index_done_calls == 1
    assert processor.callback_manager.events[-1][0] == "on_document_complete"


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_complete_after_text_insert(
    monkeypatch,
):
    import raganything.processor as processor_module

    processor = make_processor()
    processor.lightrag = SimpleNamespace()
    marked_doc_ids = []
    processed_multimodal = []
    insert_calls = []

    async def fake_insert_text_content(*args, **kwargs):
        insert_calls.append((args, kwargs))

    async def fake_mark_multimodal_processing_complete(doc_id):
        marked_doc_ids.append(doc_id)

    async def fake_process_multimodal_content(*args, **kwargs):
        processed_multimodal.append((args, kwargs))

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )
    processor._mark_multimodal_processing_complete = (
        fake_mark_multimodal_processing_complete
    )
    processor._process_multimodal_content = fake_process_multimodal_content

    await processor.insert_content_list(
        [
            [{"type": "text", "text": "First paragraph"}],
            {"type": "image", "img_path": "/tmp/chart.png", "page_idx": 0},
            {"type": "text", "text": "Second paragraph"},
        ],
        file_path="/tmp/uploaded/mixed_content_list_v2.json",
        doc_id="doc-mixed",
        skip_multimodal_processing=True,
    )

    assert len(insert_calls) == 1
    _, insert_kwargs = insert_calls[0]
    assert insert_kwargs["input"] == "First paragraph\n\nSecond paragraph"
    assert insert_kwargs["file_paths"] == "mixed_content_list_v2.json"
    assert insert_kwargs["ids"] == "doc-mixed"
    assert marked_doc_ids == ["doc-mixed"]
    assert processed_multimodal == []
    assert [event[0] for event in processor.callback_manager.events] == [
        "on_text_insert_start",
        "on_text_insert_complete",
        "on_document_complete",
    ]
