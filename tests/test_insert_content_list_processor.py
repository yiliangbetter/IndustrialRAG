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


class RecordingCallbackManager:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(tmp_path, *, allow_embedding_only_ingestion):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()
    processor.callback_manager = RecordingCallbackManager()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": allow_embedding_only_ingestion,
            "display_content_stats": False,
            "use_full_path": False,
            "content_format": "minerU",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "parser": "mineru",
        },
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_recovers_mineru_v2_plaintext(
    tmp_path,
):
    processor = make_processor(tmp_path, allow_embedding_only_ingestion=True)

    async def fail_if_multimodal_processing_runs(*args, **kwargs):
        raise AssertionError("embedding-only ingestion must skip multimodal processing")

    processor._process_multimodal_content = fail_if_multimodal_processing_runs

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Quarterly Report"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Revenue"},
                        {"type": "text", "content": "grew"},
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
                        {
                            "item_content": [{"type": "text", "content": "Second"}],
                        },
                    ]
                },
            },
            {
                "type": "image",
                "content": {"image_caption": ["System diagram", "with cache"]},
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>A</td></tr></table>"},
            },
            "ignored non-dict item",
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/docs/mineru_output/report_content_list_v2.json",
        doc_id="doc-mineru-v2",
    )

    chunks = sorted(
        processor.lightrag.text_chunks.records.values(),
        key=lambda chunk: chunk["chunk_order_index"],
    )
    assert [chunk["content"] for chunk in chunks] == [
        "Quarterly Report",
        "Revenue grew",
        "1. First\nSecond",
        "System diagram with cache",
        "<table><tr><td>A</td></tr></table>",
    ]
    assert (
        processor.lightrag.chunks_vdb.records
        == processor.lightrag.text_chunks.records
    )

    doc_status = processor.lightrag.doc_status.records["doc-mineru-v2"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["chunks_count"] == 5
    assert doc_status["chunks_list"] == list(processor.lightrag.text_chunks.records)
    assert doc_status["multimodal_processed"] is True
    assert doc_status["file_path"] == "report_content_list_v2.json"
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
    assert any(
        event_name == "on_document_complete"
        and event["doc_id"] == "doc-mineru-v2"
        for event_name, event in processor.callback_manager.events
    )


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_complete_without_batch(
    monkeypatch,
    tmp_path,
):
    import raganything.processor as processor_module

    processor = make_processor(tmp_path, allow_embedding_only_ingestion=False)
    inserted = {}
    marked_doc_ids = []

    async def fake_insert_text_content(*args, **kwargs):
        inserted["args"] = args
        inserted["kwargs"] = kwargs

    async def fake_mark_multimodal_processing_complete(doc_id):
        marked_doc_ids.append(doc_id)

    async def fail_if_multimodal_processing_runs(*args, **kwargs):
        raise AssertionError(
            "skip_multimodal_processing=True must skip multimodal batch"
        )

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )
    processor._mark_multimodal_processing_complete = (
        fake_mark_multimodal_processing_complete
    )
    processor._process_multimodal_content = fail_if_multimodal_processing_runs

    await processor.insert_content_list(
        [
            {"type": "text", "text": "Text that LightRAG should ingest"},
            {"type": "image", "img_path": "/tmp/diagram.png"},
        ],
        file_path="/docs/report.pdf",
        doc_id="doc-skip-mm",
        split_by_character="\n",
        split_by_character_only=True,
        skip_multimodal_processing=True,
    )

    assert inserted["kwargs"] == {
        "input": "Text that LightRAG should ingest",
        "file_paths": "report.pdf",
        "split_by_character": "\n",
        "split_by_character_only": True,
        "ids": "doc-skip-mm",
    }
    assert marked_doc_ids == ["doc-skip-mm"]
    assert any(
        event_name == "on_text_insert_complete"
        and event["doc_id"] == "doc-skip-mm"
        for event_name, event in processor.callback_manager.events
    )
    assert any(
        event_name == "on_document_complete"
        and event["doc_id"] == "doc-skip-mm"
        for event_name, event in processor.callback_manager.events
    )
