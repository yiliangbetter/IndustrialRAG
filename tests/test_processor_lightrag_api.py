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


class FakeUpsertStorage:
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
        self.text_chunks = FakeUpsertStorage()
        self.chunks_vdb = FakeUpsertStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class FakeCallbackManager:
    def __init__(self):
        self.events = []

    def dispatch(self, event_name, **kwargs):
        self.events.append((event_name, kwargs))


def build_processor(allow_embedding_only_ingestion=False):
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
            "display_content_stats": False,
            "allow_embedding_only_ingestion": allow_embedding_only_ingestion,
            "content_format": "minerU",
        },
    )()
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


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
async def test_processed_text_status_still_runs_pending_multimodal_stage():
    processor = build_processor()
    processor.callback_manager = FakeCallbackManager()
    processor.lightrag.doc_status.records["doc-1"] = {
        "status": DocStatus.PROCESSED,
        "multimodal_processed": False,
        "chunks_list": ["chunk-text"],
        "chunks_count": 1,
    }
    processed_batches = []

    async def fake_process_batch(multimodal_items, file_path, doc_id):
        processed_batches.append(
            {
                "multimodal_items": multimodal_items,
                "file_path": file_path,
                "doc_id": doc_id,
            }
        )

    processor._process_multimodal_content_batch_type_aware = fake_process_batch

    await processor._process_multimodal_content(
        [{"type": "image", "img_path": "/tmp/figure.png"}],
        "report.pdf",
        "doc-1",
    )

    assert processed_batches == [
        {
            "multimodal_items": [{"type": "image", "img_path": "/tmp/figure.png"}],
            "file_path": "report.pdf",
            "doc_id": "doc-1",
        }
    ]
    assert processor.lightrag.doc_status.records["doc-1"][
        "multimodal_processed"
    ] is True
    assert [
        event_name for event_name, _ in processor.callback_manager.events
    ] == ["on_multimodal_start", "on_multimodal_complete"]


@pytest.mark.asyncio
async def test_processed_multimodal_status_skips_reprocessing():
    processor = build_processor()
    processor.callback_manager = FakeCallbackManager()
    processor.lightrag.doc_status.records["doc-1"] = {
        "status": DocStatus.PROCESSED,
        "multimodal_processed": True,
    }
    processed_batches = []

    async def fake_process_batch(multimodal_items, file_path, doc_id):
        processed_batches.append((multimodal_items, file_path, doc_id))

    processor._process_multimodal_content_batch_type_aware = fake_process_batch

    await processor._process_multimodal_content(
        [{"type": "table", "table_body": "a|b"}],
        "report.pdf",
        "doc-1",
    )

    assert processed_batches == []
    assert [
        event_name for event_name, _ in processor.callback_manager.events
    ] == ["on_multimodal_start"]


@pytest.mark.asyncio
async def test_insert_content_list_recovers_mineru_v2_plaintext_for_embedding_only():
    processor = build_processor(allow_embedding_only_ingestion=True)
    processor.callback_manager = FakeCallbackManager()

    nested_mineru_content = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Risk report"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pressure readings"},
                        {"type": "text", "content": "exceeded limits"},
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
                        "item_content": [{"type": "text", "content": "Inspect valve"}],
                    },
                    {"item_content": "Record mitigation"},
                ]
            },
        },
        {"type": "table", "content": {"html": "<table><tr><td>42</td></tr></table>"}},
        {"type": "image", "content": {"image_caption": ["Pump diagram", "labels"]}},
    ]

    await processor.insert_content_list(
        nested_mineru_content,
        file_path="/tmp/risk-report_content_list_v2.json",
        doc_id="doc-mineru",
    )

    chunks = list(processor.lightrag.text_chunks.records.values())
    assert [chunk["content"] for chunk in chunks] == [
        "Risk report",
        "Pressure readings exceeded limits",
        "1. Inspect valve\nRecord mitigation",
        "<table><tr><td>42</td></tr></table>",
        "Pump diagram labels",
    ]
    assert processor.lightrag.chunks_vdb.records == processor.lightrag.text_chunks.records
    doc_status = processor.lightrag.doc_status.records["doc-mineru"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 5
    assert doc_status["file_path"] == "risk-report_content_list_v2.json"
    assert processor.lightrag.insert_done_calls == 1
    assert [
        event_name for event_name, _ in processor.callback_manager.events
    ] == ["on_document_complete"]
