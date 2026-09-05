"""Embedding-only process_document_complete must finish and skip LLM/multimodal.

The merged fix for embedding-only ingest added on_document_complete on the
early return. Without that, progress UIs hang after parse. This path is
distinct from insert_content_list embedding-only (#117) and from the
non-embedding process_document_complete orchestration (#136).
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
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
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

        async def _insert_done():
            self.insert_done_calls += 1

        self._insert_done = _insert_done


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.events.append(("document_complete", file_path, doc_id))

    def on_document_error(self, file_path, error="", stage="", doc_id=None, **kwargs):
        self.events.append(
            ("document_error", file_path, stage, doc_id, type(error).__name__)
        )


class DummyProcessor(ProcessorMixin):
    pass


def _processor(tmp_path):
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
    processor.cb = RecordingCallback()
    processor.callback_manager.register(processor.cb)
    processor.parse_calls = []
    processor.mm_calls = []

    async def fake_ensure():
        return {"success": True}

    async def fake_mm(*args, **kwargs):
        processor.mm_calls.append((args, kwargs))

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    return processor


@pytest.mark.asyncio
async def test_embedding_only_indexes_text_skips_multimodal_and_completes(
    tmp_path, monkeypatch
):
    """Parse still runs; images must not use multimodal processors or LightRAG ainsert."""
    import raganything.processor as processor_module

    processor = _processor(tmp_path)
    insert_calls = []

    async def fake_parse(file_path, *args, **kwargs):
        processor.parse_calls.append(file_path)
        return (
            [
                {"type": "text", "text": "Keep this prose", "page_idx": 0},
                {"type": "image", "img_path": "/abs/figure.png", "page_idx": 1},
            ],
            "doc-from-parse",
        )

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor.parse_document = fake_parse
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.process_document_complete(
        "/data/manuals/pump.pdf",
        doc_id="doc-mixed",
        file_name="line-a/pump.pdf",
    )

    assert processor.parse_calls == ["/data/manuals/pump.pdf"]
    assert insert_calls == []
    assert processor.mm_calls == []

    chunks = processor.lightrag.text_chunks.records
    assert len(chunks) == 1
    chunk = next(iter(chunks.values()))
    assert chunk["content"] == "Keep this prose"
    assert chunk["file_path"] == "line-a/pump.pdf"
    assert chunk["full_doc_id"] == "doc-mixed"

    status = processor.lightrag.doc_status.records["doc-mixed"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "line-a/pump.pdf"
    assert processor.lightrag.insert_done_calls == 1
    assert processor.cb.events == [
        ("document_complete", "/data/manuals/pump.pdf", "doc-mixed")
    ]


@pytest.mark.asyncio
async def test_embedding_only_empty_text_still_completes_without_chunks(
    tmp_path, monkeypatch
):
    """Whitespace-only parse output must not hang in PROCESSING or run multimodal."""
    import raganything.processor as processor_module

    processor = _processor(tmp_path)
    insert_calls = []

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "   \n\t  ", "page_idx": 0}], "doc-empty")

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor.parse_document = fake_parse
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.process_document_complete("empty.pdf", doc_id="doc-empty")

    assert insert_calls == []
    assert processor.mm_calls == []
    assert processor.lightrag.text_chunks.records == {}
    assert processor.lightrag.chunks_vdb.records == {}
    assert processor.lightrag.insert_done_calls == 0
    assert processor.cb.events == [("document_complete", "empty.pdf", "doc-empty")]


@pytest.mark.asyncio
async def test_embedding_only_parse_failure_reports_error_and_skips_complete(
    tmp_path, monkeypatch
):
    """Embedding-only does not swallow parse failures; complete must not fire."""
    import raganything.processor as processor_module

    processor = _processor(tmp_path)
    insert_calls = []

    async def fake_parse(*args, **kwargs):
        raise ValueError("empty extract")

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor.parse_document = fake_parse
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    with pytest.raises(ValueError, match="empty extract"):
        await processor.process_document_complete("broken.pdf", doc_id="doc-broken")

    assert insert_calls == []
    assert processor.mm_calls == []
    assert processor.lightrag.text_chunks.records == {}
    kinds = [event[0] for event in processor.cb.events]
    assert kinds == ["document_error"]
    assert processor.cb.events[0][2] == "parse"
    assert processor.cb.events[0][3] == "doc-broken"
