"""Batch multimodal processing with zero valid descriptions.

If every concurrent description task returns None (missing processor or
caption errors), production stores no multimodal chunks but still marks
the document's multimodal stage complete so ingest does not hang.
"""

import pytest

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
    def __init__(self, records=None):
        self.records = records or {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class RaisingProcessor:
    async def generate_description_only(self, **kwargs):
        raise RuntimeError("caption model failed")


def _make_processor(modal_processors, doc_id="doc-mm"):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": False})()
    processor.modal_processors = modal_processors
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": FakeDocStatusStorage(
                {
                    doc_id: {
                        "status": "processed",
                        "multimodal_processed": False,
                        "chunks_count": 3,
                    }
                }
            ),
            "max_parallel_insert": 2,
        },
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_missing_processor_stores_no_chunks_but_marks_complete():
    processor = _make_processor({})
    store_calls = []

    async def fake_store_chunks(chunks):
        store_calls.append(chunks)

    processor._store_chunks_to_lightrag_storage_type_aware = fake_store_chunks

    await processor._process_multimodal_content(
        [{"type": "image", "img_path": "/tmp/fig.png"}],
        file_path="manual.pdf",
        doc_id="doc-mm",
    )

    assert store_calls == []
    status = processor.lightrag.doc_status.records["doc-mm"]
    assert status["multimodal_processed"] is True
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_all_description_errors_store_no_chunks_but_mark_complete():
    processor = _make_processor({"image": RaisingProcessor()})
    store_calls = []
    entity_store_calls = []

    async def fake_store_chunks(chunks):
        store_calls.append(chunks)

    async def fake_store_entities(*args, **kwargs):
        entity_store_calls.append(args)

    processor._store_chunks_to_lightrag_storage_type_aware = fake_store_chunks
    processor._store_multimodal_main_entities = fake_store_entities

    await processor._process_multimodal_content(
        [
            {"type": "image", "img_path": "/tmp/fig1.png"},
            {"type": "image", "img_path": "/tmp/fig2.png"},
        ],
        file_path="manual.pdf",
        doc_id="doc-mm",
    )

    assert store_calls == []
    assert entity_store_calls == []
    status = processor.lightrag.doc_status.records["doc-mm"]
    assert status["multimodal_processed"] is True
