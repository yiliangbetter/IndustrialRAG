"""Regression: multimodal must not mark complete after zero/partial success."""

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
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": False})()
    processor.modal_processors = {}
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": FakeDocStatusStorage(),
            "max_parallel_insert": 2,
        },
    )()
    processor.lightrag.doc_status.records["doc-1"] = {
        "status": "processed",
        "chunks_count": 0,
        "chunks_list": [],
        "multimodal_processed": False,
    }
    return processor


@pytest.mark.asyncio
async def test_batch_multimodal_raises_when_all_descriptions_fail():
    processor = _make_processor()

    class FailingProcessor:
        async def generate_description_only(self, **kwargs):
            raise RuntimeError("llm down")

    processor.modal_processors = {"image": FailingProcessor()}

    with pytest.raises(RuntimeError, match="No valid multimodal descriptions"):
        await processor._process_multimodal_content_batch_type_aware(
            multimodal_items=[{"type": "image", "img_path": "/tmp/a.png"}],
            file_path="doc.pdf",
            doc_id="doc-1",
        )

    status = await processor.lightrag.doc_status.get_by_id("doc-1")
    assert status.get("multimodal_processed") is False


@pytest.mark.asyncio
async def test_batch_multimodal_raises_on_partial_description_failure():
    processor = _make_processor()
    calls = {"n": 0}

    class FlakyProcessor:
        async def generate_description_only(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "ok", {"entity_name": "ImageA", "entity_type": "image"}
            raise RuntimeError("second item failed")

    processor.modal_processors = {"image": FlakyProcessor()}

    with pytest.raises(RuntimeError, match="incomplete"):
        await processor._process_multimodal_content_batch_type_aware(
            multimodal_items=[
                {"type": "image", "img_path": "/tmp/a.png"},
                {"type": "image", "img_path": "/tmp/b.png"},
            ],
            file_path="doc.pdf",
            doc_id="doc-1",
        )

    status = await processor.lightrag.doc_status.get_by_id("doc-1")
    assert status.get("multimodal_processed") is False


@pytest.mark.asyncio
async def test_process_multimodal_does_not_mark_complete_after_failed_fallback():
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fail_batch(**kwargs):
        raise RuntimeError("batch failed")

    async def fail_individual(*args, **kwargs):
        raise RuntimeError("individual also failed")

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fail_batch
    processor._process_multimodal_content_individual = fail_individual

    with pytest.raises(RuntimeError, match="individual also failed"):
        await processor._process_multimodal_content(
            multimodal_items=[{"type": "image", "img_path": "/tmp/a.png"}],
            file_path="doc.pdf",
            doc_id="doc-1",
        )

    status = await processor.lightrag.doc_status.get_by_id("doc-1")
    assert status.get("multimodal_processed") is not True
