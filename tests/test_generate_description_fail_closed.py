"""Regression: generate_description_only must not soft-succeed on failure.

Previously Image/Table/Equation/GenericModalProcessor.generate_description_only
caught exceptions and returned a 2-tuple fallback. Batch multimodal ingest
treated that as success, stored vision-less chunks, and marked
multimodal_processed=True so retries were skipped.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from raganything.modalprocessors import ImageModalProcessor
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


def _make_image_processor():
    img = ImageModalProcessor.__new__(ImageModalProcessor)
    img.modal_caption_func = AsyncMock()
    img.content_source = None
    img.context_extractor = None
    return img


@pytest.mark.asyncio
async def test_generate_description_only_raises_when_img_path_missing():
    img = _make_image_processor()
    with pytest.raises(ValueError, match="No image path"):
        await img.generate_description_only(
            modal_content={"type": "image", "image_caption": ["fig"]},
            content_type="image",
        )


@pytest.mark.asyncio
async def test_generate_description_only_raises_when_image_file_missing(tmp_path):
    img = _make_image_processor()
    missing = tmp_path / "gone.png"
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        await img.generate_description_only(
            modal_content={"type": "image", "img_path": str(missing)},
            content_type="image",
        )


@pytest.mark.asyncio
async def test_batch_treats_description_exception_as_failure_not_soft_success():
    """Soft fallback used to make N/N succeed; fail-closed must refuse complete."""
    processor = _make_processor()

    class SoftFailThenRaise:
        """Mimics old ImageModalProcessor: would have returned a 2-tuple fallback.

        New contract: raise so batch can detect failure.
        """

        async def generate_description_only(self, **kwargs):
            raise ValueError("No image path provided in modal_content")

    processor.modal_processors = {"image": SoftFailThenRaise()}

    with pytest.raises(RuntimeError, match="No valid multimodal descriptions"):
        await processor._process_multimodal_content_batch_type_aware(
            multimodal_items=[{"type": "image", "image_caption": ["fig"]}],
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


@pytest.mark.asyncio
async def test_process_multimodal_content_reraises_instead_of_two_tuple(tmp_path: Path):
    img = _make_image_processor()
    img.generate_description_only = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("vision failed")
    )
    # Previously the except block returned a 2-tuple and callers unpacking
    # three values crashed with ValueError, which was then swallowed.
    with pytest.raises(RuntimeError, match="vision failed"):
        await img.process_multimodal_content(
            modal_content={"type": "image", "img_path": str(tmp_path / "x.png")},
            content_type="image",
        )
