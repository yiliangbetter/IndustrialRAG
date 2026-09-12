"""skip_multimodal_processing only applies after LightRAG text insert.

``insert_content_list`` documents this contract: skip is honored only when
non-empty text was inserted. Image-only MinerU lists (no ``type=text`` and no
harvestable ``content`` tree) must still reach multimodal processors even when
callers pass ``skip_multimodal_processing=True``. A regression that skips
unconditionally drops figures from the graph on the default pipeline flag.
"""

from __future__ import annotations

from types import SimpleNamespace

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


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    proc = DummyProcessor()
    proc.logger = FakeLogger()
    proc.callback_manager = None
    proc.lightrag = SimpleNamespace()
    proc.config = SimpleNamespace(
        display_content_stats=False,
        allow_embedding_only_ingestion=False,
        use_full_path=False,
        content_format="minerU",
    )
    return proc


@pytest.mark.asyncio
async def test_skip_multimodal_still_processes_image_only_list(monkeypatch):
    proc = _processor()
    mm_calls = []

    async def fake_init():
        return {"success": True}

    async def fake_insert(*args, **kwargs):
        raise AssertionError("image-only lists must not insert empty text")

    async def fake_mm(items, file_path, doc_id):
        mm_calls.append({"items": items, "file_path": file_path, "doc_id": doc_id})

    async def fake_mark(doc_id):
        raise AssertionError(
            "image-only skip must not mark complete without processors"
        )

    proc._ensure_lightrag_initialized = fake_init
    monkeypatch.setattr("raganything.processor.insert_text_content", fake_insert)
    proc._process_multimodal_content = fake_mm
    proc._mark_multimodal_processing_complete = fake_mark

    await proc.insert_content_list(
        [{"type": "image", "img_path": "/abs/manuals/fig-12.png", "page_idx": 3}],
        file_path="plant-a/manual.pdf",
        skip_multimodal_processing=True,
    )

    assert len(mm_calls) == 1
    assert mm_calls[0]["items"][0]["img_path"] == "/abs/manuals/fig-12.png"
    assert mm_calls[0]["file_path"] == "manual.pdf"
