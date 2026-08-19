"""Regression tests for ProcessorMixin.insert_content_list (graph ingest path).

Industrial pipelines ingest MinerU *_content_list_v2.json without re-parsing.
These tests lock nested-list flattening, init fail-closed, plaintext harvest,
and the graph insert vs multimodal/mark-complete branches. Embedding-only and
skip_multimodal=True are covered by a separate open coverage PR.
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


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "display_content_stats": False,
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
        },
    )()
    processor.lightrag = object()
    return processor


@pytest.mark.asyncio
async def test_insert_content_list_raises_when_lightrag_init_fails():
    processor = _make_processor()

    async def fake_ensure():
        return {"success": False, "error": "missing embedding_func"}

    processor._ensure_lightrag_initialized = fake_ensure

    with pytest.raises(RuntimeError, match="missing embedding_func"):
        await processor.insert_content_list(
            [{"type": "text", "text": "hello", "page_idx": 0}],
            file_path="manual.pdf",
        )


@pytest.mark.asyncio
async def test_insert_content_list_flattens_nested_mineru_wrappers(monkeypatch):
    """MinerU v2 JSON is often [[block, block]]. Without flattening, item.get crashes."""
    import raganything.processor as processor_module

    processor = _make_processor()
    captured = {"inserts": [], "multimodal": [], "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(lightrag, input, file_paths=None, ids=None, **kwargs):
        captured["inserts"].append(
            {"input": input, "file_paths": file_paths, "ids": ids}
        )

    async def fake_mm(items, file_ref, doc_id):
        captured["multimodal"].append((items, file_ref, doc_id))

    async def fake_mark(doc_id):
        captured["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    nested = [
        [
            {"type": "text", "text": "from nested wrapper", "page_idx": 0},
            {"type": "image", "img_path": "/abs/fig.png", "page_idx": 1},
        ]
    ]

    await processor.insert_content_list(
        nested, file_path="/plants/a/manual.pdf", doc_id="doc-nested"
    )

    assert len(captured["inserts"]) == 1
    assert captured["inserts"][0]["input"] == "from nested wrapper"
    assert captured["inserts"][0]["file_paths"] == "manual.pdf"
    assert captured["inserts"][0]["ids"] == "doc-nested"
    assert len(captured["multimodal"]) == 1
    assert captured["multimodal"][0][0] == [
        {"type": "image", "img_path": "/abs/fig.png", "page_idx": 1}
    ]
    assert captured["marked"] == []


@pytest.mark.asyncio
async def test_insert_content_list_marks_complete_when_text_only(monkeypatch):
    import raganything.processor as processor_module

    processor = _make_processor()
    captured = {"multimodal": 0, "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(*args, **kwargs):
        return None

    async def fake_mm(*args, **kwargs):
        captured["multimodal"] += 1

    async def fake_mark(doc_id):
        captured["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [{"type": "text", "text": "only text", "page_idx": 0}],
        file_path="plain.pdf",
        doc_id="doc-text",
    )

    assert captured["multimodal"] == 0
    assert captured["marked"] == ["doc-text"]


@pytest.mark.asyncio
async def test_insert_content_list_harvests_mineru_paragraph_plaintext(monkeypatch):
    """Blocks that are not type=text still yield searchable text via MinerU harvest."""
    import raganything.processor as processor_module

    processor = _make_processor()
    captured = {"inserts": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(lightrag, input, **kwargs):
        captured["inserts"].append(input)

    async def fake_mm(*args, **kwargs):
        return None

    async def fake_mark(doc_id):
        return None

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [
            {
                "type": "paragraph",
                "content": {"paragraph_content": "Torque spec 25 N·m"},
            },
            {
                "type": "title",
                "content": {"title_content": "Maintenance"},
            },
        ],
        file_path="ocr.json",
        doc_id="doc-harvest",
    )

    assert len(captured["inserts"]) == 1
    assert "Torque spec 25 N·m" in captured["inserts"][0]
    assert "Maintenance" in captured["inserts"][0]
