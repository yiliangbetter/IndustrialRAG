"""skip_multimodal_processing must apply after MinerU v2 plaintext harvest.

Graph ingest (pipeline + content-list batch) defaults to skip_multimodal=True.
MinerU v2 paragraph/title/list/table/image blocks are not type=text, so
separate_content would otherwise send them to multimodal processors. Harvest
fills searchable text first; the skip gate must then run so those blocks are
not double-processed into the graph.
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


MINERU_V2_BLOCKS = [
    {
        "type": "paragraph",
        "content": {"paragraph_content": "Torque spec 25 N·m"},
    },
    {
        "type": "title",
        "content": {"title_content": "Lockout procedure"},
    },
    {
        "type": "list",
        "content": {"list_items": [{"prefix": "1.", "item_content": "Isolate power"}]},
    },
    {
        "type": "table",
        "content": {"html": "<table><tr><td>24V DC</td></tr></table>"},
    },
    {
        "type": "image",
        "content": {"image_caption": ["Figure 2: wiring"]},
    },
]


@pytest.mark.asyncio
async def test_skip_multimodal_after_mineru_v2_harvest_inserts_text_only(
    monkeypatch,
):
    import raganything.processor as processor_module

    processor = _make_processor()
    captured = {"inserts": [], "multimodal": [], "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(lightrag, input, file_paths=None, ids=None, **kwargs):
        captured["inserts"].append(
            {"input": input, "file_paths": file_paths, "ids": ids}
        )

    async def fake_mm(*args, **kwargs):
        captured["multimodal"].append((args, kwargs))

    async def fake_mark(doc_id):
        captured["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        MINERU_V2_BLOCKS,
        file_path="/plants/a/manual.pdf",
        doc_id="doc-harvest-skip",
        skip_multimodal_processing=True,
    )

    assert len(captured["inserts"]) == 1
    harvested = captured["inserts"][0]["input"]
    assert "Torque spec 25 N·m" in harvested
    assert "Lockout procedure" in harvested
    assert "Isolate power" in harvested
    assert "24V DC" in harvested
    assert "Figure 2: wiring" in harvested
    assert captured["inserts"][0]["ids"] == "doc-harvest-skip"
    assert captured["inserts"][0]["file_paths"] == "manual.pdf"
    assert captured["marked"] == ["doc-harvest-skip"]
    assert captured["multimodal"] == []


@pytest.mark.asyncio
async def test_skip_multimodal_after_table_html_harvest_skips_table_processor(
    monkeypatch,
):
    """Table HTML harvest is enough text for the skip gate; table modal must not run."""
    import raganything.processor as processor_module

    processor = _make_processor()
    captured = {"inserts": [], "multimodal": 0, "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(lightrag, input, **kwargs):
        captured["inserts"].append(input)

    async def fake_mm(*args, **kwargs):
        captured["multimodal"] += 1

    async def fake_mark(doc_id):
        captured["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [
            {
                "type": "table",
                "content": {"html": "<table><tr><td>48V</td></tr></table>"},
            }
        ],
        file_path="table-only.json",
        doc_id="doc-table-skip",
        skip_multimodal_processing=True,
    )

    assert captured["inserts"] == ["<table><tr><td>48V</td></tr></table>"]
    assert captured["multimodal"] == 0
    assert captured["marked"] == ["doc-table-skip"]
