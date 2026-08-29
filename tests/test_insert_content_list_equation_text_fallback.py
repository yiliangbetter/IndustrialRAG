"""insert_content_list must recover equation/generic `text` when harvest is empty.

separate_content only keeps type=text blocks. MinerU harvest covers
paragraph/title/list/table/image. Industrial manuals often store formula
descriptions on type=equation items (`text` + `latex`). Dropping that
fallback leaves equations unsearchable except via multimodal LLM.
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


def _patch_hooks(processor, monkeypatch, captured):
    import raganything.processor as processor_module

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(lightrag, input, file_paths=None, ids=None, **kwargs):
        captured["inserts"].append(
            {"input": input, "file_paths": file_paths, "ids": ids}
        )

    async def fake_mm(items, file_ref, doc_id):
        captured["multimodal"].append((list(items), file_ref, doc_id))

    async def fake_mark(doc_id):
        captured["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)


@pytest.mark.asyncio
async def test_equation_text_is_inserted_when_harvest_yields_nothing(monkeypatch):
    processor = _make_processor()
    captured = {"inserts": [], "multimodal": [], "marked": []}
    _patch_hooks(processor, monkeypatch, captured)

    equation = {
        "type": "equation",
        "latex": "V = IR",
        "text": "Ohm's law relating voltage and current",
        "page_idx": 3,
    }

    await processor.insert_content_list(
        [equation],
        file_path="/plants/a/electrical.pdf",
        doc_id="doc-eq",
    )

    assert len(captured["inserts"]) == 1
    assert captured["inserts"][0]["input"] == "Ohm's law relating voltage and current"
    assert captured["inserts"][0]["file_paths"] == "electrical.pdf"
    assert captured["inserts"][0]["ids"] == "doc-eq"
    # Equation remains multimodal so the latex still goes through processors.
    assert captured["multimodal"] == [([equation], "electrical.pdf", "doc-eq")]
    assert captured["marked"] == []


@pytest.mark.asyncio
async def test_generic_text_fields_are_joined_in_list_order(monkeypatch):
    processor = _make_processor()
    captured = {"inserts": [], "multimodal": [], "marked": []}
    _patch_hooks(processor, monkeypatch, captured)

    await processor.insert_content_list(
        [
            {"type": "chart", "text": "Pressure trend 2024", "page_idx": 1},
            {
                "type": "equation",
                "text": "   ",
                "latex": "P=F/A",
            },  # whitespace-only skipped
            {"type": "code", "text": "def torque(): ...", "page_idx": 2},
            {"type": "chart", "text": ["not a string"], "page_idx": 3},
        ],
        file_path="notes.pdf",
        doc_id="doc-generic",
    )

    assert captured["inserts"][0]["input"] == "Pressure trend 2024\n\ndef torque(): ..."
    assert len(captured["multimodal"][0][0]) == 4


@pytest.mark.asyncio
async def test_equation_without_text_skips_insert_and_still_processes_multimodal(
    monkeypatch,
):
    processor = _make_processor()
    captured = {"inserts": [], "multimodal": [], "marked": []}
    _patch_hooks(processor, monkeypatch, captured)

    equation = {"type": "equation", "latex": "E=mc^2", "page_idx": 0}

    await processor.insert_content_list(
        [equation],
        file_path="physics.pdf",
        doc_id="doc-latex-only",
    )

    assert captured["inserts"] == []
    assert captured["multimodal"] == [([equation], "physics.pdf", "doc-latex-only")]
    assert captured["marked"] == []
