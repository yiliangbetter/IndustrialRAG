"""Tertiary text recovery on insert_content_list.

When separate_content() finds no type=text blocks and MinerU v2 harvest
yields nothing, production still scans each block's top-level "text"
string. That path is the only searchable text for custom/algorithm-style
blocks and is easy to drop without a regression test.
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

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_algorithm_block_text_is_inserted(monkeypatch):
    processor = _make_processor()
    inserted = {}
    multimodal_calls = []

    async def fake_insert_text_content(lightrag, **kwargs):
        inserted["kwargs"] = kwargs

    async def fake_process_multimodal_content(items, file_ref, doc_id):
        multimodal_calls.append((items, file_ref, doc_id))

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert_text_content
    )
    processor._process_multimodal_content = fake_process_multimodal_content

    await processor.insert_content_list(
        [
            {"type": "algorithm", "text": "PID tuning steps"},
        ],
        file_path="/docs/manual.pdf",
        doc_id="doc-algorithm",
    )

    assert inserted["kwargs"]["input"] == "PID tuning steps"
    assert inserted["kwargs"]["ids"] == "doc-algorithm"
    assert inserted["kwargs"]["file_paths"] == "manual.pdf"
    assert len(multimodal_calls) == 1
    assert multimodal_calls[0][0][0]["type"] == "algorithm"


@pytest.mark.asyncio
async def test_multiple_text_fields_are_joined(monkeypatch):
    processor = _make_processor()
    inserted = {}

    async def fake_insert_text_content(lightrag, **kwargs):
        inserted["kwargs"] = kwargs

    async def fake_process_multimodal_content(items, file_ref, doc_id):
        return None

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert_text_content
    )
    processor._process_multimodal_content = fake_process_multimodal_content

    await processor.insert_content_list(
        [
            {"type": "algorithm", "text": "Step 1: isolate the loop"},
            {"type": "custom", "text": "Step 2: set Ki to zero"},
            {"type": "custom", "text": "   "},
            {"type": "custom", "text": 42},
        ],
        file_path="manual.pdf",
        doc_id="doc-joined",
    )

    assert inserted["kwargs"]["input"] == (
        "Step 1: isolate the loop\n\nStep 2: set Ki to zero"
    )


@pytest.mark.asyncio
async def test_blank_text_field_does_not_insert(monkeypatch):
    processor = _make_processor()
    insert_calls = []

    async def fake_insert_text_content(lightrag, **kwargs):
        insert_calls.append(kwargs)

    async def fake_process_multimodal_content(items, file_ref, doc_id):
        return None

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert_text_content
    )
    processor._process_multimodal_content = fake_process_multimodal_content

    await processor.insert_content_list(
        [
            {"type": "algorithm", "text": "\n\t  "},
        ],
        file_path="manual.pdf",
        doc_id="doc-blank",
    )

    assert insert_calls == []
