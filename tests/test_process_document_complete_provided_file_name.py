"""process_document_complete must honor caller-provided file_name for citations.

Folder batch remaps nested files to relative paths (line-a/child.pdf). If the
orchestrator ignores file_name and cites basename only, retrieval points at
the wrong document. Parser kwargs (lang) and split_by_character must also
reach parse/insert rather than being dropped.
"""

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

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=False,
        use_full_path=False,
        parser="mineru",
        parse_method="auto",
        parser_output_dir="./output",
        display_content_stats=False,
        content_format="minerU",
    )
    processor.lightrag = object()
    return processor


@pytest.mark.asyncio
async def test_provided_file_name_is_used_for_text_insert_citation(monkeypatch):
    processor = _processor()
    parse_kwargs = {}
    insert_kwargs = {}
    marked = []

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(file_path, output_dir, parse_method, display_stats, **kwargs):
        parse_kwargs["file_path"] = file_path
        parse_kwargs["parse_method"] = parse_method
        parse_kwargs.update(kwargs)
        return ([{"type": "text", "text": "section 3.1", "page_idx": 0}], "doc-nested")

    async def fake_insert(lightrag, input=None, **kwargs):
        insert_kwargs["input"] = input
        insert_kwargs.update(kwargs)

    async def fake_mark(doc_id):
        marked.append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr("raganything.processor.insert_text_content", fake_insert)

    await processor.process_document_complete(
        "/abs/line-a/child.pdf",
        file_name="line-a/child.pdf",
        parse_method="ocr",
        split_by_character="\n",
        split_by_character_only=True,
        lang="ch",
    )

    assert parse_kwargs["file_path"] == "/abs/line-a/child.pdf"
    assert parse_kwargs["parse_method"] == "ocr"
    assert parse_kwargs["lang"] == "ch"
    assert insert_kwargs["file_paths"] == "line-a/child.pdf"
    assert insert_kwargs["input"] == "section 3.1"
    assert insert_kwargs["ids"] == "doc-nested"
    assert insert_kwargs["split_by_character"] == "\n"
    assert insert_kwargs["split_by_character_only"] is True
    assert marked == ["doc-nested"]
