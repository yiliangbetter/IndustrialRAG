"""Ingest must attach the full content list as caption context.

Image/table descriptions pull nearby procedure text via
set_content_source_for_context. If process_document_complete or
insert_content_list skip that call, captions are generated without the
surrounding manual pages even though the parse succeeded.
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


def _dummy():
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": "./output",
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
        },
    )()
    dummy.lightrag = object()
    dummy.source_calls = []
    dummy.mm_calls = []
    dummy.mark_calls = []
    dummy.insert_kwargs = []

    def set_source(content_source, content_format="auto"):
        dummy.source_calls.append((list(content_source), content_format))

    async def fake_ensure():
        return {"success": True}

    async def fake_mm(items, file_path, doc_id):
        dummy.mm_calls.append((list(items), file_path, doc_id))

    async def fake_mark(doc_id):
        dummy.mark_calls.append(doc_id)

    dummy.set_content_source_for_context = set_source
    dummy._ensure_lightrag_initialized = fake_ensure
    dummy._process_multimodal_content = fake_mm
    dummy._mark_multimodal_processing_complete = fake_mark
    return dummy


@pytest.mark.asyncio
async def test_process_document_complete_sets_source_when_multimodal(monkeypatch):
    dummy = _dummy()
    content_list = [
        {"type": "text", "text": "Step 1 loosen the cover"},
        {"type": "image", "img_path": "/abs/fig.png", "page_idx": 1},
    ]

    async def fake_parse(*args, **kwargs):
        return content_list, "doc-1"

    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete("manual.pdf")

    assert dummy.source_calls == [(content_list, "minerU")]
    assert dummy.mm_calls[0][0] == [content_list[1]]
    assert dummy.insert_kwargs[0]["ids"] == "doc-1"


@pytest.mark.asyncio
async def test_process_document_complete_skips_source_for_text_only(monkeypatch):
    dummy = _dummy()

    async def fake_parse(*args, **kwargs):
        return ([{"type": "text", "text": "procedure only"}], "doc-text")

    dummy.parse_document = fake_parse

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete("manual.pdf")

    assert dummy.source_calls == []
    assert dummy.mm_calls == []
    assert dummy.mark_calls == ["doc-text"]


@pytest.mark.asyncio
async def test_process_document_complete_embedding_only_skips_source(monkeypatch):
    dummy = _dummy()
    dummy.config.allow_embedding_only_ingestion = True
    embed_calls = []

    async def fake_parse(*args, **kwargs):
        return (
            [
                {"type": "text", "text": "body"},
                {"type": "image", "img_path": "/abs/fig.png"},
            ],
            "doc-embed",
        )

    async def fake_embed(*, text_content, file_ref, doc_id):
        embed_calls.append((text_content, file_ref, doc_id))

    dummy.parse_document = fake_parse
    dummy._insert_text_content_embedding_only = fake_embed

    await dummy.process_document_complete("manual.pdf")

    assert dummy.source_calls == []
    assert dummy.mm_calls == []
    assert embed_calls[0][2] == "doc-embed"


@pytest.mark.asyncio
async def test_insert_content_list_sets_source_when_multimodal(monkeypatch):
    dummy = _dummy()
    content_list = [
        {"type": "text", "text": "torque spec 12 Nm"},
        {"type": "table", "table_body": "a|b", "page_idx": 2},
    ]

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.insert_content_list(content_list, file_path="manual.pdf")

    assert dummy.source_calls == [(content_list, "minerU")]
    assert dummy.mm_calls[0][0] == [content_list[1]]
    assert dummy.insert_kwargs[0]["file_paths"] == "manual.pdf"


@pytest.mark.asyncio
async def test_insert_content_list_skips_source_for_text_only(monkeypatch):
    dummy = _dummy()

    import raganything.processor as processor_module

    async def fake_insert(*args, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.insert_content_list(
        [{"type": "text", "text": "text only"}],
        file_path="manual.pdf",
        doc_id="doc-text",
    )

    assert dummy.source_calls == []
    assert dummy.mm_calls == []
    assert dummy.mark_calls == ["doc-text"]
