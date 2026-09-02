"""insert_content_list must forward split options into LightRAG text insert.

Content-list ingest is the MinerU JSON → graph path used by batch scripts.
If split_by_character is dropped, manuals chunk on tokens instead of the
intended delimiter and citations no longer line up with plant sections.
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
            "use_full_path": False,
            "allow_embedding_only_ingestion": False,
            "display_content_stats": False,
            "content_format": "minerU",
        },
    )()
    dummy.lightrag = object()
    dummy.mark_calls = []
    dummy.insert_kwargs = []

    async def fake_init():
        return {"success": True}

    async def fake_mark(doc_id):
        dummy.mark_calls.append(doc_id)

    dummy._ensure_lightrag_initialized = fake_init
    dummy._mark_multimodal_processing_complete = fake_mark
    return dummy


@pytest.mark.asyncio
async def test_insert_content_list_forwards_split_kwargs_and_ids(monkeypatch):
    dummy = _dummy()
    import raganything.processor as processor_module

    async def fake_insert(lightrag, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.insert_content_list(
        [{"type": "text", "text": "Section A\nSection B", "page_idx": 0}],
        file_path="/plants/line-a/manual.pdf",
        doc_id="doc-manual",
        split_by_character="\n",
        split_by_character_only=True,
    )

    assert dummy.insert_kwargs == [
        {
            "input": "Section A\nSection B",
            "file_paths": "manual.pdf",
            "split_by_character": "\n",
            "split_by_character_only": True,
            "ids": "doc-manual",
        }
    ]
    assert dummy.mark_calls == ["doc-manual"]


@pytest.mark.asyncio
async def test_insert_content_list_uses_full_path_when_configured(monkeypatch):
    dummy = _dummy()
    dummy.config.use_full_path = True
    import raganything.processor as processor_module

    async def fake_insert(lightrag, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.insert_content_list(
        [{"type": "text", "text": "body", "page_idx": 0}],
        file_path="/plants/line-a/manual.pdf",
        doc_id="doc-manual",
        split_by_character=None,
        split_by_character_only=False,
    )

    assert dummy.insert_kwargs[0]["file_paths"] == "/plants/line-a/manual.pdf"
    assert dummy.insert_kwargs[0]["split_by_character"] is None
    assert dummy.insert_kwargs[0]["split_by_character_only"] is False
