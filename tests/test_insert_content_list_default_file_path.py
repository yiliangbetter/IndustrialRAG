"""Omitting insert_content_list file_path must cite unknown_document.

Graph ingest from pre-parsed JSON often skips file_path. If the default
citation becomes empty or a cwd path, LightRAG stores unusable references
and operators cannot trace chunks back to a source document.
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
async def test_omitted_file_path_cites_unknown_document(monkeypatch):
    import raganything.processor as processor_module

    processor = _make_processor()
    captured = {"inserts": [], "marked": []}

    async def fake_ensure():
        return {"success": True}

    async def fake_insert(lightrag, input, file_paths=None, ids=None, **kwargs):
        captured["inserts"].append(
            {
                "input": input,
                "file_paths": file_paths,
                "ids": ids,
            }
        )

    async def fake_mark(doc_id):
        captured["marked"].append(doc_id)

    processor._ensure_lightrag_initialized = fake_ensure
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [{"type": "text", "text": "Install the pump per section 4.", "page_idx": 0}],
        doc_id="doc-no-path",
    )

    assert len(captured["inserts"]) == 1
    assert captured["inserts"][0]["file_paths"] == "unknown_document"
    assert captured["inserts"][0]["ids"] == "doc-no-path"
    assert captured["inserts"][0]["input"] == "Install the pump per section 4."
    assert captured["marked"] == ["doc-no-path"]
