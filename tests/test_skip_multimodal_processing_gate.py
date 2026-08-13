"""skip_multimodal_processing only skips after a successful text insert.

If the content list has no harvestable text, multimodal items still run.
A regression that always skips (or never skips) would drop tables/images or
double-process them into the graph.
"""

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
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


class RecordingDocumentCallback(ProcessingCallback):
    def __init__(self):
        self.document_completes = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.document_completes.append((file_path, doc_id))


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(tmp_path):
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
            "display_content_stats": False,
            "parse_method": "auto",
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "use_full_path": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.lightrag = object()
    processor.callback_manager = CallbackManager()
    processor.parse_cache = None

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_skip_multimodal_after_text_insert_marks_complete(tmp_path, monkeypatch):
    import raganything.processor as processor_module

    processor = make_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    mm_calls = []
    mark_calls = []
    insert_calls = []

    async def fake_mm(*args, **kwargs):
        mm_calls.append((args, kwargs))

    async def fake_mark(doc_id):
        mark_calls.append(doc_id)

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [
            {"type": "text", "text": "Prose body"},
            {"type": "image", "img_path": "/abs/figure.png"},
            {"type": "table", "table_body": "col|val"},
        ],
        file_path="mixed.pdf",
        doc_id="doc-skip",
        skip_multimodal_processing=True,
    )

    assert len(insert_calls) == 1
    assert insert_calls[0]["input"] == "Prose body"
    assert insert_calls[0]["ids"] == "doc-skip"
    assert insert_calls[0]["file_paths"] == "mixed.pdf"
    assert mark_calls == ["doc-skip"]
    assert mm_calls == []
    assert callback.document_completes == [("mixed.pdf", "doc-skip")]


@pytest.mark.asyncio
async def test_skip_multimodal_without_text_still_processes_multimodal(
    tmp_path, monkeypatch
):
    """No text insert means the skip gate is not taken; images still process."""
    import raganything.processor as processor_module

    processor = make_processor(tmp_path)
    mm_calls = []
    mark_calls = []
    insert_calls = []

    async def fake_mm(*args, **kwargs):
        mm_calls.append((args, kwargs))

    async def fake_mark(doc_id):
        mark_calls.append(doc_id)

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor._process_multimodal_content = fake_mm
    processor._mark_multimodal_processing_complete = fake_mark
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [{"type": "image", "img_path": "/abs/figure.png"}],
        file_path="figure-only.pdf",
        doc_id="doc-figure",
        skip_multimodal_processing=True,
    )

    assert insert_calls == []
    assert mark_calls == []
    assert len(mm_calls) == 1
    items, file_ref, doc_id = mm_calls[0][0]
    assert items == [{"type": "image", "img_path": "/abs/figure.png"}]
    assert file_ref == "figure-only.pdf"
    assert doc_id == "doc-figure"
