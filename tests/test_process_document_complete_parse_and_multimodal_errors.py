"""document_error glue after parse/multimodal failures in process_document_complete.

Init and text-insert error callbacks are covered elsewhere. If parse_document
or multimodal processing raises after a successful init, scanners must still
see on_document_error with the correct stage and the original exception.
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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_document_error(self, file_path, error="", stage="", doc_id=None, **kw):
        self.events.append(
            {
                "event": "document_error",
                "file_path": file_path,
                "stage": stage,
                "doc_id": doc_id,
                "error": error,
            }
        )

    def on_document_complete(self, file_path, doc_id=None, **kw):
        self.events.append({"event": "document_complete", "doc_id": doc_id})


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
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
    processor.lightrag = object()
    processor.callback_manager = CallbackManager()
    processor.cb = RecordingCallback()
    processor.callback_manager.register(processor.cb)
    return processor


@pytest.mark.asyncio
async def test_parse_document_failure_dispatches_document_error_and_reraises():
    processor = _make_processor()
    boom = RuntimeError("mineru crashed")

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        raise boom

    processor._ensure_lightrag_initialized = fake_init
    processor.parse_document = fake_parse

    with pytest.raises(RuntimeError, match="mineru crashed"):
        await processor.process_document_complete("manual.pdf", doc_id="doc-known")

    errors = [e for e in processor.cb.events if e["event"] == "document_error"]
    assert len(errors) == 1
    assert errors[0]["file_path"] == "manual.pdf"
    assert errors[0]["stage"] == "parse"
    assert errors[0]["doc_id"] == "doc-known"
    assert errors[0]["error"] is boom
    assert not any(e["event"] == "document_complete" for e in processor.cb.events)


@pytest.mark.asyncio
async def test_multimodal_failure_dispatches_document_error_with_multimodal_stage(
    monkeypatch,
):
    processor = _make_processor()
    boom = RuntimeError("figure extract failed")

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        return (
            [
                {"type": "text", "text": "caption nearby"},
                {"type": "image", "img_path": "/abs/fig.png"},
            ],
            "doc-from-content",
        )

    async def fake_mm(*args, **kwargs):
        raise boom

    async def fake_insert(*args, **kwargs):
        return None

    processor._ensure_lightrag_initialized = fake_init
    processor.parse_document = fake_parse
    processor._process_multimodal_content = fake_mm

    import raganything.processor as processor_module

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    with pytest.raises(RuntimeError, match="figure extract failed"):
        await processor.process_document_complete("figures.pdf")

    errors = [e for e in processor.cb.events if e["event"] == "document_error"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "multimodal"
    assert errors[0]["doc_id"] == "doc-from-content"
    assert errors[0]["error"] is boom
    assert not any(e["event"] == "document_complete" for e in processor.cb.events)


@pytest.mark.asyncio
async def test_parse_failure_without_callback_manager_still_raises():
    processor = _make_processor()
    processor.callback_manager = None

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        raise FileNotFoundError("File not found: gone.pdf")

    processor._ensure_lightrag_initialized = fake_init
    processor.parse_document = fake_parse

    with pytest.raises(FileNotFoundError, match="gone.pdf"):
        await processor.process_document_complete("gone.pdf")
