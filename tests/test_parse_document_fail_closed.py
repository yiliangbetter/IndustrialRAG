"""Parse-document fail-closed contracts and citation/id helpers.

Empty extracts and missing files must raise rather than return a processed
doc_id. File references and content-based IDs decide citation and cache identity.
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


class RecordingParseCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_parse_start(self, file_path, **kwargs):
        self.events.append(("start", file_path))

    def on_parse_complete(self, file_path, **kwargs):
        self.events.append(("complete", file_path))

    def on_parse_error(self, file_path, error="", **kwargs):
        self.events.append(("error", file_path, str(error)))


class DummyProcessor(ProcessorMixin):
    pass


def make_parse_processor(tmp_path, parse_method="auto"):
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": parse_method,
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.parse_cache = None
    processor.callback_manager = CallbackManager()
    return processor


@pytest.mark.asyncio
async def test_parse_document_missing_file_raises():
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": "./output",
            "parse_method": "auto",
            "display_content_stats": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.parse_cache = None

    with pytest.raises(FileNotFoundError, match="File not found"):
        await processor.parse_document("/nonexistent/manual.pdf")


@pytest.mark.asyncio
async def test_parse_document_empty_extract_raises(monkeypatch, tmp_path):
    import raganything.processor as processor_module

    class EmptyParser:
        def parse_pdf(self, **kwargs):
            return []

    monkeypatch.setattr(processor_module, "get_parser", lambda name: EmptyParser())

    processor = make_parse_processor(tmp_path)
    callback = RecordingParseCallback()
    processor.callback_manager.register(callback)

    async def fake_store(*args, **kwargs):
        return None

    monkeypatch.setattr(
        DummyProcessor, "_store_cached_result", fake_store, raising=False
    )

    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ValueError, match="No content was extracted"):
        await processor.parse_document(str(pdf))

    kinds = [event[0] for event in callback.events]
    assert "start" in kinds
    assert "complete" not in kinds


@pytest.mark.asyncio
async def test_parse_document_parser_error_dispatches_callback(monkeypatch, tmp_path):
    import raganything.processor as processor_module

    class FailingParser:
        def parse_pdf(self, **kwargs):
            raise RuntimeError("mineru crashed")

    monkeypatch.setattr(processor_module, "get_parser", lambda name: FailingParser())

    processor = make_parse_processor(tmp_path)
    callback = RecordingParseCallback()
    processor.callback_manager.register(callback)

    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(RuntimeError, match="mineru crashed"):
        await processor.parse_document(str(pdf))

    assert any(event[0] == "error" for event in callback.events)
    error_event = next(event for event in callback.events if event[0] == "error")
    assert "mineru crashed" in error_event[2]


def test_get_file_reference_basename_vs_full_path():
    processor = DummyProcessor()
    processor.config = type("Config", (), {"use_full_path": False})()
    assert processor._get_file_reference("/data/uploads/manual.pdf") == "manual.pdf"

    processor.config.use_full_path = True
    assert processor._get_file_reference("/data/uploads/manual.pdf") == (
        "/data/uploads/manual.pdf"
    )
    # Relative paths are returned as-is when use_full_path is on.
    assert processor._get_file_reference("uploads/manual.pdf") == "uploads/manual.pdf"


def test_content_based_doc_id_is_stable_and_content_sensitive():
    processor = DummyProcessor()
    text_blocks = [{"type": "text", "text": "Same body"}]
    image_blocks = [{"type": "image", "img_path": "/abs/a.png"}]
    other_image = [{"type": "image", "img_path": "/abs/b.png"}]
    table_blocks = [{"type": "table", "table_body": "a|b"}]

    first = processor._generate_content_based_doc_id(text_blocks)
    second = processor._generate_content_based_doc_id(text_blocks)
    assert first == second
    assert first.startswith("doc-")

    assert processor._generate_content_based_doc_id(image_blocks) != first
    assert processor._generate_content_based_doc_id(image_blocks) != (
        processor._generate_content_based_doc_id(other_image)
    )
    assert processor._generate_content_based_doc_id(table_blocks) != first
    # Whitespace around text is stripped before hashing.
    assert (
        processor._generate_content_based_doc_id(
            [{"type": "text", "text": "  Same body  "}]
        )
        == first
    )
