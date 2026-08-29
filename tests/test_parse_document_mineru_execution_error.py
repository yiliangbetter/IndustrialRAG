"""parse_document must re-raise MineruExecutionError after on_parse_error.

MinerU CLI failures use a typed exception with return_code/error_msg. Swallowing
it would let ingest continue with an empty extract; catching it only as a generic
Exception would still fail closed but drop the typed callback payload operators
use to distinguish CLI crashes from missing files.
"""

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
from raganything.parser import MineruExecutionError
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
        self.events.append(("error", file_path, error))


class DummyProcessor(ProcessorMixin):
    pass


class FailingMineruParser:
    def parse_pdf(self, **kwargs):
        raise MineruExecutionError(1, ["cuda oom", "page 3 failed"])


@pytest.mark.asyncio
async def test_mineru_execution_error_dispatches_and_reraises(tmp_path):
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.parse_cache = None
    processor.doc_parser = FailingMineruParser()
    processor.callback_manager = CallbackManager()
    callback = RecordingParseCallback()
    processor.callback_manager.register(callback)

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(MineruExecutionError, match="return code 1") as exc_info:
        await processor.parse_document(str(pdf))

    assert exc_info.value.return_code == 1
    assert exc_info.value.error_msg == ["cuda oom", "page 3 failed"]
    kinds = [event[0] for event in callback.events]
    assert kinds == ["start", "error"]
    error_event = callback.events[1]
    assert error_event[2] is exc_info.value
