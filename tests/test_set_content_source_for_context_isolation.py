"""RAGAnything.set_content_source_for_context must isolate processor failures.

Caption context is applied to every modal processor. One broken processor
must not skip the rest, and an uninitialized processor map must no-op
instead of crashing ingest/query setup.
"""

from raganything.raganything import RAGAnything


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.debugs = []
        self.infos = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg) % args if args else str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg) % args if args else str(msg))


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def set_content_source(self, content_source, content_format="auto"):
        self.calls.append((content_source, content_format))


class ExplodingProcessor:
    def set_content_source(self, content_source, content_format="auto"):
        raise RuntimeError("processor exploded")


def _dummy():
    dummy = type("DummyRAG", (), {})()
    dummy.logger = FakeLogger()
    dummy.set_content_source_for_context = (
        RAGAnything.set_content_source_for_context.__get__(dummy)
    )
    return dummy


def test_uninitialized_processors_warn_and_do_not_raise():
    dummy = _dummy()
    dummy.modal_processors = {}

    dummy.set_content_source_for_context(
        [{"type": "text", "text": "around the figure"}],
        "minerU",
    )

    assert dummy.modal_processors == {}
    assert any("not initialized" in msg.lower() for msg in dummy.logger.warnings)


def test_processor_error_does_not_block_remaining_processors():
    dummy = _dummy()
    image = ExplodingProcessor()
    table = RecordingProcessor()
    equation = RecordingProcessor()
    dummy.modal_processors = {
        "image": image,
        "table": table,
        "equation": equation,
    }
    source = [{"type": "text", "text": "nameplate context"}]

    dummy.set_content_source_for_context(source, "minerU")

    assert table.calls == [(source, "minerU")]
    assert equation.calls == [(source, "minerU")]
    assert any(
        "image" in msg and "processor exploded" in msg for msg in dummy.logger.errors
    )
    assert any("format: minerU" in msg for msg in dummy.logger.infos)
