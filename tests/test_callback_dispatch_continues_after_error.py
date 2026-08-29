"""CallbackManager.dispatch must isolate handler failures.

Observability callbacks share a process-wide manager. One raising handler
must not drop later subscribers or skip the event log, or parse/query
metrics silently go missing while ingest continues.
"""

from raganything.callbacks import CallbackManager, ProcessingCallback


class BoomCallback(ProcessingCallback):
    def on_parse_start(self, file_path, **kwargs):
        raise RuntimeError("metrics backend down")


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.starts = []

    def on_parse_start(self, file_path, **kwargs):
        self.starts.append(file_path)


def test_later_callback_still_runs_after_earlier_raises():
    manager = CallbackManager()
    recorder = RecordingCallback()
    manager.register(BoomCallback())
    manager.register(recorder)
    manager.enable_event_log(True)

    manager.dispatch("on_parse_start", file_path="manual.pdf", parser="mineru")

    assert recorder.starts == ["manual.pdf"]
    assert len(manager.event_log) == 1
    assert manager.event_log[0].event_type == "on_parse_start"
    assert manager.event_log[0].file_path == "manual.pdf"


def test_error_in_first_handler_does_not_unregister_siblings():
    manager = CallbackManager()
    first = RecordingCallback()
    second = RecordingCallback()
    manager.register(first)
    manager.register(BoomCallback())
    manager.register(second)

    manager.dispatch("on_parse_start", file_path="a.pdf")
    manager.dispatch("on_parse_start", file_path="b.pdf")

    assert first.starts == ["a.pdf", "b.pdf"]
    assert second.starts == ["a.pdf", "b.pdf"]
