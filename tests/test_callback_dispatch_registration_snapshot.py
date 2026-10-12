"""Callback dispatch must use the registration list from the start of the event.

Handlers unregister and register siblings while a parse or query event is in
flight. Iterating the live list would drop a callback removed mid-dispatch
or deliver the same event to a listener that was just added.
"""

import importlib.util
import sys
from pathlib import Path


def _load_callbacks():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "callbacks.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_callbacks_dispatch_snapshot", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass string annotations look the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dispatch_keeps_the_registration_snapshot():
    callbacks = _load_callbacks()
    manager = callbacks.CallbackManager()
    seen = []

    class First(callbacks.ProcessingCallback):
        def on_parse_start(self, file_path, **kwargs):
            seen.append(("first", file_path))
            manager.unregister(second)
            manager.register(late)

    class Second(callbacks.ProcessingCallback):
        def on_parse_start(self, file_path, **kwargs):
            seen.append(("second", file_path))

    class Late(callbacks.ProcessingCallback):
        def on_parse_start(self, file_path, **kwargs):
            seen.append(("late", file_path))

    first = First()
    second = Second()
    late = Late()
    manager.register(first)
    manager.register(second)
    manager.enable_event_log(True)

    manager.dispatch("on_parse_start", file_path="manual.pdf", parser="mineru")

    assert seen == [("first", "manual.pdf"), ("second", "manual.pdf")]
    assert [event.file_path for event in manager.event_log] == ["manual.pdf"]

    seen.clear()
    manager.dispatch("on_parse_start", file_path="next.pdf")

    assert seen == [("first", "next.pdf"), ("late", "next.pdf")]
    assert [event.file_path for event in manager.event_log] == [
        "manual.pdf",
        "next.pdf",
    ]
