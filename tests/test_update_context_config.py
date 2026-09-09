"""RAGAnything.update_context_config must retune extractors without dropping processors.

Live caption windows are changed through this API. Unknown keys must not be
setattr'd onto config, missing processors must still persist known fields, and
extractor rebuild failures must not raise out to the caller.
"""

from types import SimpleNamespace

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


def _dummy():
    dummy = type("DummyRAG", (), {})()
    dummy.logger = FakeLogger()
    dummy.config = SimpleNamespace(
        context_window=1,
        context_mode="page",
        max_context_tokens=2000,
        include_headers=True,
        include_captions=True,
    )
    dummy.lightrag = None
    dummy.modal_processors = {}
    dummy.context_extractor = None
    dummy.update_context_config = RAGAnything.update_context_config.__get__(dummy)
    dummy._create_context_extractor = lambda: SimpleNamespace(name="new-extractor")
    dummy._create_context_config = lambda: "cfg"
    return dummy


def test_known_keys_update_config_and_unknown_keys_are_ignored():
    dummy = _dummy()

    dummy.update_context_config(
        context_window=3,
        context_mode="chunk",
        not_a_real_key="danger",
    )

    assert dummy.config.context_window == 3
    assert dummy.config.context_mode == "chunk"
    assert dummy.config.max_context_tokens == 2000
    assert not hasattr(dummy.config, "not_a_real_key")
    assert any(
        "Unknown context config parameter" in msg for msg in dummy.logger.warnings
    )


def test_missing_processors_updates_config_without_rebuilding_extractor():
    dummy = _dummy()
    dummy.lightrag = object()
    dummy.modal_processors = {}
    calls = []

    def boom():
        calls.append("created")
        raise AssertionError("extractor should not be rebuilt")

    dummy._create_context_extractor = boom

    dummy.update_context_config(max_context_tokens=512)

    assert dummy.config.max_context_tokens == 512
    assert calls == []
    assert dummy.context_extractor is None


def test_rebuilds_extractor_and_assigns_it_to_all_processors():
    dummy = _dummy()
    dummy.lightrag = object()
    old_extractor = object()
    image = SimpleNamespace(context_extractor=old_extractor)
    table = SimpleNamespace(context_extractor=old_extractor)
    dummy.modal_processors = {"image": image, "table": table}
    new_extractor = SimpleNamespace(name="rebuilt")
    dummy._create_context_extractor = lambda: new_extractor

    dummy.update_context_config(include_captions=False)

    assert dummy.config.include_captions is False
    assert dummy.context_extractor is new_extractor
    assert image.context_extractor is new_extractor
    assert table.context_extractor is new_extractor
    assert any("applied to all processors" in msg for msg in dummy.logger.infos)


def test_extractor_rebuild_failure_is_logged_and_does_not_raise():
    dummy = _dummy()
    dummy.lightrag = object()
    old_extractor = object()
    image = SimpleNamespace(context_extractor=old_extractor)
    dummy.modal_processors = {"image": image}

    def boom():
        raise RuntimeError("tokenizer missing")

    dummy._create_context_extractor = boom

    dummy.update_context_config(context_window=4)

    assert dummy.config.context_window == 4
    assert image.context_extractor is old_extractor
    assert dummy.context_extractor is None
    assert any("tokenizer missing" in msg for msg in dummy.logger.errors)
