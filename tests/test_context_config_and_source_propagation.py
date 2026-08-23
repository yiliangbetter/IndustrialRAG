"""Regression tests for context-config and content-source propagation.

Wrong context windows or a missed set_content_source leave multimodal
captions without surrounding text. These helpers are shared by every
processor, so a silent miss has a large blast radius.
"""

import raganything.raganything as rag_module
from raganything.config import RAGAnythingConfig
from raganything.modalprocessors import BaseModalProcessor, ContextExtractor


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeProcessor:
    def __init__(self, name):
        self.name = name
        self.context_extractor = "old-extractor"
        self.content_source = None
        self.content_format = None
        self.set_calls = []

    def set_content_source(self, content_source, content_format="auto"):
        self.content_source = content_source
        self.content_format = content_format
        self.set_calls.append((content_source, content_format))


class ExplodingProcessor(FakeProcessor):
    def set_content_source(self, content_source, content_format="auto"):
        raise RuntimeError(f"{self.name} failed")


class ConcreteProcessor(BaseModalProcessor):
    def __init__(self):
        self.content_source = None
        self.content_format = "auto"
        self.context_extractor = None

    async def process_multimodal_content(self, *args, **kwargs):
        pass


def _make_rag(monkeypatch, tmp_path, **config_overrides):
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module, "get_parser", lambda name: object())

    defaults = {
        "working_dir": str(tmp_path / "rag_workdir"),
        "parser": "mineru",
    }
    defaults.update(config_overrides)
    rag = rag_module.RAGAnything(config=RAGAnythingConfig(**defaults))
    rag.logger = FakeLogger()
    return rag


def test_update_context_config_propagates_new_extractor(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path, context_window=1, context_mode="page")
    rag.lightrag = type("FakeLightRAG", (), {"tokenizer": object()})()
    image = FakeProcessor("image")
    table = FakeProcessor("table")
    rag.modal_processors = {"image": image, "table": table}

    rag.update_context_config(context_window=4, context_mode="chunk")

    assert rag.config.context_window == 4
    assert rag.config.context_mode == "chunk"
    assert isinstance(rag.context_extractor, ContextExtractor)
    assert rag.context_extractor.config.context_window == 4
    assert rag.context_extractor.config.context_mode == "chunk"
    assert image.context_extractor is rag.context_extractor
    assert table.context_extractor is rag.context_extractor


def test_update_context_config_ignores_unknown_keys_without_extractor(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path, context_window=1)
    rag.lightrag = None
    rag.modal_processors = {}

    rag.update_context_config(context_window=9, not_a_real_field=True)

    assert rag.config.context_window == 9
    assert not hasattr(rag.config, "not_a_real_field")
    assert rag.context_extractor is None


def test_set_content_source_for_context_applies_to_all_processors(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    image = FakeProcessor("image")
    table = FakeProcessor("table")
    rag.modal_processors = {"image": image, "table": table}
    source = [{"type": "text", "text": "nearby paragraph"}]

    rag.set_content_source_for_context(source, "minerU")

    assert image.content_source is source
    assert table.content_source is source
    assert image.content_format == "minerU"
    assert table.content_format == "minerU"


def test_set_content_source_isolates_processor_errors(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    bad = ExplodingProcessor("image")
    good = FakeProcessor("table")
    rag.modal_processors = {"image": bad, "table": good}
    source = [{"type": "table", "table_body": "a|b"}]

    rag.set_content_source_for_context(source, "auto")

    assert good.content_source is source
    assert good.content_format == "auto"
    assert bad.content_source is None


def test_set_content_source_noop_when_processors_uninitialized(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    rag.modal_processors = {}

    rag.set_content_source_for_context([{"type": "text", "text": "x"}], "auto")


def test_initialize_processors_respects_feature_flags(monkeypatch, tmp_path):
    captured = []

    class TrackingProcessor:
        def __init__(self, **kwargs):
            captured.append(self.__class__.__name__)
            self.kwargs = kwargs

    class FakeImage(TrackingProcessor):
        pass

    class FakeTable(TrackingProcessor):
        pass

    class FakeEquation(TrackingProcessor):
        pass

    class FakeGeneric(TrackingProcessor):
        pass

    monkeypatch.setattr(rag_module, "ImageModalProcessor", FakeImage)
    monkeypatch.setattr(rag_module, "TableModalProcessor", FakeTable)
    monkeypatch.setattr(rag_module, "EquationModalProcessor", FakeEquation)
    monkeypatch.setattr(rag_module, "GenericModalProcessor", FakeGeneric)

    rag = _make_rag(
        monkeypatch,
        tmp_path,
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=False,
    )
    rag.lightrag = type("FakeLightRAG", (), {"tokenizer": object()})()

    rag._initialize_processors()

    assert "image" not in rag.modal_processors
    assert "equation" not in rag.modal_processors
    assert isinstance(rag.modal_processors["table"], FakeTable)
    assert isinstance(rag.modal_processors["generic"], FakeGeneric)
    assert "FakeImage" not in captured
    assert "FakeEquation" not in captured


def test_get_context_for_item_returns_empty_without_source():
    proc = ConcreteProcessor()
    assert proc._get_context_for_item({"page_idx": 0, "index": 0}) == ""


def test_get_context_for_item_swallows_extractor_errors():
    proc = ConcreteProcessor()
    proc.content_source = [{"type": "text", "text": "ctx"}]
    proc.content_format = "auto"

    class BoomExtractor:
        def extract_context(self, *args, **kwargs):
            raise RuntimeError("extractor failed")

    proc.context_extractor = BoomExtractor()
    assert proc._get_context_for_item({"page_idx": 1}) == ""
