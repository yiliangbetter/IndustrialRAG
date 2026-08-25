"""Processor registration must honor feature flags and isolate source wiring.

Disabled image/table/equation processors must not be constructed. One
failing set_content_source must not skip the remaining processors.
get_config_info must not leak callables or model kwargs.
"""

import pytest

from raganything.config import RAGAnythingConfig


class FakeProcessor:
    def __init__(self, lightrag=None, modal_caption_func=None, context_extractor=None):
        self.lightrag = lightrag
        self.modal_caption_func = modal_caption_func
        self.context_extractor = context_extractor
        self.content_source = None
        self.fail_source = False

    def set_content_source(self, content_source, content_format="auto"):
        if self.fail_source:
            raise RuntimeError("source wiring failed")
        self.content_source = (content_source, content_format)


class FakeImageProcessor(FakeProcessor):
    pass


class FakeTableProcessor(FakeProcessor):
    pass


class FakeEquationProcessor(FakeProcessor):
    pass


class FakeGenericProcessor(FakeProcessor):
    pass


def _make_rag(monkeypatch, tmp_path, **config_kwargs):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module, "ImageModalProcessor", FakeImageProcessor)
    monkeypatch.setattr(rag_module, "TableModalProcessor", FakeTableProcessor)
    monkeypatch.setattr(rag_module, "EquationModalProcessor", FakeEquationProcessor)
    monkeypatch.setattr(rag_module, "GenericModalProcessor", FakeGenericProcessor)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        **config_kwargs,
    )
    rag = rag_module.RAGAnything(
        config=config,
        llm_model_func=lambda *a, **k: "llm",
        vision_model_func=lambda *a, **k: "vision",
        embedding_func=lambda *a, **k: [0.0],
    )
    rag.lightrag = type("FakeLightRAG", (), {"tokenizer": object()})()
    return rag


def test_disabled_image_processing_omits_image_processor(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path, enable_image_processing=False)

    rag._initialize_processors()

    assert "image" not in rag.modal_processors
    assert isinstance(rag.modal_processors["table"], FakeTableProcessor)
    assert isinstance(rag.modal_processors["equation"], FakeEquationProcessor)
    assert isinstance(rag.modal_processors["generic"], FakeGenericProcessor)


def test_image_processor_prefers_vision_model_func(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)

    rag._initialize_processors()

    image = rag.modal_processors["image"]
    table = rag.modal_processors["table"]
    assert image.modal_caption_func is rag.vision_model_func
    assert table.modal_caption_func is rag.llm_model_func
    assert rag.modal_processors["generic"].modal_caption_func is rag.llm_model_func


def test_image_processor_falls_back_to_llm_without_vision(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    rag.vision_model_func = None

    rag._initialize_processors()

    assert rag.modal_processors["image"].modal_caption_func is rag.llm_model_func


def test_embedding_only_init_only_registers_generic_processor(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path, allow_embedding_only_ingestion=True)

    assert rag.config.enable_image_processing is False
    assert rag.config.enable_table_processing is False
    assert rag.config.enable_equation_processing is False

    rag._initialize_processors()

    assert set(rag.modal_processors) == {"generic"}
    assert isinstance(rag.modal_processors["generic"], FakeGenericProcessor)


def test_set_content_source_continues_after_one_processor_fails(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    rag._initialize_processors()
    rag.modal_processors["image"].fail_source = True
    source = [{"type": "text", "text": "intro", "page_idx": 0}]

    rag.set_content_source_for_context(source, "minerU")

    assert rag.modal_processors["image"].content_source is None
    assert rag.modal_processors["table"].content_source == (source, "minerU")
    assert rag.modal_processors["equation"].content_source == (source, "minerU")
    assert rag.modal_processors["generic"].content_source == (source, "minerU")


def test_get_config_info_redacts_callables_and_model_kwargs(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    rag.lightrag_kwargs = {
        "chunk_token_size": 512,
        "llm_model_func": lambda prompt: prompt,
        "llm_model_kwargs": {"api_key": "secret-key"},
        "vector_db_storage_cls_kwargs": {"password": "secret"},
    }

    info = rag.get_config_info()
    custom = info["lightrag_config"]["custom_parameters"]

    assert custom["chunk_token_size"] == 512
    assert "llm_model_func" not in custom
    assert "llm_model_kwargs" not in custom
    assert "vector_db_storage_cls_kwargs" not in custom
    assert "secret-key" not in str(info)
    assert "secret" not in str(custom)
