"""RAGAnything context-config mapping used by every multimodal processor.

ContextExtractor internals are covered elsewhere. This locks the field mapping
from RAGAnythingConfig and the fail-closed extractor factory when LightRAG is
missing.
"""

import pytest


def _make_rag(monkeypatch, tmp_path, **config_kwargs):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **kwargs: None)

    defaults = {"working_dir": str(tmp_path / "workdir")}
    defaults.update(config_kwargs)
    return rag_module.RAGAnything(config=RAGAnythingConfig(**defaults))


def test_create_context_config_maps_rag_config_fields(monkeypatch, tmp_path):
    rag = _make_rag(
        monkeypatch,
        tmp_path,
        context_window=3,
        context_mode="chunk",
        max_context_tokens=512,
        include_headers=False,
        include_captions=False,
        context_filter_content_types=["text", "table"],
    )

    ctx = rag._create_context_config()
    assert ctx.context_window == 3
    assert ctx.context_mode == "chunk"
    assert ctx.max_context_tokens == 512
    assert ctx.include_headers is False
    assert ctx.include_captions is False
    assert ctx.filter_content_types == ["text", "table"]


def test_create_context_extractor_requires_lightrag(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    rag.lightrag = None
    with pytest.raises(ValueError, match="LightRAG must be initialized"):
        rag._create_context_extractor()


def test_create_context_extractor_passes_tokenizer(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)

    class FakeTokenizer:
        pass

    tokenizer = FakeTokenizer()
    rag.lightrag = type("FakeLightRAG", (), {"tokenizer": tokenizer})()

    extractor = rag._create_context_extractor()
    assert extractor.tokenizer is tokenizer
    assert extractor.config.context_mode == rag.config.context_mode
