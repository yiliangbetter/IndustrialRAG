"""Processor init must fail closed when LightRAG has not been created."""

import pytest

from raganything.config import RAGAnythingConfig


def test_initialize_processors_raises_without_lightrag(monkeypatch, tmp_path):
    import raganything.raganything as rag_module

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    rag = rag_module.RAGAnything(
        config=RAGAnythingConfig(working_dir=str(tmp_path / "rag_workdir"))
    )
    rag.lightrag = None
    rag.llm_model_func = lambda *args, **kwargs: None

    with pytest.raises(
        ValueError, match="LightRAG instance must be initialized before creating processors"
    ):
        rag._initialize_processors()

    assert rag.modal_processors == {}
