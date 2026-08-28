"""RAGAnything.get_processor_info must not depend on real parser installs.

Installation probes are stubbed so CI stays deterministic. This covers the
uninitialized vs initialized status split and model-presence flags used by
ops/debug surfaces.
"""

import pytest


class InstantInstallParser:
    def check_installation(self):
        return True


class NeverInstallParser:
    def check_installation(self):
        return False


def _make_rag(monkeypatch, tmp_path, *, vision=False, llm=True, embedding=False):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    monkeypatch.setattr(rag_module, "MineruParser", InstantInstallParser)
    monkeypatch.setattr(
        rag_module,
        "SUPPORTED_PARSERS",
        ("mineru", "docling", "paddleocr"),
    )

    def fake_get_parser(name):
        if name == "paddleocr":
            return NeverInstallParser()
        return InstantInstallParser()

    monkeypatch.setattr(rag_module, "get_parser", fake_get_parser)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **kwargs: None)

    rag = rag_module.RAGAnything(
        config=RAGAnythingConfig(working_dir=str(tmp_path / "workdir")),
        llm_model_func=(lambda *a, **k: "ok") if llm else None,
        vision_model_func=(lambda *a, **k: "ok") if vision else None,
        embedding_func=(lambda *a, **k: None) if embedding else None,
    )
    return rag


def test_uninitialized_status_and_parser_install_map(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path, vision=False, embedding=True)
    info = rag.get_processor_info()

    assert info["status"] == "Not initialized"
    assert info["processors"] == {}
    assert info["mineru_installed"] is True
    assert info["parser_installation"]["mineru"] is True
    assert info["parser_installation"]["docling"] is True
    assert info["parser_installation"]["paddleocr"] is False
    assert info["models"]["llm_model"] == "External function"
    assert info["models"]["vision_model"] == "Not provided"
    assert info["models"]["embedding_model"] == "External function"
    assert "config" in info


def test_initialized_lists_enabled_processor_classes(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path, vision=True, llm=True)

    class ImageProc:
        pass

    class GenericProc:
        pass

    rag.modal_processors = {
        "image": ImageProc(),
        "generic": GenericProc(),
    }

    info = rag.get_processor_info()
    assert info["status"] == "Initialized"
    assert info["processors"]["image"]["class"] == "ImageProc"
    assert info["processors"]["image"]["enabled"] is True
    assert isinstance(info["processors"]["image"]["supports"], list)
    assert info["processors"]["generic"]["class"] == "GenericProc"
    assert info["models"]["vision_model"] == "External function"
