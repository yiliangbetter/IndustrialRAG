"""RAGAnything.update_config must mutate known fields and ignore unknown keys.

Callers use this to retune parse method and multimodal flags after construction.
Silently applying unknown names or rebuilding processors here would either drop
intended settings or thrash initialized modal processors.
"""

import pytest


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.debugs = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def error(self, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg) % args if args else str(msg))


def _make_rag(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **kwargs: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "workdir"),
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        parser_output_dir=str(tmp_path / "out"),
    )
    rag = rag_module.RAGAnything(config=config)
    rag.logger = FakeLogger()
    return rag


class TestUpdateConfig:
    def test_known_keys_are_applied(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)

        rag.update_config(
            parse_method="ocr",
            enable_image_processing=False,
            max_concurrent_files=4,
        )

        assert rag.config.parse_method == "ocr"
        assert rag.config.enable_image_processing is False
        assert rag.config.max_concurrent_files == 4
        # Untouched fields stay as constructed.
        assert rag.config.enable_table_processing is True

        info = rag.get_config_info()
        assert info["parsing"]["parse_method"] == "ocr"
        assert info["multimodal_processing"]["enable_image_processing"] is False
        assert info["batch_processing"]["max_concurrent_files"] == 4

    def test_unknown_keys_are_ignored(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)
        original_method = rag.config.parse_method

        rag.update_config(not_a_real_setting="danger", parse_method="txt")

        assert rag.config.parse_method == "txt"
        assert not hasattr(rag.config, "not_a_real_setting")
        assert any("Unknown config parameter" in msg for msg in rag.logger.warnings)
        assert original_method != "txt"

    def test_does_not_reinitialize_processors(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)
        calls = []

        def boom_init():
            calls.append("init")

        monkeypatch.setattr(rag, "_initialize_processors", boom_init)
        rag.modal_processors = {"image": object()}

        rag.update_config(enable_image_processing=False)

        assert calls == []
        assert "image" in rag.modal_processors
