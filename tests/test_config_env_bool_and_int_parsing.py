"""Env bool/int parsing for RAGAnythingConfig ingest and multimodal flags.

Scalar dataclass defaults call get_env_value at class-definition time, so
these tests reload raganything.config after setting env. Wrong parsing of
\"false\"/\"0\" would keep image/table/equation processing on, or enable
embedding-only ingest, with a large blast radius. Invalid ints must fall
back to coded defaults rather than crash at import.
"""

import importlib

import pytest

import raganything.config as config_mod

_BOOL_AND_INT_ENV = (
    "ALLOW_EMBEDDING_ONLY_INGESTION",
    "ENABLE_IMAGE_PROCESSING",
    "ENABLE_TABLE_PROCESSING",
    "ENABLE_EQUATION_PROCESSING",
    "DISPLAY_CONTENT_STATS",
    "RECURSIVE_FOLDER_PROCESSING",
    "INCLUDE_HEADERS",
    "INCLUDE_CAPTIONS",
    "USE_FULL_PATH",
    "MAX_CONCURRENT_FILES",
    "CONTEXT_WINDOW",
    "MAX_CONTEXT_TOKENS",
)


def _reload_config(monkeypatch, **env_values):
    for key in _BOOL_AND_INT_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_mod).RAGAnythingConfig


@pytest.fixture
def restore_config_defaults(monkeypatch):
    yield
    for key in _BOOL_AND_INT_ENV:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(config_mod)


class TestConfigEnvBoolParsing:
    def test_false_and_zero_disable_multimodal_flags(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            ENABLE_IMAGE_PROCESSING="false",
            ENABLE_TABLE_PROCESSING="0",
            ENABLE_EQUATION_PROCESSING="no",
        )
        config = Config()
        assert config.enable_image_processing is False
        assert config.enable_table_processing is False
        assert config.enable_equation_processing is False

    def test_true_and_one_enable_embedding_only_and_full_path(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            ALLOW_EMBEDDING_ONLY_INGESTION="true",
            USE_FULL_PATH="1",
        )
        config = Config()
        assert config.allow_embedding_only_ingestion is True
        assert config.use_full_path is True

    def test_constructor_kwargs_override_env_bools(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(monkeypatch, ENABLE_IMAGE_PROCESSING="false")
        config = Config(enable_image_processing=True)
        assert config.enable_image_processing is True


class TestConfigEnvIntParsing:
    def test_valid_int_env_is_applied(self, monkeypatch, restore_config_defaults):
        Config = _reload_config(
            monkeypatch,
            MAX_CONCURRENT_FILES="4",
            CONTEXT_WINDOW="3",
            MAX_CONTEXT_TOKENS="512",
        )
        config = Config()
        assert config.max_concurrent_files == 4
        assert config.context_window == 3
        assert config.max_context_tokens == 512

    def test_invalid_int_env_falls_back_to_defaults(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            MAX_CONCURRENT_FILES="nope",
            CONTEXT_WINDOW="x",
            MAX_CONTEXT_TOKENS="",
        )
        config = Config()
        assert config.max_concurrent_files == 1
        assert config.context_window == 1
        assert config.max_context_tokens == 2000
