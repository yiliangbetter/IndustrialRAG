"""Remaining RAGAnythingConfig env scalars that control routing and storage.

Bool/int multimodal flags are covered by an open coverage PR. These leftover
string/bool fields are still read at class-definition time, so tests reload
raganything.config after monkeypatch.setenv. Wrong PARSER/WORKING_DIR/OUTPUT_DIR
or ignored INCLUDE_*/folder recursion flags would mis-route every ingest.
"""

import importlib

import pytest

import raganything.config as config_mod

_SCALAR_ENV = (
    "PARSER",
    "PARSE_METHOD",
    "WORKING_DIR",
    "OUTPUT_DIR",
    "CONTENT_FORMAT",
    "CONTEXT_MODE",
    "DISPLAY_CONTENT_STATS",
    "RECURSIVE_FOLDER_PROCESSING",
    "INCLUDE_HEADERS",
    "INCLUDE_CAPTIONS",
    "ALLOW_EMBEDDING_ONLY_INGESTION",
    "ENABLE_IMAGE_PROCESSING",
    "ENABLE_TABLE_PROCESSING",
    "ENABLE_EQUATION_PROCESSING",
    "USE_FULL_PATH",
    "MAX_CONCURRENT_FILES",
    "CONTEXT_WINDOW",
    "MAX_CONTEXT_TOKENS",
)


def _reload_config(monkeypatch, **env_values):
    for key in _SCALAR_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_mod).RAGAnythingConfig


@pytest.fixture
def restore_config_defaults(monkeypatch):
    yield
    for key in _SCALAR_ENV:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(config_mod)


class TestConfigEnvStringRouting:
    def test_parser_working_dir_and_output_dir_from_env(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            PARSER="docling",
            WORKING_DIR="./plant_a_storage",
            OUTPUT_DIR="./plant_a_parsed",
        )
        config = Config()
        assert config.parser == "docling"
        assert config.working_dir == "./plant_a_storage"
        assert config.parser_output_dir == "./plant_a_parsed"

    def test_parse_method_content_format_and_context_mode_from_env(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            PARSE_METHOD="ocr",
            CONTENT_FORMAT="text_chunks",
            CONTEXT_MODE="chunk",
        )
        config = Config()
        assert config.parse_method == "ocr"
        assert config.content_format == "text_chunks"
        assert config.context_mode == "chunk"

    def test_constructor_kwargs_override_string_env(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(monkeypatch, PARSER="docling", PARSE_METHOD="ocr")
        config = Config(parser="mineru", parse_method="txt")
        assert config.parser == "mineru"
        assert config.parse_method == "txt"


class TestConfigEnvRemainingBools:
    def test_false_disables_stats_recursion_and_context_includes(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            DISPLAY_CONTENT_STATS="false",
            RECURSIVE_FOLDER_PROCESSING="0",
            INCLUDE_HEADERS="no",
            INCLUDE_CAPTIONS="false",
        )
        config = Config()
        assert config.display_content_stats is False
        assert config.recursive_folder_processing is False
        assert config.include_headers is False
        assert config.include_captions is False

    def test_true_enables_stats_and_recursion(
        self, monkeypatch, restore_config_defaults
    ):
        Config = _reload_config(
            monkeypatch,
            DISPLAY_CONTENT_STATS="true",
            RECURSIVE_FOLDER_PROCESSING="1",
        )
        config = Config()
        assert config.display_content_stats is True
        assert config.recursive_folder_processing is True
