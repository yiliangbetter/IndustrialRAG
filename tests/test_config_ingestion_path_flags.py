"""Env/config flags that gate ingestion path selection and citations.

``RAGAnythingConfig`` evaluates many ``get_env_value(...)`` defaults at import
time, so these tests reload the config module after mutating the environment.
Distinct from open #88 (list ``default_factory`` parsing) and #70 (legacy
``MINERU_PARSE_METHOD``).
"""

from __future__ import annotations

import importlib
import sys

import pytest


INGEST_ENV_KEYS = (
    "ALLOW_EMBEDDING_ONLY_INGESTION",
    "USE_FULL_PATH",
    "CONTENT_FORMAT",
    "RECURSIVE_FOLDER_PROCESSING",
    "DISPLAY_CONTENT_STATS",
    "PARSER",
    "PARSE_METHOD",
    "MINERU_PARSE_METHOD",
    "WORKING_DIR",
    "OUTPUT_DIR",
)


def _reload_config_class(monkeypatch, **env_values):
    """Clear/set env keys, then reload ``raganything.config`` for fresh defaults."""
    for key in INGEST_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    # Drop cached module so field defaults re-read the environment.
    sys.modules.pop("raganything.config", None)
    config_module = importlib.import_module("raganything.config")
    return config_module.RAGAnythingConfig


def test_allow_embedding_only_ingestion_defaults_false(monkeypatch):
    RAGAnythingConfig = _reload_config_class(monkeypatch)
    assert RAGAnythingConfig().allow_embedding_only_ingestion is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_allow_embedding_only_ingestion_reads_env(monkeypatch, raw, expected):
    RAGAnythingConfig = _reload_config_class(
        monkeypatch, ALLOW_EMBEDDING_ONLY_INGESTION=raw
    )
    assert RAGAnythingConfig().allow_embedding_only_ingestion is expected


def test_allow_embedding_only_ingestion_constructor_override(monkeypatch):
    RAGAnythingConfig = _reload_config_class(
        monkeypatch, ALLOW_EMBEDDING_ONLY_INGESTION="false"
    )
    config = RAGAnythingConfig(allow_embedding_only_ingestion=True)
    assert config.allow_embedding_only_ingestion is True


def test_use_full_path_defaults_false_and_reads_env(monkeypatch):
    RAGAnythingConfig = _reload_config_class(monkeypatch)
    assert RAGAnythingConfig().use_full_path is False

    RAGAnythingConfig = _reload_config_class(monkeypatch, USE_FULL_PATH="true")
    assert RAGAnythingConfig().use_full_path is True


def test_content_format_default_and_env_override(monkeypatch):
    RAGAnythingConfig = _reload_config_class(monkeypatch)
    assert RAGAnythingConfig().content_format == "minerU"

    RAGAnythingConfig = _reload_config_class(monkeypatch, CONTENT_FORMAT="text_chunks")
    assert RAGAnythingConfig().content_format == "text_chunks"


def test_recursive_folder_processing_default_and_env_override(monkeypatch):
    RAGAnythingConfig = _reload_config_class(monkeypatch)
    assert RAGAnythingConfig().recursive_folder_processing is True

    RAGAnythingConfig = _reload_config_class(
        monkeypatch, RECURSIVE_FOLDER_PROCESSING="false"
    )
    assert RAGAnythingConfig().recursive_folder_processing is False


def test_display_content_stats_default_and_env_override(monkeypatch):
    RAGAnythingConfig = _reload_config_class(monkeypatch)
    assert RAGAnythingConfig().display_content_stats is True

    RAGAnythingConfig = _reload_config_class(monkeypatch, DISPLAY_CONTENT_STATS="0")
    assert RAGAnythingConfig().display_content_stats is False
