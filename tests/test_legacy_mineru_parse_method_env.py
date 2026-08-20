"""Regression tests for legacy MINERU_PARSE_METHOD env fallback.

PARSE_METHOD is the current env name. MINERU_PARSE_METHOD must still win when
PARSE_METHOD is unset, or existing deployments silently switch to "auto".
"""

import pytest

from raganything.config import RAGAnythingConfig


class TestLegacyMineruParseMethodEnv:
    def test_legacy_env_applied_when_parse_method_unset(self, monkeypatch):
        monkeypatch.delenv("PARSE_METHOD", raising=False)
        monkeypatch.setenv("MINERU_PARSE_METHOD", "ocr")

        with pytest.warns(DeprecationWarning, match="MINERU_PARSE_METHOD"):
            config = RAGAnythingConfig()

        assert config.parse_method == "ocr"

    def test_parse_method_env_blocks_legacy_override(self, monkeypatch):
        monkeypatch.setenv("PARSE_METHOD", "txt")
        monkeypatch.setenv("MINERU_PARSE_METHOD", "ocr")

        config = RAGAnythingConfig()

        assert config.parse_method != "ocr"

    def test_neither_env_keeps_constructor_parse_method(self, monkeypatch):
        monkeypatch.delenv("PARSE_METHOD", raising=False)
        monkeypatch.delenv("MINERU_PARSE_METHOD", raising=False)

        config = RAGAnythingConfig(parse_method="txt")
        assert config.parse_method == "txt"
