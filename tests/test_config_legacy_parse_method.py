"""Regression tests for legacy MINERU_PARSE_METHOD config fallback.

Deployments still set MINERU_PARSE_METHOD from older docs. Losing that mapping
silently switches Office/PDF parsing away from the intended OCR/txt method.
"""

from __future__ import annotations

import pytest

from raganything.config import RAGAnythingConfig


def test_legacy_mineru_parse_method_applies_when_parse_method_unset(monkeypatch):
    monkeypatch.delenv("PARSE_METHOD", raising=False)
    monkeypatch.setenv("MINERU_PARSE_METHOD", "ocr")

    with pytest.warns(DeprecationWarning, match="MINERU_PARSE_METHOD"):
        config = RAGAnythingConfig()

    assert config.parse_method == "ocr"


def test_parse_method_env_wins_over_legacy_mineru_parse_method(monkeypatch):
    monkeypatch.setenv("PARSE_METHOD", "txt")
    monkeypatch.setenv("MINERU_PARSE_METHOD", "ocr")

    config = RAGAnythingConfig(parse_method="txt")

    assert config.parse_method == "txt"
