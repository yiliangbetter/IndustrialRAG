"""OCR_LANG must feed MinerU when MINERU_LANG is unset or whitespace.

Pipeline and reingest copy the same helper. Operators often keep OCR_LANG
from PaddleOCR jobs; dropping that fallback silently OCR's Chinese manuals
with MinerU's default language.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_script(
        "rag_pipeline_ocr_lang_fallback",
        "scripts/rag_pipeline_parse_graph_chat.py",
    )


@pytest.fixture(scope="module")
def reingest():
    return _load_script(
        "reingest_ocr_lang_fallback",
        "scripts/reingest_uploaded_documents_ocr.py",
    )


def _clear_lang_env(monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.mark.parametrize("script_fixture", ["pipeline", "reingest"])
def test_ocr_lang_used_when_mineru_lang_missing(script_fixture, request, monkeypatch):
    script = request.getfixturevalue(script_fixture)
    _clear_lang_env(monkeypatch)
    monkeypatch.setenv("OCR_LANG", "en")

    kwargs = script._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "en"
    assert kwargs["backend"] == "pipeline"


@pytest.mark.parametrize("script_fixture", ["pipeline", "reingest"])
def test_whitespace_mineru_lang_falls_through_to_ocr_lang(
    script_fixture, request, monkeypatch
):
    script = request.getfixturevalue(script_fixture)
    _clear_lang_env(monkeypatch)
    monkeypatch.setenv("MINERU_LANG", "   ")
    monkeypatch.setenv("OCR_LANG", "ch")

    kwargs = script._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "ch"


@pytest.mark.parametrize("script_fixture", ["pipeline", "reingest"])
def test_mineru_lang_wins_over_ocr_lang(script_fixture, request, monkeypatch):
    script = request.getfixturevalue(script_fixture)
    _clear_lang_env(monkeypatch)
    monkeypatch.setenv("MINERU_LANG", "ch")
    monkeypatch.setenv("OCR_LANG", "en")

    kwargs = script._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "ch"
