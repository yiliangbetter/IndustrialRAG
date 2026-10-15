"""Whitespace MinerU env values must not override OCR language or backend defaults.

``scripts/rag_pipeline_parse_graph_chat.py`` strips ``MINERU_LANG`` before
falling back to ``OCR_LANG``. A blank ``MINERU_LANG=`` line is common in
``.env`` files. Treating that blank as set sends MinerU an empty language
and skips ``OCR_LANG=ch``. A blank ``MINERU_BACKEND`` must likewise leave
the MinerU pipeline default in place.

Distinct from the open coverage that locks an unset ``MINERU_LANG`` and a
blank ``OCR_LANG`` on this helper.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_KEYS = (
    "MINERU_LANG",
    "OCR_LANG",
    "MINERU_BACKEND",
    "MINERU_SOURCE",
    "MINERU_DEVICE",
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_blank_mineru_lang_1015",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def test_whitespace_mineru_lang_falls_through_to_stripped_ocr_lang(
    pipeline, monkeypatch
):
    monkeypatch.setenv("MINERU_LANG", " \t ")
    monkeypatch.setenv("OCR_LANG", "  ch  ")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "ch"
    assert kwargs["backend"] == "pipeline"


def test_padded_mineru_lang_is_stripped_and_still_wins(pipeline, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", "  en  ")
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "en"


def test_whitespace_mineru_flags_do_not_replace_pipeline_default(pipeline, monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.setenv("MINERU_BACKEND", "   ")
    monkeypatch.setenv("MINERU_SOURCE", "  ")
    monkeypatch.setenv("MINERU_DEVICE", " ")
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("mineru")

    assert kwargs == {"backend": "pipeline"}
