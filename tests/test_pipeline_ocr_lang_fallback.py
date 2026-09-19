"""Pipeline MinerU kwargs must honor OCR_LANG when MINERU_LANG is unset.

``scripts/rag_pipeline_parse_graph_chat.py`` copies ``_mineru_parse_kwargs``.
Overnight jobs often set ``OCR_LANG=ch`` without ``MINERU_LANG``. Dropping
that fallback silently OCRs Chinese manuals with MinerU's default language.
``MINERU_LANG`` must still win when both are set.

Distinct from #130 (pipeline helper ``MINERU_LANG`` / Darwin CPU), #186
(reingest copy of OCR_LANG), and #187 (async_main forwarding ``MINERU_LANG``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_KEYS = (
    "MINERU_LANG",
    "OCR_LANG",
    "MINERU_BACKEND",
    "MINERU_SOURCE",
    "MINERU_DEVICE",
    "PARSER",
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_ocr_lang_fallback_bd82",
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


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
    def __init__(self):
        self.parse_kwargs = []

    async def parse_document(
        self,
        file_path,
        output_dir=None,
        parse_method=None,
        display_stats=None,
        **kwargs,
    ):
        self.parse_kwargs.append(dict(kwargs))
        return [{"type": "text", "text": "ok"}], "doc-1"

    async def insert_content_list(self, content_list, file_path=None, **kwargs):
        return None

    async def finalize_storages(self):
        return None


def test_ocr_lang_fills_lang_when_mineru_lang_unset(pipeline, monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "ch"
    assert kwargs["backend"] == "pipeline"


def test_mineru_lang_wins_over_ocr_lang(pipeline, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", "en")
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "en"


def test_blank_ocr_lang_does_not_set_lang(pipeline, monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.setenv("OCR_LANG", "   ")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("mineru")

    assert "lang" not in kwargs


async def _run_ingest_only(pipeline, monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.pdf").write_bytes(b"%PDF")
    rag = FakeRAG()

    async def fake_build_rag(working_dir, parser_output_dir):
        config = SimpleNamespace(
            parser="mineru",
            supported_file_extensions=[".pdf"],
            display_content_stats=False,
        )
        return rag, config, FakeLogger()

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_pipeline_parse_graph_chat",
            "--input-folder",
            str(docs),
            "-w",
            str(tmp_path / "wd"),
            "--parser-output-dir",
            str(tmp_path / "out"),
            "--ingest-only",
        ],
    )
    await pipeline.async_main()
    return rag


@pytest.mark.asyncio
async def test_async_main_forwards_ocr_lang_when_mineru_lang_unset(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    rag = await _run_ingest_only(pipeline, monkeypatch, tmp_path)

    assert rag.parse_kwargs == [{"lang": "ch", "backend": "pipeline"}]
