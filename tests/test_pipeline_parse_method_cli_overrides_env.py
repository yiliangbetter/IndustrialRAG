"""Pipeline --parse-method must override PARSE_METHOD env.

#185 locks the omitted-flag path (argparse default reads PARSE_METHOD).
Operators use ``--parse-method ocr`` to force OCR on a single overnight
run without rewriting the shared env. If CLI loses to env, scanned manuals
stay on ``txt`` and drop figures/stamps.

Distinct from #164 (injected parse_method on ``_ingest_folder``) and #187
(MinerU kwargs, not method).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_method_cli_override",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_parse_method_env():
    before = os.environ.get("PARSE_METHOD")
    yield
    if before is None:
        os.environ.pop("PARSE_METHOD", None)
    else:
        os.environ["PARSE_METHOD"] = before


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
    async def finalize_storages(self):
        return None


@pytest.mark.asyncio
async def test_parse_method_cli_overrides_parse_method_env(
    pipeline, monkeypatch, tmp_path
):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("PARSE_METHOD", "txt")
    captured = {}

    async def fake_build_rag(working_dir, parser_output_dir):
        return FakeRAG(), SimpleNamespace(parser="mineru"), FakeLogger()

    async def fake_ingest_folder(*args, **kwargs):
        captured.update(kwargs)
        return 1, 0

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
    monkeypatch.setattr(pipeline, "_ingest_folder", fake_ingest_folder)
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
            "--parse-method",
            "ocr",
            "--ingest-only",
        ],
    )

    await pipeline.async_main()

    assert captured["parse_method"] == "ocr"
