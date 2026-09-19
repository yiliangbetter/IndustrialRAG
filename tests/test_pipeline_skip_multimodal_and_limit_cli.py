"""Pipeline async_main must forward --skip-multimodal and --limit.

``_ingest_folder`` already honors injected skip/limit flags (#164). Overnight
jobs go through ``async_main``. Dropping ``skip_multimodal=args.skip_multimodal``
would enable vision/table processors on the text-first graph path. Ignoring
``--limit`` would parse an entire plant-manual tree during a dry run.

Distinct from #164 (direct helper), #186 (``--no-recursive`` / query mode),
and #187 (MinerU ``parse_extra`` glue).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_skip_multimodal_limit_cli",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
    async def finalize_storages(self):
        return None


def _config():
    return SimpleNamespace(
        parser="mineru",
        supported_file_extensions=[".pdf"],
        display_content_stats=False,
    )


async def _run_ingest_only(pipeline, monkeypatch, tmp_path, extra_argv):
    docs = tmp_path / "docs"
    docs.mkdir()
    captured = {}

    async def fake_build_rag(working_dir, parser_output_dir):
        return FakeRAG(), _config(), FakeLogger()

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
            "--ingest-only",
            *extra_argv,
        ],
    )
    await pipeline.async_main()
    return captured


@pytest.mark.asyncio
async def test_async_main_defaults_to_skip_multimodal_and_unlimited(
    pipeline, monkeypatch, tmp_path
):
    captured = await _run_ingest_only(pipeline, monkeypatch, tmp_path, extra_argv=[])

    assert captured["skip_multimodal"] is True
    assert captured["limit"] == 0


@pytest.mark.asyncio
async def test_async_main_forwards_no_skip_multimodal(pipeline, monkeypatch, tmp_path):
    captured = await _run_ingest_only(
        pipeline, monkeypatch, tmp_path, extra_argv=["--no-skip-multimodal"]
    )

    assert captured["skip_multimodal"] is False


@pytest.mark.asyncio
async def test_async_main_forwards_limit(pipeline, monkeypatch, tmp_path):
    captured = await _run_ingest_only(
        pipeline, monkeypatch, tmp_path, extra_argv=["--limit", "2"]
    )

    assert captured["limit"] == 2
    assert captured["skip_multimodal"] is True
