"""Pipeline model download must only target huggingface or modelscope.

``scripts/rag_pipeline_parse_graph_chat.py`` copies the re-ingest downloader.
A typo in ``MINERU_MODEL_SOURCE`` has to fall back to huggingface, while
``modelscope`` (any case) must be kept. ``check=False`` lets a failed
download continue into parse instead of aborting the run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_model_source_allowlist",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _capture_download(pipeline, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs.get("cwd"), kwargs.get("check")))
        return None

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    return calls


def test_unknown_and_blank_sources_download_from_huggingface(pipeline, monkeypatch):
    calls = _capture_download(pipeline, monkeypatch)

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "not-a-hub")
    pipeline._download_mineru_pipeline_models()

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "   ")
    pipeline._download_mineru_pipeline_models()

    expected = (
        ["mineru-models-download", "-s", "huggingface", "-m", "pipeline"],
        str(REPO_ROOT),
        False,
    )
    assert calls == [expected, expected]


def test_modelscope_case_is_kept_and_unset_defaults_to_huggingface(
    pipeline, monkeypatch
):
    calls = _capture_download(pipeline, monkeypatch)

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "  ModelScope  ")
    pipeline._download_mineru_pipeline_models()
    assert calls[-1][0][2] == "modelscope"
    assert calls[-1][2] is False

    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    pipeline._download_mineru_pipeline_models()
    assert calls[-1][0][2] == "huggingface"
