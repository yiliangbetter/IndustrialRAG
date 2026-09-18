"""OCR re-ingest copies of MinerU Darwin defaults and model-download source.

Pipeline's helpers are locked by #130 (kwargs) and #176 (download source).
``scripts/reingest_uploaded_documents_ocr.py`` duplicates both. A drift on
Mac would skip the Darwin CPU default and hang GPU OCR; a typo
``MINERU_MODEL_SOURCE`` would point ``mineru-models-download`` at an unknown
hub on the default (non-skip) re-ingest path. Distinct from #177/#186
(OCR_LANG / paddleocr empty kwargs) and #179 (default ``backend=pipeline``
on ``process_document_complete``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_mineru_kwargs_download",
        REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clear_mineru_env(monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)


def test_mineru_parse_kwargs_darwin_defaults_cpu_for_mineru_only(reingest, monkeypatch):
    _clear_mineru_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")

    kwargs = reingest._mineru_parse_kwargs("mineru")
    assert kwargs["backend"] == "pipeline"
    assert kwargs["device"] == "cpu"

    other = reingest._mineru_parse_kwargs("docling")
    assert "device" not in other
    assert "backend" not in other


def test_mineru_parse_kwargs_env_overrides_win_over_darwin_default(
    reingest, monkeypatch
):
    _clear_mineru_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("MINERU_BACKEND", "vlm-http-client")
    monkeypatch.setenv("MINERU_SOURCE", "modelscope")
    monkeypatch.setenv("MINERU_DEVICE", "mps")
    monkeypatch.setenv("MINERU_LANG", "ch")

    kwargs = reingest._mineru_parse_kwargs("mineru")
    assert kwargs == {
        "lang": "ch",
        "backend": "vlm-http-client",
        "source": "modelscope",
        "device": "mps",
    }


def test_download_models_unknown_source_remaps_to_huggingface(reingest, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs.get("cwd"), kwargs.get("check")))
        return None

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "not-a-hub")
    monkeypatch.setattr(reingest.subprocess, "run", fake_run)

    reingest._download_mineru_pipeline_models()

    assert calls == [
        (
            ["mineru-models-download", "-s", "huggingface", "-m", "pipeline"],
            str(REPO_ROOT),
            False,
        )
    ]


def test_download_models_modelscope_is_preserved_and_unset_defaults_huggingface(
    reingest, monkeypatch
):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return None

    monkeypatch.setattr(reingest.subprocess, "run", fake_run)

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "ModelScope")
    reingest._download_mineru_pipeline_models()
    assert calls[-1][2] == "modelscope"

    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    reingest._download_mineru_pipeline_models()
    assert calls[-1][2] == "huggingface"
