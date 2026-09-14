"""Parser CLIs must prepend the project venv bin when it exists.

Pipeline, OCR re-ingest, and MinerU JSON export locate ``mineru`` /
``mineru-models-download`` via PATH. If ``<repo>/.venv/bin`` exists it must
be prepended; a missing venv must leave PATH unchanged. Export also falls
back to ``<repo>/../../.venv/bin`` when the local venv is absent.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_path"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_script("rag_pipeline_parse_graph_chat.py")


@pytest.fixture(scope="module")
def reingest():
    return _load_script("reingest_uploaded_documents_ocr.py")


@pytest.fixture(scope="module")
def export_mod():
    return _load_script("export_parse_json_no_llm.py")


@pytest.fixture(autouse=True)
def _restore_path():
    before = os.environ.get("PATH")
    yield
    if before is None:
        os.environ.pop("PATH", None)
    else:
        os.environ["PATH"] = before


def test_pipeline_prepends_venv_bin_when_present(pipeline, monkeypatch, tmp_path):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "_ROOT", tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")

    pipeline._ensure_venv_bin_on_path()

    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[0] == str(venv_bin)
    assert "/usr/bin" in parts


def test_pipeline_leaves_path_when_venv_missing(pipeline, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_ROOT", tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")

    pipeline._ensure_venv_bin_on_path()

    assert os.environ["PATH"] == "/usr/bin"


def test_reingest_prepends_venv_bin_when_present(reingest, monkeypatch, tmp_path):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setattr(reingest, "_ROOT", tmp_path)
    monkeypatch.setenv("PATH", "/opt/bin")

    reingest._ensure_venv_path()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(venv_bin)


def test_export_uses_local_venv_before_parent_fallback(
    export_mod, monkeypatch, tmp_path
):
    local_bin = tmp_path / ".venv" / "bin"
    parent_bin = tmp_path.parent.parent / ".venv" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setattr(export_mod, "_ROOT", tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")

    export_mod._venv_path()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(local_bin)
    assert str(parent_bin) not in os.environ["PATH"].split(os.pathsep)


def test_export_falls_back_to_grandparent_venv(export_mod, monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    repo = workspace / "a" / "proj"
    repo.mkdir(parents=True)
    parent_bin = workspace / ".venv" / "bin"
    parent_bin.mkdir(parents=True)
    monkeypatch.setattr(export_mod, "_ROOT", repo)
    monkeypatch.setenv("PATH", "/usr/bin")

    export_mod._venv_path()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(parent_bin)
