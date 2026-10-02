"""OCR re-ingest must keep Hugging Face embeddings on the local cache.

``scripts/reingest_uploaded_documents_ocr.py`` sets ``HF_HUB_OFFLINE`` only
when the embedding backend is Hugging Face and ``HF_HOME`` names a cache.
Any other backend must leave the hub reachable so MinerU weight downloads
are not forced offline. An explicit ``HF_HUB_OFFLINE`` value or
``HF_FORCE_ONLINE`` must not be overwritten.

``_ensure_hf_hub_timeouts`` fills download timeouts when they are absent
and must keep operator values. ``main`` applies both gates before the
missing-folder exit, which is before model download and LightRAG import.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"

_ENV_KEYS = (
    "EMBEDDING_BACKEND",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HF_FORCE_ONLINE",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "NO_PROXY",
    "no_proxy",
    "PATH",
)


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_hf_hub_offline_c05e", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _clear_hub_env(monkeypatch):
    for key in (
        "EMBEDDING_BACKEND",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_FORCE_ONLINE",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_non_hf_backend_leaves_hub_online(reingest, monkeypatch, tmp_path):
    _clear_hub_env(monkeypatch)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("HF_HOME", str(cache))

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert "HF_HUB_OFFLINE" not in os.environ


def test_default_backend_leaves_hub_online(reingest, monkeypatch, tmp_path):
    _clear_hub_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert "HF_HUB_OFFLINE" not in os.environ


def test_hf_backend_with_cache_sets_offline(reingest, monkeypatch, tmp_path):
    _clear_hub_env(monkeypatch)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setenv("EMBEDDING_BACKEND", "  HF  ")
    monkeypatch.setenv("HF_HOME", str(cache))

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize("hf_home", ["", "   ", None])
def test_hf_backend_without_cache_stays_online(reingest, monkeypatch, hf_home):
    _clear_hub_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    if hf_home is None:
        monkeypatch.delenv("HF_HOME", raising=False)
    else:
        monkeypatch.setenv("HF_HOME", hf_home)

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert "HF_HUB_OFFLINE" not in os.environ


def test_existing_offline_flag_is_preserved(reingest, monkeypatch, tmp_path):
    _clear_hub_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert os.environ["HF_HUB_OFFLINE"] == "0"


@pytest.mark.parametrize("force_online", ["1", "true", "YES"])
def test_force_online_keeps_hub_reachable(
    reingest, monkeypatch, tmp_path, force_online
):
    _clear_hub_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_FORCE_ONLINE", force_online)

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert "HF_HUB_OFFLINE" not in os.environ


@pytest.mark.parametrize("force_online", ["0", "false", "no"])
def test_negative_force_online_still_uses_local_cache(
    reingest, monkeypatch, tmp_path, force_online
):
    _clear_hub_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_FORCE_ONLINE", force_online)

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_timeouts_default_when_unset_and_keep_operator_values(reingest, monkeypatch):
    _clear_hub_env(monkeypatch)

    reingest._ensure_hf_hub_timeouts()

    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "120"

    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "30")

    reingest._ensure_hf_hub_timeouts()

    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "1800"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"


def test_main_applies_hub_gates_before_missing_folder_exit(
    reingest, monkeypatch, tmp_path
):
    _clear_hub_env(monkeypatch)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    missing = tmp_path / "missing-uploads"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(missing),
            "--skip-model-download",
        ],
    )

    def _download_must_not_run(*_args, **_kwargs):
        raise AssertionError("mineru model download must not run")

    monkeypatch.setattr(reingest.subprocess, "run", _download_must_not_run)

    with pytest.raises(SystemExit, match="Not a directory"):
        asyncio.run(reingest.main())

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "120"
