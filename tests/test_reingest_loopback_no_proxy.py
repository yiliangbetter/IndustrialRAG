"""OCR re-ingest must keep loopback MinerU calls off the corporate proxy.

``scripts/reingest_uploaded_documents_ocr.py`` merges ``127.0.0.1``,
``localhost``, and ``::1`` into both ``NO_PROXY`` and ``no_proxy`` before it
checks the upload folder or downloads models. Existing bypass entries must
survive, blank tokens must be dropped, and hosts already listed must not be
duplicated. A missing call here sends mineru-api ``/health`` through
``HTTP(S)_PROXY`` and the re-ingest fails with a loopback 502.
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
    "NO_PROXY",
    "no_proxy",
    "PATH",
    "HF_HUB_OFFLINE",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
)
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "reingest_loopback_no_proxy_5486", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reingest():
    return _load_script_module()


@pytest.fixture(autouse=True)
def _restore_proxy_env():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _proxy_parts() -> list[str]:
    raw = os.environ["NO_PROXY"]
    assert os.environ["no_proxy"] == raw
    return [part.strip() for part in raw.split(",") if part.strip()]


def test_merges_existing_entries_and_adds_missing_loopback(reingest, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "corp.example,  localhost ,,")
    monkeypatch.setenv("no_proxy", "api.internal,127.0.0.1")

    reingest._ensure_loopback_no_proxy()

    assert _proxy_parts() == [
        "corp.example",
        "localhost",
        "api.internal",
        "127.0.0.1",
        "::1",
    ]


def test_empty_proxy_env_sets_only_loopback_hosts(reingest, monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    reingest._ensure_loopback_no_proxy()

    assert _proxy_parts() == list(_LOOPBACK)


def test_second_call_does_not_duplicate_hosts(reingest, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1,corp.example")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1,corp.example")

    reingest._ensure_loopback_no_proxy()
    once = _proxy_parts()
    reingest._ensure_loopback_no_proxy()

    assert once == ["127.0.0.1", "localhost", "::1", "corp.example"]
    assert _proxy_parts() == once


def test_main_applies_loopback_bypass_before_missing_folder_exit(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.setenv("NO_PROXY", "corp.example")
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

    with pytest.raises(SystemExit, match="Not a directory"):
        asyncio.run(reingest.main())

    parts = _proxy_parts()
    assert parts[0] == "corp.example"
    assert parts[1:] == list(_LOOPBACK)
