"""Regression tests for ``scripts/reingest_uploaded_documents_ocr.py`` helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"


def _load_script_module():
    name = "reingest_uploaded_documents_ocr_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def reingest():
    return _load_script_module()


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    keys = (
        "EMBEDDING_BACKEND",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HF_FORCE_ONLINE",
        "NO_PROXY",
        "no_proxy",
        "MINERU_LANG",
        "OCR_LANG",
        "MINERU_BACKEND",
        "MINERU_SOURCE",
        "MINERU_DEVICE",
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BINDING_HOST",
        "PARSER",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def test_prefer_hf_hub_offline_sets_flag_for_local_hf(reingest, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_FORCE_ONLINE", raising=False)

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_prefer_hf_hub_offline_skips_when_force_online(reingest, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    monkeypatch.setenv("HF_FORCE_ONLINE", "true")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert "HF_HUB_OFFLINE" not in os.environ


def test_prefer_hf_hub_offline_noop_for_openai_backend(reingest, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    reingest._prefer_hf_hub_offline_for_embeddings()

    assert "HF_HUB_OFFLINE" not in os.environ


def test_ensure_loopback_no_proxy_merges_existing_entries(reingest, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com, localhost")
    monkeypatch.setenv("no_proxy", "internal.test,example.com")

    reingest._ensure_loopback_no_proxy()

    expected = "example.com,localhost,internal.test,127.0.0.1,::1"
    assert os.environ["NO_PROXY"] == expected
    assert os.environ["no_proxy"] == expected


def test_resolve_keys_falls_back_embedding_key_to_llm_key(reingest, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-secret")
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    llm_key, emb_key = reingest._resolve_keys()

    assert llm_key == "llm-secret"
    assert emb_key == "llm-secret"


def test_mineru_parse_kwargs_reads_env_and_defaults_backend(reingest, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", "en")
    monkeypatch.setenv("MINERU_SOURCE", "modelscope")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)

    with patch.object(sys, "platform", "linux"):
        kwargs = reingest._mineru_parse_kwargs("mineru")

    assert kwargs == {
        "lang": "en",
        "source": "modelscope",
        "backend": "pipeline",
    }


def test_mineru_parse_kwargs_defaults_device_cpu_on_darwin(reingest, monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)

    with patch.object(sys, "platform", "darwin"):
        kwargs = reingest._mineru_parse_kwargs("mineru")

    assert kwargs["backend"] == "pipeline"
    assert kwargs["device"] == "cpu"


@pytest.mark.asyncio
async def test_main_requires_embedding_api_key_when_host_set(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "docs"
    folder.mkdir()

    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embeddings.example/v1")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--skip-model-download",
        ],
    )

    with (
        patch(
            "raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback",
            lambda _root: None,
        ),
        pytest.raises(
            SystemExit,
            match="EMBEDDING_BINDING_HOST is set; set EMBEDDING_API_KEY",
        ),
    ):
        await reingest.main()
