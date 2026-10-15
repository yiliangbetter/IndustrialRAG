"""OCR reingest must keep hub timeouts and ignore blank MinerU env values.

``scripts/reingest_uploaded_documents_ocr.py`` sets Hugging Face hub
timeouts with ``setdefault`` before any download, and it copies the same
``_mineru_parse_kwargs`` / model-source rules as the pipeline script.
A later assignment would clobber a longer operator timeout. A blank
``MINERU_LANG`` or ``MINERU_MODEL_SOURCE`` would otherwise OCR with the
wrong language or pass an empty ``-s`` to ``mineru-models-download``.
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
    "MINERU_MODEL_SOURCE",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HUB_OFFLINE",
    "HF_HOME",
    "EMBEDDING_BACKEND",
    "HF_FORCE_ONLINE",
    "NO_PROXY",
    "no_proxy",
    "PATH",
)


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_hub_timeouts_blank_env_1015",
        REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py",
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


def test_unset_hub_timeouts_default_to_long_downloads(reingest, monkeypatch):
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)

    reingest._ensure_hf_hub_timeouts()

    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "120"


def test_preset_hub_timeouts_including_empty_are_kept(reingest, monkeypatch):
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "")

    reingest._ensure_hf_hub_timeouts()

    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "1800"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == ""


@pytest.mark.asyncio
async def test_main_applies_hub_timeouts_before_folder_check(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(tmp_path / "missing-uploads"),
            "--skip-model-download",
        ],
    )

    with pytest.raises(SystemExit, match="Not a directory"):
        await reingest.main()

    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "120"


def test_whitespace_mineru_lang_falls_through_to_ocr_lang(reingest, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", " \t ")
    monkeypatch.setenv("OCR_LANG", "  ch  ")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = reingest._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "ch"
    assert kwargs["backend"] == "pipeline"


def test_padded_mineru_lang_is_stripped_and_still_wins(reingest, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", "  en  ")
    monkeypatch.setenv("OCR_LANG", "ch")
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = reingest._mineru_parse_kwargs("mineru")

    assert kwargs["lang"] == "en"


def test_whitespace_flags_are_omitted_for_non_mineru_parser(reingest, monkeypatch):
    monkeypatch.setenv("MINERU_LANG", "   ")
    monkeypatch.setenv("OCR_LANG", "   ")
    monkeypatch.setenv("MINERU_BACKEND", "   ")
    monkeypatch.setenv("MINERU_SOURCE", "  ")
    monkeypatch.setenv("MINERU_DEVICE", " ")

    kwargs = reingest._mineru_parse_kwargs("paddleocr")

    assert kwargs == {}


def test_blank_model_source_downloads_from_huggingface(reingest, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return None

    monkeypatch.setattr(reingest.subprocess, "run", fake_run)

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "   ")
    reingest._download_mineru_pipeline_models()
    assert calls[-1] == [
        "mineru-models-download",
        "-s",
        "huggingface",
        "-m",
        "pipeline",
    ]

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "  ModelScope  ")
    reingest._download_mineru_pipeline_models()
    assert calls[-1][2] == "modelscope"
