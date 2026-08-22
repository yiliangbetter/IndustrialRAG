"""Fail-closed gates for parse/graph pipeline and OCR re-ingest scripts.

These CLIs are the production ingest entrypoints. Missing folders, empty
inputs, missing API keys, HF-offline embedding defaults, and MinerU env
kwargs must stay deterministic so a bad run cannot start a hang or a
partial graph write.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
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


def test_normalize_ext_adds_dot(pipeline):
    assert pipeline._normalize_ext("PDF") == ".pdf"
    assert pipeline._normalize_ext(".md") == ".md"


def test_collect_files_recursive_and_non_recursive(pipeline, tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    recursive = pipeline._collect_files(tmp_path, [".pdf"], recursive=True)
    assert {p.name for p in recursive} == {"a.pdf", "b.pdf"}

    top_only = pipeline._collect_files(tmp_path, ["pdf"], recursive=False)
    assert {p.name for p in top_only} == {"a.pdf"}


def test_mineru_parse_kwargs_defaults_and_env(pipeline, monkeypatch):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    mineru = pipeline._mineru_parse_kwargs("mineru")
    assert mineru["backend"] == "pipeline"
    assert "device" not in mineru
    assert "lang" not in mineru

    other = pipeline._mineru_parse_kwargs("docling")
    assert other == {}

    monkeypatch.setenv("MINERU_LANG", "ch")
    monkeypatch.setenv("MINERU_BACKEND", "vlm-http-client")
    monkeypatch.setenv("MINERU_SOURCE", "modelscope")
    monkeypatch.setenv("MINERU_DEVICE", "cuda")
    override = pipeline._mineru_parse_kwargs("mineru")
    assert override == {
        "lang": "ch",
        "backend": "vlm-http-client",
        "source": "modelscope",
        "device": "cuda",
    }


def test_mineru_parse_kwargs_darwin_defaults_cpu(pipeline, monkeypatch):
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    kwargs = pipeline._mineru_parse_kwargs("mineru")
    assert kwargs["backend"] == "pipeline"
    assert kwargs["device"] == "cpu"


@pytest.mark.asyncio
async def test_pipeline_async_main_rejects_missing_input_dir(
    pipeline, monkeypatch, tmp_path
):
    missing = tmp_path / "nope"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_pipeline_parse_graph_chat",
            "--input-folder",
            str(missing),
        ],
    )

    with pytest.raises(SystemExit, match="Not a directory"):
        await pipeline.async_main()


@pytest.mark.asyncio
async def test_ingest_folder_exits_when_no_supported_files(pipeline, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    config = SimpleNamespace(
        supported_file_extensions=[".pdf"], display_content_stats=False
    )

    with pytest.raises(SystemExit, match="No supported files"):
        await pipeline._ingest_folder(
            rag=SimpleNamespace(),
            config=config,
            logger=SimpleNamespace(
                info=lambda *a, **k: None, error=lambda *a, **k: None
            ),
            input_folder=empty,
            parser_output_dir=tmp_path / "out",
            parse_method="auto",
            parse_extra={},
            recursive=True,
            limit=0,
            skip_multimodal=True,
        )


@pytest.mark.asyncio
async def test_build_rag_requires_llm_key(pipeline, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)

    with pytest.raises(SystemExit, match="OPENAI_API_KEY or LLM_BINDING_API_KEY"):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")


@pytest.mark.asyncio
async def test_build_rag_requires_embedding_key_when_host_set(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embed.example")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")

    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
        await pipeline._build_rag(tmp_path / "wd", tmp_path / "out")


def test_reingest_resolve_keys_prefers_binding_then_openai(reingest, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BINDING_API_KEY", "bind-key")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    llm_key, emb_key = reingest._resolve_keys()
    assert llm_key == "bind-key"
    assert emb_key == "bind-key"

    monkeypatch.setenv("EMBEDDING_API_KEY", "emb-only")
    llm_key, emb_key = reingest._resolve_keys()
    assert llm_key == "bind-key"
    assert emb_key == "emb-only"


def test_reingest_prefer_hf_hub_offline_gates(reingest, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_FORCE_ONLINE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    reingest._prefer_hf_hub_offline_for_embeddings()
    assert "HF_HUB_OFFLINE" not in os.environ

    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    reingest._prefer_hf_hub_offline_for_embeddings()
    assert "HF_HUB_OFFLINE" not in os.environ

    monkeypatch.setenv("HF_HOME", "/tmp/hf")
    reingest._prefer_hf_hub_offline_for_embeddings()
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    reingest._prefer_hf_hub_offline_for_embeddings()
    assert os.environ["HF_HUB_OFFLINE"] == "0"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_FORCE_ONLINE", "true")
    reingest._prefer_hf_hub_offline_for_embeddings()
    assert "HF_HUB_OFFLINE" not in os.environ


def test_reingest_loopback_no_proxy_merges(reingest, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "corp.local")
    monkeypatch.delenv("no_proxy", raising=False)
    reingest._ensure_loopback_no_proxy()
    parts = [p.strip() for p in os.environ["NO_PROXY"].split(",") if p.strip()]
    assert parts[:1] == ["corp.local"]
    assert "127.0.0.1" in parts
    assert "localhost" in parts
    assert "::1" in parts
    assert os.environ["NO_PROXY"] == os.environ["no_proxy"]


@pytest.mark.asyncio
async def test_reingest_main_rejects_missing_folder(reingest, monkeypatch, tmp_path):
    missing = tmp_path / "missing"
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
        await reingest.main()
