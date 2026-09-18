"""Pipeline async_main must forward MinerU env kwargs into parse_document.

``_mineru_parse_kwargs`` is unit-tested in #130/#177. ``_ingest_folder``
forwards an injected ``parse_extra`` in #164. This locks the overnight
glue: ``MINERU_LANG`` / ``MINERU_BACKEND`` / ``MINERU_SOURCE`` /
``MINERU_DEVICE`` must reach ``rag.parse_document``. A non-MinerU parser
must not inherit ``backend=pipeline``. Distinct from #186 (``--no-recursive``
/ ``RAG_QUERY_MODE``) and #185 (``PARSE_METHOD`` argparse default).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_KEYS = (
    "MINERU_LANG",
    "OCR_LANG",
    "MINERU_BACKEND",
    "MINERU_SOURCE",
    "MINERU_DEVICE",
    "PARSER",
    "PARSE_METHOD",
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_forwards_mineru_parse_extra",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
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


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeRAG:
    def __init__(self):
        self.parse_kwargs = []

    async def parse_document(
        self,
        file_path,
        output_dir=None,
        parse_method=None,
        display_stats=None,
        **kwargs,
    ):
        self.parse_kwargs.append(dict(kwargs))
        return [{"type": "text", "text": "ok"}], "doc-1"

    async def insert_content_list(self, content_list, file_path=None, **kwargs):
        return None

    async def finalize_storages(self):
        return None


async def _run_ingest_only(pipeline, monkeypatch, tmp_path, *, parser: str):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.pdf").write_bytes(b"%PDF")
    rag = FakeRAG()

    async def fake_build_rag(working_dir, parser_output_dir):
        config = SimpleNamespace(
            parser=parser,
            supported_file_extensions=[".pdf"],
            display_content_stats=False,
        )
        return rag, config, FakeLogger()

    monkeypatch.setattr(pipeline, "_build_rag", fake_build_rag)
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
        ],
    )
    await pipeline.async_main()
    return rag


@pytest.mark.asyncio
async def test_async_main_forwards_mineru_env_kwargs_to_parse_document(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.setenv("MINERU_LANG", "ch")
    monkeypatch.setenv("MINERU_BACKEND", "vlm-http-client")
    monkeypatch.setenv("MINERU_SOURCE", "modelscope")
    monkeypatch.setenv("MINERU_DEVICE", "cuda")
    monkeypatch.setattr(sys, "platform", "linux")

    rag = await _run_ingest_only(pipeline, monkeypatch, tmp_path, parser="mineru")

    assert rag.parse_kwargs == [
        {
            "lang": "ch",
            "backend": "vlm-http-client",
            "source": "modelscope",
            "device": "cuda",
        }
    ]


@pytest.mark.asyncio
async def test_async_main_does_not_inject_pipeline_backend_for_paddleocr(
    pipeline, monkeypatch, tmp_path
):
    monkeypatch.delenv("MINERU_LANG", raising=False)
    monkeypatch.delenv("OCR_LANG", raising=False)
    monkeypatch.delenv("MINERU_BACKEND", raising=False)
    monkeypatch.delenv("MINERU_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    rag = await _run_ingest_only(pipeline, monkeypatch, tmp_path, parser="paddleocr")

    assert rag.parse_kwargs == [{}]
