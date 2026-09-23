"""Demo Q&A omitted ``-w`` must keep the ``rag_storage`` store.

``scripts/run_demo_question_bank.py`` defaults LightRAG persistence to
``<repo>/rag_storage``. Pointing that default at ``rag_storage_pipeline`` or
``rag_storage_wt1536`` would answer the question bank against the wrong index.
An explicit ``-w`` must still override the default. Distinct from #189 (embed
CLI / ``--limit``), #190 (``--timing-log``), and #192 (``PARSER`` env).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_demo_question_bank.py"

_ENV_KEYS = (
    "EMBEDDING_FUNC_MAX_ASYNC",
    "EMBEDDING_BATCH_NUM",
    "RAG_QUERY_MODE",
    "DEMO_QA_QUERY_DELAY_SECONDS",
    "WORKING_DIR",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_omitted_working_dir_37e0", SCRIPT_PATH
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


def _run_main(demo_script, monkeypatch, tmp_path, extra_argv=None):
    captured = {}

    async def fake_run_batch(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(demo_script, "_ROOT", checkout)
    monkeypatch.setattr(demo_script, "run_batch", fake_run_batch)
    monkeypatch.delenv("WORKING_DIR", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_demo_question_bank",
            "--input",
            str(tmp_path / "in.xlsx"),
            "--out-xlsx",
            str(tmp_path / "out.xlsx"),
            "--out-jsonl",
            str(tmp_path / "out.jsonl"),
            *(extra_argv or []),
        ],
    )
    demo_script.main()
    captured["checkout"] = checkout
    return captured


def test_omitted_working_dir_uses_repo_rag_storage(demo_script, monkeypatch, tmp_path):
    captured = _run_main(demo_script, monkeypatch, tmp_path)
    assert captured["args"][3] == captured["checkout"] / "rag_storage"


def test_explicit_working_dir_overrides_rag_storage_default(
    demo_script, monkeypatch, tmp_path
):
    custom = tmp_path / "custom_store"
    captured = _run_main(
        demo_script,
        monkeypatch,
        tmp_path,
        extra_argv=["-w", str(custom)],
    )
    assert captured["args"][3] == custom
