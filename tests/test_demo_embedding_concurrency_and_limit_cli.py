"""Demo Q&A argparse must forward embedding concurrency and --limit.

``scripts/run_demo_question_bank.py`` uses ``--embedding-max-async`` /
``--embedding-batch-num`` (default env ``EMBEDDING_FUNC_MAX_ASYNC`` /
``EMBEDDING_BATCH_NUM``, else 1) and ``--limit`` (default 0 = all).
Ignoring those flags either stampede the embedding host or answer the
entire plant-manual question bank during a dry run.

Distinct from #178 (``run_batch`` empty-row / limit consumption), #183
(``run_batch`` embed dims), #186 (query-mode / LLM_MODEL), #187 (delay),
and #188 (pipeline/graph LightRAG constructor env, not this CLI).
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
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_embed_limit_cli_bd82", SCRIPT_PATH
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

    monkeypatch.setattr(demo_script, "run_batch", fake_run_batch)
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
            "-w",
            str(tmp_path / "wd"),
            *(extra_argv or []),
        ],
    )
    demo_script.main()
    return captured


def test_omitted_embedding_concurrency_and_limit_use_defaults(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.delenv("EMBEDDING_FUNC_MAX_ASYNC", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_NUM", raising=False)

    captured = _run_main(demo_script, monkeypatch, tmp_path)

    assert captured["args"][6] == 1
    assert captured["args"][7] == 1
    assert captured["args"][8] == 0


def test_omitted_embedding_concurrency_reads_env(demo_script, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "3")
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "4")

    captured = _run_main(demo_script, monkeypatch, tmp_path)

    assert captured["args"][6] == 3
    assert captured["args"][7] == 4


def test_cli_embedding_concurrency_and_limit_override_env(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "8")
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "10")

    captured = _run_main(
        demo_script,
        monkeypatch,
        tmp_path,
        extra_argv=[
            "--embedding-max-async",
            "2",
            "--embedding-batch-num",
            "5",
            "--limit",
            "3",
        ],
    )

    assert captured["args"][6] == 2
    assert captured["args"][7] == 5
    assert captured["args"][8] == 3
