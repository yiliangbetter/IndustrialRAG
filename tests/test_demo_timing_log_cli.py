"""Demo Q&A argparse must forward --timing-log into run_batch.

``scripts/run_demo_question_bank.py`` appends one JSON line per answered
question when ``--timing-log`` is set. Dropping that positional (or always
passing None) hides per-query wall time during plant-manual dry runs.
Omitted flag must stay None; an explicit path must reach ``run_batch``.

Distinct from #178 (``run_batch`` writes the log when the param is set)
and #189 (embed concurrency / ``--limit`` CLI, not this path).
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
        "run_demo_question_bank_timing_log_cli_7d2e", SCRIPT_PATH
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


def test_omitted_timing_log_is_none(demo_script, monkeypatch, tmp_path):
    captured = _run_main(demo_script, monkeypatch, tmp_path)
    assert captured["args"][9] is None


def test_timing_log_cli_forwards_path(demo_script, monkeypatch, tmp_path):
    log_path = tmp_path / "logs" / "query_timing.jsonl"
    captured = _run_main(
        demo_script,
        monkeypatch,
        tmp_path,
        extra_argv=["--timing-log", str(log_path)],
    )
    assert captured["args"][9] == log_path
