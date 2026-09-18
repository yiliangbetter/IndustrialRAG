"""Demo Q&A argparse delay defaults must honor DEMO_QA_QUERY_DELAY_SECONDS.

``scripts/run_demo_question_bank.py`` sleeps between answers to avoid
embedding rate limits. Omitting ``--delay`` must keep the env value
(default 4 seconds); ignoring it either burns Ark/OpenAI quotas or
stalls a local batch. Distinct from #186 (``RAG_QUERY_MODE`` / ``LLM_MODEL``)
and #178 (empty-row / timing log).
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
    "RAG_QUERY_MODE",
    "DEMO_QA_QUERY_DELAY_SECONDS",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_delay_env", SCRIPT_PATH
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


def test_omitted_delay_defaults_to_four_seconds(demo_script, monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_QA_QUERY_DELAY_SECONDS", raising=False)

    captured = _run_main(demo_script, monkeypatch, tmp_path)

    assert captured["args"][5] == 4.0


def test_omitted_delay_reads_env(demo_script, monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_QA_QUERY_DELAY_SECONDS", "0.25")

    captured = _run_main(demo_script, monkeypatch, tmp_path)

    assert captured["args"][5] == 0.25


def test_cli_delay_overrides_env(demo_script, monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_QA_QUERY_DELAY_SECONDS", "9")

    captured = _run_main(
        demo_script,
        monkeypatch,
        tmp_path,
        extra_argv=["--delay", "0"],
    )

    assert captured["args"][5] == 0.0
