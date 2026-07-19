"""Regression tests for ``scripts/run_demo_question_bank.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_demo_question_bank.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_questions():
    return _load_script_module()


@pytest.mark.asyncio
async def test_embedding_host_requires_its_own_api_key(
    demo_questions, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        demo_questions, "ensure_hf_home_from_repo_fallback", lambda _root: None
    )
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embeddings.example/v1")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    with pytest.raises(
        SystemExit, match="EMBEDDING_BINDING_HOST is set; set EMBEDDING_API_KEY"
    ):
        await demo_questions.run_batch(
            input_xlsx=tmp_path / "unused.xlsx",
            out_xlsx=tmp_path / "answers.xlsx",
            out_jsonl=tmp_path / "answers.jsonl",
            working_dir=tmp_path / "rag-storage",
            mode="mix",
            delay_seconds=0,
            embedding_func_max_async=1,
            embedding_batch_num=1,
            limit_questions=0,
            timing_log=None,
        )
