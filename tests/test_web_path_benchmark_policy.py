from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_web_path_q1_17.py"


def _load_runner():
    name = "run_web_path_q1_17_benchmark_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("allow_llm_cache", [False, True])
def test_round_runner_uses_explicit_cache_policy(
    monkeypatch,
    tmp_path: Path,
    allow_llm_cache: bool,
) -> None:
    runner = _load_runner()
    captured: dict[str, object] = {}

    class FakeRag:
        async def finalize_storages(self) -> None:
            captured["finalized"] = True

    async def build_rag(_wd, _pod, *, enable_llm_cache=None):
        captured["enable_llm_cache"] = enable_llm_cache
        return FakeRag(), None, None

    async def run_cases(*_args, **_kwargs):
        return []

    def write_report(*_args, **kwargs):
        captured["report_cache"] = kwargs["llm_cache_enabled"]

    monkeypatch.setattr(
        runner,
        "_load_rpc",
        lambda: SimpleNamespace(_build_rag=build_rag),
    )
    monkeypatch.setattr(runner, "run_cases", run_cases)
    monkeypatch.setattr(runner, "write_report", write_report)

    summaries, failed = asyncio.run(
        runner.run_all_rounds(
            [],
            mode="mix",
            wd=tmp_path / "storage",
            pod=tmp_path / "parser",
            write_dumps=False,
            skip_gate=True,
            web_sim="none",
            rounds=1,
            report_dir=tmp_path,
            session_stamp="test",
            suffix="empty",
            allow_llm_cache=allow_llm_cache,
        )
    )

    assert captured == {
        "enable_llm_cache": allow_llm_cache,
        "report_cache": allow_llm_cache,
        "finalized": True,
    }
    assert summaries[0]["llm_cache_enabled"] is allow_llm_cache
    assert summaries[0]["ok"] is True
    assert failed is False
