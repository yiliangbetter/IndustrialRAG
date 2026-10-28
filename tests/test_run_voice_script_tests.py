"""Regression coverage for the legacy voice evaluation harness wiring."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import query_progress_hooks
import run_voice_script_tests as voice


def test_default_voice_cases_are_tracked_and_complete() -> None:
    cases = voice._load_cases_file(voice._DEFAULT_CASES)

    assert len(cases) == 29
    assert [case["id"] for case in cases] == list(range(1, 30))


def test_cli_path_defaults_prefer_production_environment(monkeypatch, tmp_path) -> None:
    web_wd = tmp_path / "web_storage"
    legacy_wd = tmp_path / "legacy_storage"
    web_pod = tmp_path / "web_parser"
    legacy_pod = tmp_path / "legacy_parser"
    monkeypatch.setenv("RAG_WEB_WORKING_DIR", str(web_wd))
    monkeypatch.setenv("RAG_WORKING_DIR", str(legacy_wd))
    monkeypatch.setenv("RAG_WEB_PARSER_OUTPUT_DIR", str(web_pod))
    monkeypatch.setenv("PARSER_OUTPUT_DIR", str(legacy_pod))

    args = voice._parse_args([])

    assert args.wd == web_wd
    assert args.pod == web_pod


def test_cli_path_defaults_fall_back_to_legacy_environment(
    monkeypatch, tmp_path
) -> None:
    legacy_wd = tmp_path / "legacy_storage"
    legacy_pod = tmp_path / "legacy_parser"
    monkeypatch.delenv("RAG_WEB_WORKING_DIR", raising=False)
    monkeypatch.delenv("RAG_WEB_PARSER_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("RAG_WORKING_DIR", str(legacy_wd))
    monkeypatch.setenv("PARSER_OUTPUT_DIR", str(legacy_pod))

    args = voice._parse_args([])

    assert args.wd == legacy_wd
    assert args.pod == legacy_pod


def test_cli_path_defaults_fall_back_to_local_directories(monkeypatch) -> None:
    for name in (
        "RAG_WEB_WORKING_DIR",
        "RAG_WORKING_DIR",
        "RAG_WEB_PARSER_OUTPUT_DIR",
        "PARSER_OUTPUT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    args = voice._parse_args([])

    assert args.wd == voice._ROOT / "data" / "rag_storage"
    assert args.pod == voice._ROOT / "data" / "pipeline_parse"


def test_explicit_cli_paths_override_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_WEB_WORKING_DIR", str(tmp_path / "web_storage"))
    monkeypatch.setenv("RAG_WEB_PARSER_OUTPUT_DIR", str(tmp_path / "web_parser"))
    explicit_wd = tmp_path / "explicit_storage"
    explicit_pod = tmp_path / "explicit_parser"

    args = voice._parse_args(["--wd", str(explicit_wd), "--pod", str(explicit_pod)])

    assert args.wd == explicit_wd
    assert args.pod == explicit_pod


@asynccontextmanager
async def _noop_hooks():
    yield asyncio.Queue()


@pytest.mark.asyncio
async def test_run_cases_passes_distinct_parser_root_to_runtime(
    monkeypatch, tmp_path
) -> None:
    working_dir = tmp_path / "rag_storage"
    parser_dir = tmp_path / "pipeline_parse"
    media_roots: list[list[Path]] = []
    finalized: list[bool] = []

    class FakeRag:
        async def aquery(self, query, **_kwargs):
            assert query == "测试问法"
            return "标准答案"

        async def finalize_storages(self):
            finalized.append(True)

    async def build_rag(wd, pod, *, enable_llm_cache=None):
        assert wd == working_dir
        assert pod == parser_dir
        assert enable_llm_cache is False
        return FakeRag(), SimpleNamespace(), SimpleNamespace()

    fake_rpc = SimpleNamespace(
        _build_rag=build_rag,
        _query_extras_from_env=lambda _query: {},
    )
    monkeypatch.setattr(voice, "_load_rpc", lambda: fake_rpc)
    monkeypatch.setattr(query_progress_hooks, "query_progress_hooks", _noop_hooks)
    monkeypatch.setattr(
        query_progress_hooks,
        "set_query_media_roots",
        lambda roots: media_roots.append(roots),
    )
    monkeypatch.setattr(
        query_progress_hooks, "set_query_text_for_images", lambda _query: None
    )

    rows = await voice.run_cases(
        [
            {
                "id": 1,
                "group": "技术",
                "standard_question": "标准问题",
                "utterances": ["测试问法"],
                "expect": "technical",
                "grade": {"text_any": ["标准答案"]},
            }
        ],
        mode="rag",
        wd=working_dir,
        pod=parser_dir,
        query_mode="mix",
        include_skip=False,
    )

    assert rows[0]["grade"]["ok"] is True
    assert media_roots == [[parser_dir.resolve()]]
    assert finalized == [True]
