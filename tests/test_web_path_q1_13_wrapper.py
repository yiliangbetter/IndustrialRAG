from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_web_path_q1_13.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("run_web_path_q1_13_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_exposes_only_q1_through_q13() -> None:
    wrapper = _load_wrapper()

    assert list(wrapper.REF) == list(range(1, 14))
    assert wrapper.grade_text is wrapper._suite.grade_text
    assert wrapper.grade_images is wrapper._suite.grade_images


def test_run_cases_preserves_legacy_execution_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    wrapper = _load_wrapper()
    captured: dict = {}
    sentinel = [{"id": 1}]

    async def fake_run_cases(ids, **kwargs):
        captured["ids"] = ids
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(wrapper._suite, "run_cases", fake_run_cases)
    result = asyncio.run(
        wrapper.run_cases(
            [1, 13],
            mode="mix",
            wd=tmp_path / "storage",
            pod=tmp_path / "parser",
        )
    )

    assert result is sentinel
    assert captured == {
        "ids": [1, 13],
        "mode": "mix",
        "wd": tmp_path / "storage",
        "pod": tmp_path / "parser",
        "write_dumps": False,
        "skip_gate": True,
        "web_sim": "none",
    }


def test_main_keeps_legacy_case_set_and_report_location(
    monkeypatch, tmp_path: Path
) -> None:
    wrapper = _load_wrapper()
    storage = tmp_path / "storage"
    parser = tmp_path / "parser"
    captured: dict = {}

    async def fake_run_cases(ids, **kwargs):
        captured["ids"] = ids
        captured.update(kwargs)
        return [{"id": case_id, "grade": {"ok": True}} for case_id in ids]

    def fake_write_report(path, rows):
        captured["report_path"] = path
        captured["report_rows"] = rows
        path.write_text("report", encoding="utf-8")

    monkeypatch.setattr(wrapper, "_ROOT", tmp_path)
    monkeypatch.setattr(wrapper, "run_cases", fake_run_cases)
    monkeypatch.setattr(wrapper, "write_report", fake_write_report)
    monkeypatch.setenv("RAG_WEB_WORKING_DIR", str(storage))
    monkeypatch.setenv("RAG_WEB_PARSER_OUTPUT_DIR", str(parser))
    monkeypatch.setenv("RAG_QUERY_MODE", "local")

    wrapper.main()

    assert captured["ids"] == list(range(1, 14))
    assert captured["mode"] == "local"
    assert captured["wd"] == storage
    assert captured["pod"] == parser
    assert captured["report_path"].parent == tmp_path / "logs" / "web_path_q1_13"
    assert captured["report_path"].suffix == ".md"
    json_reports = list(captured["report_path"].parent.glob("*.json"))
    assert len(json_reports) == 1
