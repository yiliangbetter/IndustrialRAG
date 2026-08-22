"""Regression tests for scripts/export_parse_json_no_llm.py helpers and CLI.

MinerU nested API calls hang when loopback is proxied. The export script must
merge 127.0.0.1/localhost/::1 into NO_PROXY/no_proxy without dropping existing
bypass entries. CLI fail-closed: any BatchParser failure returns exit 1, and
timeout/worker values are clamped before the pool is created.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "export_parse_json_no_llm.py"
)


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_parse_json_no_llm", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_mod():
    return _load_export_module()


def test_loopback_no_proxy_merges_existing_entries(export_mod, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com, .internal")
    monkeypatch.setenv("no_proxy", "api.local")

    export_mod._loopback_no_proxy()

    for key in ("NO_PROXY", "no_proxy"):
        parts = [p.strip() for p in os.environ[key].split(",") if p.strip()]
        assert parts == [
            "example.com",
            ".internal",
            "api.local",
            "127.0.0.1",
            "localhost",
            "::1",
        ]


def test_loopback_no_proxy_dedupes_existing_loopback(export_mod, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.delenv("no_proxy", raising=False)

    export_mod._loopback_no_proxy()

    parts = [p.strip() for p in os.environ["NO_PROXY"].split(",") if p.strip()]
    assert parts.count("127.0.0.1") == 1
    assert parts.count("localhost") == 1
    assert "::1" in parts
    assert os.environ["NO_PROXY"] == os.environ["no_proxy"]


def test_loopback_no_proxy_sets_both_keys_when_empty(export_mod, monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    export_mod._loopback_no_proxy()

    expected = "127.0.0.1,localhost,::1"
    assert os.environ["NO_PROXY"] == expected
    assert os.environ["no_proxy"] == expected


def test_main_returns_1_when_batch_has_failures(export_mod, monkeypatch, tmp_path):
    inp = tmp_path / "doc.pdf"
    inp.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_parse_json_no_llm",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )

    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def process_batch(self, **kwargs):
            captured["process"] = kwargs
            return SimpleNamespace(
                summary=lambda: "1 failed",
                failed_files=["doc.pdf"],
                errors={"doc.pdf": "timeout"},
            )

    with (
        patch.object(export_mod, "_venv_path"),
        patch("raganything.batch_parser.BatchParser", FakeBatchParser),
    ):
        code = export_mod.main()

    assert code == 1
    assert captured["init"]["parser_type"] == "mineru"


def test_main_returns_0_on_success(export_mod, monkeypatch, tmp_path):
    inp = tmp_path / "docs"
    inp.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_parse_json_no_llm",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )

    class FakeBatchParser:
        def __init__(self, **kwargs):
            pass

        def process_batch(self, **kwargs):
            return SimpleNamespace(
                summary=lambda: "ok",
                failed_files=[],
                errors={},
            )

    with (
        patch.object(export_mod, "_venv_path"),
        patch("raganything.batch_parser.BatchParser", FakeBatchParser),
    ):
        code = export_mod.main()

    assert code == 0


def test_main_clamps_timeout_and_workers(export_mod, monkeypatch, tmp_path):
    inp = tmp_path / "doc.pdf"
    inp.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_parse_json_no_llm",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--max-workers",
            "0",
            "--timeout-per-file",
            "10",
        ],
    )

    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def process_batch(self, **kwargs):
            return SimpleNamespace(summary=lambda: "ok", failed_files=[], errors={})

    with (
        patch.object(export_mod, "_venv_path"),
        patch("raganything.batch_parser.BatchParser", FakeBatchParser),
    ):
        export_mod.main()

    assert captured["max_workers"] == 1
    assert captured["timeout_per_file"] == 300
