"""Export omitted ``--source`` must read ``MINERU_MODEL_SOURCE``.

``scripts/export_parse_json_no_llm.py`` defaults ``--source`` from
``os.getenv("MINERU_MODEL_SOURCE", "modelscope")``. Hardcoding modelscope
would still ``setdefault`` an existing env, but ``process_batch(source=...)``
would send the wrong MinerU hub. Distinct from #180 (unset env → modelscope,
and explicit ``--source`` flags).
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
    spec = importlib.util.spec_from_file_location(
        "export_omitted_source_reads_mineru_env_8a56", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_mod():
    return _load_export_module()


@pytest.fixture(autouse=True)
def _restore_mineru_source_env():
    before = os.environ.get("MINERU_MODEL_SOURCE")
    yield
    if before is None:
        os.environ.pop("MINERU_MODEL_SOURCE", None)
    else:
        os.environ["MINERU_MODEL_SOURCE"] = before


class _CapturingBatchParser:
    last_process = None

    def __init__(self, **kwargs):
        pass

    def process_batch(self, **kwargs):
        type(self).last_process = kwargs
        return SimpleNamespace(summary=lambda: "ok", failed_files=[], errors={})


def _run_export_without_source_flag(export_mod, argv):
    _CapturingBatchParser.last_process = None
    with (
        patch.object(export_mod, "_venv_path"),
        patch.object(export_mod, "_loopback_no_proxy"),
        patch("raganything.batch_parser.BatchParser", _CapturingBatchParser),
        patch.object(export_mod.sys, "argv", argv),
    ):
        return export_mod.main()


def test_omitted_source_forwards_huggingface_env(export_mod, monkeypatch, tmp_path):
    inp = tmp_path / "manual.pdf"
    inp.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "huggingface")

    code = _run_export_without_source_flag(
        export_mod,
        [
            "export_parse_json_no_llm",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )

    assert code == 0
    process = _CapturingBatchParser.last_process
    assert process["source"] == "huggingface"
    assert os.environ["MINERU_MODEL_SOURCE"] == "huggingface"


def test_omitted_source_forwards_local_env(export_mod, monkeypatch, tmp_path):
    inp = tmp_path / "docs"
    inp.mkdir()
    out = tmp_path / "parsed"
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "local")

    code = _run_export_without_source_flag(
        export_mod,
        [
            "export_parse_json_no_llm",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
    )

    assert code == 0
    process = _CapturingBatchParser.last_process
    assert process["source"] == "local"
    assert os.environ["MINERU_MODEL_SOURCE"] == "local"
