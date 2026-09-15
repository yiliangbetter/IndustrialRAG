"""Export CLI must forward MinerU parse knobs into BatchParser.process_batch.

``scripts/export_parse_json_no_llm.py`` is the parse-only path for industrial
manuals. Dropping ``--lang``/``--method``/``--source`` or inverting
``--no-formula``/``--no-table`` silently changes OCR quality and artifact
layout. Distinct from #130 (NO_PROXY merge, exit codes, worker/timeout clamp).
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
        "export_parse_json_forwards_mineru_kwargs_1c11", _SCRIPT
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
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        type(self).last_init = kwargs
        type(self).last_process = None

    def process_batch(self, **kwargs):
        type(self).last_process = kwargs
        return SimpleNamespace(summary=lambda: "ok", failed_files=[], errors={})


def _run_export(export_mod, argv):
    with (
        patch.object(export_mod, "_venv_path"),
        patch.object(export_mod, "_loopback_no_proxy"),
        patch("raganything.batch_parser.BatchParser", _CapturingBatchParser),
        patch.object(export_mod.sys, "argv", argv),
    ):
        return export_mod.main()


def test_export_defaults_forward_auto_chinese_pipeline_modelscope(
    export_mod, monkeypatch, tmp_path
):
    inp = tmp_path / "manual.pdf"
    inp.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)

    code = _run_export(
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
    assert process["file_paths"] == [str(inp.resolve())]
    assert process["output_dir"] == str(out.resolve())
    assert process["parse_method"] == "auto"
    assert process["recursive"] is True
    assert process["lang"] == "ch"
    assert process["backend"] == "pipeline"
    assert process["source"] == "modelscope"
    assert process["formula"] is True
    assert process["table"] is True
    assert "device" not in process
    assert os.environ["MINERU_MODEL_SOURCE"] == "modelscope"


def test_export_flags_forward_ocr_lang_source_device_and_disable_formula_table(
    export_mod, monkeypatch, tmp_path
):
    inp = tmp_path / "docs"
    inp.mkdir()
    out = tmp_path / "parsed"
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "huggingface")

    code = _run_export(
        export_mod,
        [
            "export_parse_json_no_llm",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--method",
            "ocr",
            "--lang",
            "en",
            "--backend",
            "vlm",
            "--source",
            "huggingface",
            "--device",
            "cpu",
            "--no-formula",
            "--no-table",
        ],
    )

    assert code == 0
    process = _CapturingBatchParser.last_process
    assert process["file_paths"] == [str(inp.resolve())]
    assert process["parse_method"] == "ocr"
    assert process["lang"] == "en"
    assert process["backend"] == "vlm"
    assert process["source"] == "huggingface"
    assert process["device"] == "cpu"
    assert process["formula"] is False
    assert process["table"] is False
    # setdefault must not override an existing MinerU source
    assert os.environ["MINERU_MODEL_SOURCE"] == "huggingface"
