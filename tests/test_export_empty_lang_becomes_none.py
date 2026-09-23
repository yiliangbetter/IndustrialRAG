"""Export ``--lang ""`` must reach MinerU as ``lang=None``.

``scripts/export_parse_json_no_llm.py`` forwards ``lang=args.lang or None``.
An empty language tag is the operator's way to omit MinerU ``-l`` rather than
force the Chinese default. Treating ``""`` as ``"ch"`` would OCR English
manuals with the wrong language model. Non-empty tags are already covered by
#180; this only locks the empty-string falsy path.
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
        "export_empty_lang_becomes_none_37e0", _SCRIPT
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


def test_empty_lang_flag_forwards_none(export_mod, monkeypatch, tmp_path):
    inp = tmp_path / "manual.pdf"
    inp.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "out"
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    _CapturingBatchParser.last_process = None

    with (
        patch.object(export_mod, "_venv_path"),
        patch.object(export_mod, "_loopback_no_proxy"),
        patch("raganything.batch_parser.BatchParser", _CapturingBatchParser),
        patch.object(
            export_mod.sys,
            "argv",
            [
                "export_parse_json_no_llm",
                "--input",
                str(inp),
                "--output",
                str(out),
                "--lang",
                "",
            ],
        ),
    ):
        code = export_mod.main()

    assert code == 0
    process = _CapturingBatchParser.last_process
    assert process["lang"] is None
    assert process["parse_method"] == "auto"
    assert process["backend"] == "pipeline"
