"""Export CLI omitted paths must stay on uploaded_documents and v4 output.

``scripts/export_parse_json_no_llm.py`` is the parse-only path for industrial
manuals. The default output tree is ``output/data_upload_test_v4`` so MinerU
artifacts do not land in the v3 graph-ingest corpus. Distinct from #180
(explicit MinerU flags and passed-in ``--input``/``--output``) and #193
(omitted ``--source`` reading ``MINERU_MODEL_SOURCE``).
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
        "export_parse_json_omitted_io_defaults", _SCRIPT
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


def test_omitted_input_and_output_use_uploaded_documents_and_v4(
    export_mod, monkeypatch, tmp_path
):
    monkeypatch.setattr(export_mod, "_ROOT", tmp_path)
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    monkeypatch.setattr(
        export_mod.sys,
        "argv",
        ["export_parse_json_no_llm"],
    )

    with (
        patch.object(export_mod, "_venv_path"),
        patch.object(export_mod, "_loopback_no_proxy"),
        patch("raganything.batch_parser.BatchParser", _CapturingBatchParser),
    ):
        code = export_mod.main()

    expected_in = (tmp_path / "uploaded_documents").resolve()
    expected_out = (tmp_path / "output" / "data_upload_test_v4").resolve()
    process = _CapturingBatchParser.last_process
    assert code == 0
    assert process["file_paths"] == [str(expected_in)]
    assert process["output_dir"] == str(expected_out)
    assert expected_out.is_dir()
    assert not (tmp_path / "output" / "data_upload_test_v3").exists()
