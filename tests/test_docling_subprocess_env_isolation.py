"""Docling's CLI subprocess must stay fail-closed and must not leak env.

``_run_docling_command`` accepts ``**kwargs`` so callers can pass ``env``.
Those extra keywords are not subprocess options. Forwarding ``shell=True`` or
``check=False`` would turn a failed Docling run into a successful empty parse,
or run the argument vector through the shell. An empty ``env`` mapping must
not be passed through: subprocess replaces the child environment when ``env``
is ``{}`` and drops ``PATH``. A real override must be a copy, so the parent
process does not keep the child's variables.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock


def _load_docling_parser():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_docling_env_fd71", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_parser_module = _load_docling_parser()
DoclingParser = _parser_module.DoclingParser


def _patch_run(monkeypatch):
    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=""))
    monkeypatch.setattr(_parser_module.subprocess, "run", mock_run)
    return mock_run


def test_extra_kwargs_cannot_weaken_docling_subprocess(tmp_path, monkeypatch):
    mock_run = _patch_run(monkeypatch)

    DoclingParser()._run_docling_command(
        tmp_path / "manual.pdf",
        tmp_path / "out",
        "manual",
        check=False,
        capture_output=False,
        shell=True,
        timeout=1,
        env={"DOCLING_DEVICE": "cpu"},
    )

    assert mock_run.call_count == 1
    cmd = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs
    assert cmd[0] == "docling"
    assert "--to" in cmd
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert "shell" not in kwargs
    assert "timeout" not in kwargs
    assert kwargs["env"]["DOCLING_DEVICE"] == "cpu"
    assert kwargs["env"]["PATH"] == os.environ["PATH"]
    assert "DOCLING_DEVICE" not in os.environ


def test_empty_env_mapping_does_not_replace_parent_environment(tmp_path, monkeypatch):
    mock_run = _patch_run(monkeypatch)

    DoclingParser()._run_docling_command(
        tmp_path / "manual.pdf",
        tmp_path / "out",
        "manual",
        env={},
    )

    assert mock_run.call_args.kwargs["env"] is None


def test_custom_env_is_copied_and_does_not_mutate_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCLING_PARENT", "parent-value")
    mock_run = _patch_run(monkeypatch)

    custom = {"DOCLING_PARENT": "child-value", "DOCLING_ONLY": "secret"}
    DoclingParser()._run_docling_command(
        tmp_path / "manual.pdf",
        tmp_path / "out",
        "manual",
        env=custom,
    )

    passed = mock_run.call_args.kwargs["env"]
    assert passed is not custom
    assert passed is not os.environ
    assert passed["DOCLING_PARENT"] == "child-value"
    assert passed["DOCLING_ONLY"] == "secret"
    assert passed["PATH"] == os.environ["PATH"]
    assert os.environ["DOCLING_PARENT"] == "parent-value"
    assert "DOCLING_ONLY" not in os.environ

    custom["DOCLING_ONLY"] = "mutated"
    custom["INJECTED"] = "nope"
    assert passed["DOCLING_ONLY"] == "secret"
    assert "INJECTED" not in passed
