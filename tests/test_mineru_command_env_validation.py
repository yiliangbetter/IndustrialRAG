"""MinerU command construction must fail closed on a bad env or unknown flag.

`_run_mineru_command` is the only place that builds the mineru CLI. A list
passed as `env`, or a non-string entry, must raise before Popen. An unknown
keyword must not be dropped or turned into a CLI flag. A string env is copied
into the child process and must not leak into the parent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_mineru_env_18f4", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()
MineruParser = parser_module.MineruParser


def _reject_popen(*_args, **_kwargs):
    raise AssertionError("mineru must not start when command arguments are invalid")


def test_non_dict_env_does_not_start_mineru(monkeypatch):
    monkeypatch.setattr(parser_module.subprocess, "Popen", _reject_popen)

    with pytest.raises(TypeError, match="env must be a dictionary, got list"):
        MineruParser._run_mineru_command(
            input_path="manual.pdf",
            output_dir="out",
            env=["MINERU_MODEL_SOURCE=local"],
        )


@pytest.mark.parametrize(
    "env",
    [
        {"MINERU_DEVICE": 0},
        {1: "cpu"},
    ],
)
def test_non_string_env_entries_do_not_start_mineru(monkeypatch, env):
    monkeypatch.setattr(parser_module.subprocess, "Popen", _reject_popen)

    with pytest.raises(TypeError, match="env keys and values must be strings"):
        MineruParser._run_mineru_command(
            input_path="manual.pdf",
            output_dir="out",
            env=env,
        )


def test_unknown_keyword_does_not_start_mineru(monkeypatch):
    monkeypatch.setattr(parser_module.subprocess, "Popen", _reject_popen)

    with pytest.raises(TypeError, match="not_a_flag") as excinfo:
        MineruParser._run_mineru_command(
            input_path="manual.pdf",
            output_dir="out",
            not_a_flag=True,
            also_unknown="x",
        )

    message = str(excinfo.value)
    assert "also_unknown" in message
    assert "unexpected keyword argument" in message


def test_string_env_is_forwarded_without_mutating_the_parent(monkeypatch):
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)
    monkeypatch.delenv("MINERU_DEVICE", raising=False)
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        raise FileNotFoundError("mineru")

    monkeypatch.setattr(parser_module.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="mineru command not found"):
        MineruParser._run_mineru_command(
            input_path="manual.pdf",
            output_dir="out",
            env={"MINERU_MODEL_SOURCE": "local", "MINERU_DEVICE": ""},
        )

    assert captured["cmd"][0] == "mineru"
    assert captured["env"]["MINERU_MODEL_SOURCE"] == "local"
    assert captured["env"]["MINERU_DEVICE"] == ""
    assert "MINERU_MODEL_SOURCE" not in parser_module.os.environ
    assert "MINERU_DEVICE" not in parser_module.os.environ
