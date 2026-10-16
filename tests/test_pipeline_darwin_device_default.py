"""Pipeline MinerU kwargs must force CPU on Darwin unless a device is set.

``scripts/rag_pipeline_parse_graph_chat.py`` lowercases the parser name, then
injects ``device=cpu`` only for MinerU when ``sys.platform == "darwin"`` and
``MINERU_DEVICE`` is empty. The reingest copy is locked elsewhere. These tests
cover the pipeline copy: mixed-case ``PARSER`` values, Linux not receiving the
Mac default, and a blank device still falling through to CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_KEYS = (
    "MINERU_LANG",
    "OCR_LANG",
    "MINERU_BACKEND",
    "MINERU_SOURCE",
    "MINERU_DEVICE",
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_darwin_device_1016",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clear_mineru_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("parser_name", ["MinerU", "MINERU", "mineru"])
def test_darwin_defaults_cpu_for_mixed_case_mineru_only(
    pipeline, monkeypatch, parser_name
):
    _clear_mineru_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")

    kwargs = pipeline._mineru_parse_kwargs(parser_name)

    assert kwargs["backend"] == "pipeline"
    assert kwargs["device"] == "cpu"
    assert "lang" not in kwargs
    assert "source" not in kwargs

    other = pipeline._mineru_parse_kwargs("PaddleOCR")
    assert other == {}


def test_linux_mixed_case_mineru_does_not_force_cpu(pipeline, monkeypatch):
    _clear_mineru_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")

    kwargs = pipeline._mineru_parse_kwargs("MinerU")

    assert kwargs == {"backend": "pipeline"}


def test_explicit_device_wins_and_blank_device_uses_darwin_cpu(pipeline, monkeypatch):
    _clear_mineru_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")

    monkeypatch.setenv("MINERU_DEVICE", "  mps  ")
    assert pipeline._mineru_parse_kwargs("MinerU")["device"] == "mps"

    monkeypatch.setenv("MINERU_DEVICE", " \t ")
    kwargs = pipeline._mineru_parse_kwargs("MinerU")
    assert kwargs["backend"] == "pipeline"
    assert kwargs["device"] == "cpu"
