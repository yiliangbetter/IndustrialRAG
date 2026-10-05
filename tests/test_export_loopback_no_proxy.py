"""Parse export must keep loopback MinerU calls off the corporate proxy.

``scripts/export_parse_json_no_llm.py`` merges ``127.0.0.1``, ``localhost``,
and ``::1`` into both ``NO_PROXY`` and ``no_proxy`` before it builds a
``BatchParser``. Existing bypass entries must survive, blank tokens must be
dropped, and hosts already listed must not be duplicated. A missing call here
sends mineru-api ``/health`` through ``HTTP(S)_PROXY`` and the export fails
with a loopback 502.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_parse_json_no_llm.py"

_ENV_KEYS = (
    "NO_PROXY",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "PATH",
    "MINERU_MODEL_SOURCE",
)
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_raganything_export_loopback_c885", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def export_script():
    return _load_script_module()


@pytest.fixture(autouse=True)
def _restore_proxy_env():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _proxy_parts() -> list[str]:
    raw = os.environ["NO_PROXY"]
    assert os.environ["no_proxy"] == raw
    return [part.strip() for part in raw.split(",") if part.strip()]


def test_merges_existing_entries_and_adds_missing_loopback(export_script, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "corp.example,  localhost ,,")
    monkeypatch.setenv("no_proxy", "api.internal,127.0.0.1")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")

    export_script._loopback_no_proxy()

    assert _proxy_parts() == [
        "corp.example",
        "localhost",
        "api.internal",
        "127.0.0.1",
        "::1",
    ]
    assert os.environ["HTTP_PROXY"] == "http://proxy.internal:8080"
    assert os.environ["HTTPS_PROXY"] == "http://proxy.internal:8080"


def test_empty_proxy_env_sets_only_loopback_hosts(export_script, monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    export_script._loopback_no_proxy()

    assert _proxy_parts() == list(_LOOPBACK)


def test_second_call_does_not_duplicate_hosts(export_script, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1,corp.example")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1,corp.example")

    export_script._loopback_no_proxy()
    once = _proxy_parts()
    export_script._loopback_no_proxy()

    assert once == ["127.0.0.1", "localhost", "::1", "corp.example"]
    assert _proxy_parts() == once


def test_main_applies_loopback_bypass_before_batch_parser(
    export_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("NO_PROXY", "corp.example")
    monkeypatch.delenv("no_proxy", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    seen: dict[str, str | None] = {}

    class _Result:
        failed_files: list[str] = []
        errors: dict[str, str] = {}

        def summary(self) -> str:
            return "ok"

    class _BatchParser:
        def __init__(self, **kwargs):
            seen["at_init"] = os.environ.get("NO_PROXY")
            seen["no_proxy_at_init"] = os.environ.get("no_proxy")

        def process_batch(self, **kwargs):
            seen["during_batch"] = os.environ.get("NO_PROXY")
            return _Result()

    package = types.ModuleType("raganything")
    package.__path__ = []
    batch_parser_mod = types.ModuleType("raganything.batch_parser")
    batch_parser_mod.BatchParser = _BatchParser
    monkeypatch.setitem(sys.modules, "raganything", package)
    monkeypatch.setitem(sys.modules, "raganything.batch_parser", batch_parser_mod)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_parse_json_no_llm",
            "--input",
            str(docs),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert export_script.main() == 0

    assert seen["at_init"] == seen["no_proxy_at_init"] == seen["during_batch"]
    parts = [part.strip() for part in seen["at_init"].split(",") if part.strip()]
    assert parts[0] == "corp.example"
    assert parts[1:] == list(_LOOPBACK)
