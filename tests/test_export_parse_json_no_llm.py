"""Regression tests for scripts/export_parse_json_no_llm.py helpers.

MinerU nested API calls hang when loopback is proxied. The export script must
merge 127.0.0.1/localhost/::1 into NO_PROXY/no_proxy without dropping existing
bypass entries or duplicating hosts.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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
