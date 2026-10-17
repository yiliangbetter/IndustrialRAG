"""MinerU must merge differing NO_PROXY and no_proxy lists for the child only.

Corporate proxies and lowercase no_proxy often disagree. The MinerU runner
reads both, drops blank tokens, keeps the first occurrence of each host, and
appends 127.0.0.1, localhost, and ::1 when missing. The merged list is written
to both keys on the child environment. The parent process environment stays
unchanged so a parse does not rewrite the caller's proxy bypass list.
"""

import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSER_PATH = REPO_ROOT / "raganything" / "parser.py"

_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _load_mineru_parser():
    name = "_raganything_parser_proxy_split_6585"
    spec = importlib.util.spec_from_file_location(name, PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MineruParser


MineruParser = _load_mineru_parser()


def _ready_process():
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


def _parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@patch("subprocess.Popen")
def test_split_proxy_lists_merge_onto_the_child_only(mock_popen, monkeypatch):
    parent_no_proxy = "corp.example,  10.0.0.1 "
    parent_lower = "10.0.0.1, cache.internal,,"
    monkeypatch.setenv("NO_PROXY", parent_no_proxy)
    monkeypatch.setenv("no_proxy", parent_lower)
    mock_popen.return_value = _ready_process()

    MineruParser._run_mineru_command("scan.pdf", "out")

    env = mock_popen.call_args.kwargs["env"]
    assert env is not os.environ
    parts = _parts(env["NO_PROXY"])
    assert parts == [
        "corp.example",
        "10.0.0.1",
        "cache.internal",
        *_LOOPBACK,
    ]
    assert env["NO_PROXY"] == env["no_proxy"]
    assert parts.count("10.0.0.1") == 1
    assert os.environ["NO_PROXY"] == parent_no_proxy
    assert os.environ["no_proxy"] == parent_lower


@patch("subprocess.Popen")
def test_custom_lowercase_no_proxy_keeps_uppercase_parent_hosts(
    mock_popen, monkeypatch
):
    monkeypatch.setenv("NO_PROXY", "corp.example")
    monkeypatch.setenv("no_proxy", "old.internal")
    mock_popen.return_value = _ready_process()

    MineruParser._run_mineru_command(
        "scan.pdf",
        "out",
        env={"no_proxy": "cache.internal"},
    )

    env = mock_popen.call_args.kwargs["env"]
    parts = _parts(env["NO_PROXY"])
    assert parts == ["corp.example", "cache.internal", *_LOOPBACK]
    assert "old.internal" not in parts
    assert env["NO_PROXY"] == env["no_proxy"]
    assert os.environ["NO_PROXY"] == "corp.example"
    assert os.environ["no_proxy"] == "old.internal"
