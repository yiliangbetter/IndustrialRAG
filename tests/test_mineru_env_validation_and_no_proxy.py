"""MinerU subprocess env validation and loopback NO_PROXY merge.

Corporate HTTP(S)_PROXY breaks mineru-api /health on loopback (502). The
command runner must inject 127.0.0.1/localhost/::1 into NO_PROXY/no_proxy
without dropping existing entries, and reject malformed env/kwargs before
spawning a process.
"""

from unittest.mock import MagicMock, patch

import pytest

from raganything.parser import MineruParser


def _ready_process(return_code=0):
    process = MagicMock()
    process.poll.return_value = return_code
    process.wait.return_value = return_code
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


def test_env_must_be_a_string_dictionary():
    parser = MineruParser()
    with pytest.raises(TypeError, match="env must be a dictionary"):
        parser._run_mineru_command("scan.pdf", "out", env="BAD")


def test_env_keys_and_values_must_be_strings():
    parser = MineruParser()
    with pytest.raises(TypeError, match="env keys and values must be strings"):
        parser._run_mineru_command("scan.pdf", "out", env={1: "x"})
    with pytest.raises(TypeError, match="env keys and values must be strings"):
        parser._run_mineru_command("scan.pdf", "out", env={"FOO": 1})


def test_unexpected_kwargs_fail_fast_before_spawn():
    parser = MineruParser()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        parser._run_mineru_command("scan.pdf", "out", typo_flag=True)


@patch("subprocess.Popen")
def test_loopback_hosts_are_added_to_no_proxy(mock_popen, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "corp.local")
    monkeypatch.delenv("no_proxy", raising=False)
    mock_popen.return_value = _ready_process()

    MineruParser()._run_mineru_command("scan.pdf", "out")

    env = mock_popen.call_args.kwargs["env"]
    parts = [p.strip() for p in env["NO_PROXY"].split(",") if p.strip()]
    assert parts[0] == "corp.local"
    assert "127.0.0.1" in parts
    assert "localhost" in parts
    assert "::1" in parts
    assert env["NO_PROXY"] == env["no_proxy"]
    assert len(parts) == len(set(parts))


@patch("subprocess.Popen")
def test_custom_env_is_merged_and_still_gets_loopback_no_proxy(mock_popen, monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    mock_popen.return_value = _ready_process()

    MineruParser()._run_mineru_command(
        "scan.pdf",
        "out",
        env={"FOO": "bar", "NO_PROXY": "intranet.local"},
    )

    env = mock_popen.call_args.kwargs["env"]
    assert env["FOO"] == "bar"
    parts = [p.strip() for p in env["NO_PROXY"].split(",") if p.strip()]
    assert "intranet.local" in parts
    assert "127.0.0.1" in parts
    assert "localhost" in parts
    assert "::1" in parts
    assert env["NO_PROXY"] == env["no_proxy"]


@patch("subprocess.Popen")
def test_existing_loopback_entries_are_not_duplicated(mock_popen, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1")
    mock_popen.return_value = _ready_process()

    MineruParser()._run_mineru_command("scan.pdf", "out")

    env = mock_popen.call_args.kwargs["env"]
    parts = [p.strip() for p in env["NO_PROXY"].split(",") if p.strip()]
    assert parts.count("127.0.0.1") == 1
    assert parts.count("localhost") == 1
    assert parts.count("::1") == 1
