"""MinerU command runner must fail closed on stderr errors even when exit is 0.

#139 covers timeout, CLI flags, nonzero exit, and missing binary. This file
locks the remaining gate: MinerU can exit 0 while still printing ``error`` on
stderr; ingest must not treat that as success. Warnings-only stderr must not
trip the same gate. Spawn errors other than FileNotFoundError must wrap as
RuntimeError so callers get a consistent install/runtime failure.
"""

from unittest.mock import patch

import pytest

from raganything.parser import MineruExecutionError, MineruParser


class _LinePipe:
    """Minimal file-like object for Popen stdout/stderr reader threads."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""

    def close(self):
        self.closed = True


class _FakeProcess:
    """Stay 'running' until reader threads close both pipes, then exit 0."""

    def __init__(self, *, stdout_lines=(), stderr_lines=(), return_code=0):
        self.stdout = _LinePipe(stdout_lines)
        self.stderr = _LinePipe(stderr_lines)
        self.return_code = return_code

    def poll(self):
        if not self.stdout.closed or not self.stderr.closed:
            return None
        return self.return_code

    def wait(self):
        return self.return_code

    def kill(self):
        pass


@patch("raganything.parser.time.sleep", lambda *_args, **_kwargs: None)
@patch("subprocess.Popen")
def test_exit_zero_with_stderr_error_raises_execution_error(mock_popen):
    mock_popen.return_value = _FakeProcess(stderr_lines=["ERROR: cuda OOM\n"])
    parser = MineruParser()

    with pytest.raises(MineruExecutionError) as excinfo:
        parser._run_mineru_command("scan.pdf", "out")

    assert excinfo.value.return_code == 0
    assert any("ERROR: cuda OOM" in str(item) for item in excinfo.value.error_msg)


@patch("raganything.parser.time.sleep", lambda *_args, **_kwargs: None)
@patch("subprocess.Popen")
def test_exit_zero_with_warning_only_stderr_succeeds(mock_popen, caplog):
    mock_popen.return_value = _FakeProcess(
        stderr_lines=["warning: using CPU fallback\n"]
    )
    parser = MineruParser()

    with caplog.at_level("WARNING", logger="raganything.parser"):
        parser._run_mineru_command("scan.pdf", "out")

    assert any("using CPU fallback" in record.message for record in caplog.records)


@patch("subprocess.Popen")
def test_permission_error_on_spawn_is_wrapped_as_runtime_error(mock_popen):
    mock_popen.side_effect = PermissionError("mineru: permission denied")
    parser = MineruParser()

    with pytest.raises(RuntimeError, match="Unexpected error running mineru command"):
        parser._run_mineru_command("scan.pdf", "out")
