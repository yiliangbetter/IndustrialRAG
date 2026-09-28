"""Stdout text that contains ``error`` must not fail a zero-exit MinerU run.

``_run_mineru_command`` records stderr lines that contain ``error`` and fails
the parse when that list is non-empty, even if the process exits 0. Stdout is only logged. MinerU progress lines often
include the word ``error`` (confidence, retries). Treating those as failures
would reject a successful parse. Stderr lines that contain ``error`` still
fail the run; this test locks only the stdout side of that split.
"""

from unittest.mock import patch

from raganything.parser import MineruParser


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
    """Stay running until both reader threads close their pipes, then exit 0."""

    def __init__(self, *, stdout_lines=(), stderr_lines=()):
        self.stdout = _LinePipe(stdout_lines)
        self.stderr = _LinePipe(stderr_lines)
        self.return_code = 0

    def poll(self):
        if not self.stdout.closed or not self.stderr.closed:
            return None
        return self.return_code

    def wait(self):
        return self.return_code

    def kill(self):
        raise AssertionError("successful MinerU run must not be killed")


@patch("raganything.parser.time.sleep", lambda *_args, **_kwargs: None)
@patch("subprocess.Popen")
def test_stdout_error_word_with_exit_zero_succeeds(mock_popen):
    mock_popen.return_value = _FakeProcess(
        stdout_lines=["ERROR: layout confidence 0.42 on page 3\n"],
        stderr_lines=["progress: page 3/10\n"],
    )

    MineruParser()._run_mineru_command("scan.pdf", "out")
