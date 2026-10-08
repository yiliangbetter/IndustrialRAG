"""Blank MinerU option strings must not become empty CLI flags.

``lang=""`` is falsy, so ``-l`` is omitted. Treating an empty string like a
provided value would emit ``-l ""`` and MinerU would consume the next token
as the language. The same rule applies to backend, source, device, and vlm_url.
"""

from unittest.mock import MagicMock, patch

from raganything.parser import MineruParser


def _ready_process():
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


@patch("subprocess.Popen")
def test_blank_optional_mineru_flags_are_omitted(mock_popen):
    mock_popen.return_value = _ready_process()

    MineruParser._run_mineru_command(
        "scan.pdf",
        "out",
        method="auto",
        lang="",
        backend="",
        source="",
        device="",
        vlm_url="",
    )

    cmd = mock_popen.call_args[0][0]
    assert cmd[:6] == ["mineru", "-p", "scan.pdf", "-o", "out", "-m"]
    assert cmd[cmd.index("-m") + 1] == "auto"
    for flag in ("-l", "-b", "--source", "-d", "-u", "-f", "-t", "-s", "-e"):
        assert flag not in cmd
