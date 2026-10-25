"""MinerU formula and table switches must emit explicit disable flags.

``formula=False`` adds ``-f false`` and ``table=False`` adds ``-t false``.
Leaving either switch on must not emit that flag, so disabling formulas does
not also disable tables, and the reverse.
"""

from unittest.mock import MagicMock, patch

import pytest

from raganything.parser import MineruParser


def _ready_process():
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


def _flag(cmd, name):
    if name not in cmd:
        return None
    return cmd[cmd.index(name) + 1]


@pytest.mark.parametrize(
    ("formula", "table", "expect_formula", "expect_table"),
    [
        (False, True, "false", None),
        (True, False, None, "false"),
        (False, False, "false", "false"),
    ],
)
@patch("subprocess.Popen")
def test_disabled_formula_and_table_emit_false_flags(
    mock_popen, formula, table, expect_formula, expect_table
):
    mock_popen.return_value = _ready_process()

    MineruParser._run_mineru_command(
        "scan.pdf",
        "out",
        method="ocr",
        formula=formula,
        table=table,
    )

    cmd = mock_popen.call_args[0][0]
    assert cmd[:6] == ["mineru", "-p", "scan.pdf", "-o", "out", "-m"]
    assert _flag(cmd, "-m") == "ocr"
    assert _flag(cmd, "-f") == expect_formula
    assert _flag(cmd, "-t") == expect_table
    if expect_formula and expect_table:
        assert cmd.index("-f") < cmd.index("-t")
