"""MinerU must keep an explicit last page of 0 on the command line.

``end_page=0`` selects the first page. The flag is appended only when the
value is not None. Treating 0 as missing drops ``-e`` and MinerU parses the
whole document. ``start_page=0`` is already locked with a non-zero end page;
this checks the end bound on its own.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


def _load_parser():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_end_page_zero_7d5b",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser()


def _ready_process():
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


def test_end_page_zero_is_passed_and_start_page_stays_omitted(monkeypatch):
    mock_popen = MagicMock(return_value=_ready_process())
    monkeypatch.setattr(parser_module.subprocess, "Popen", mock_popen)

    parser_module.MineruParser._run_mineru_command(
        "cover.pdf",
        "out",
        method="ocr",
        end_page=0,
    )

    cmd = mock_popen.call_args[0][0]
    assert cmd[cmd.index("-e") + 1] == "0"
    assert "-s" not in cmd
    assert "-f" not in cmd
    assert "-t" not in cmd
