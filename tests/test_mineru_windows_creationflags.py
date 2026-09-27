"""MinerU subprocess must hide the console on Windows and stay portable on POSIX.

LibreOffice and Docling already pass CREATE_NO_WINDOW. MinerU is the primary
parser and spawns via Popen, so dropping the flag pops a console (or hangs a
headless Windows service) without failing the parse.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

_CREATE_NO_WINDOW = 0x08000000


def _load_parser_module():
    """Load parser.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_mineru_windows", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()
MineruParser = parser_module.MineruParser


def _ready_process():
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


def test_mineru_command_passes_create_no_window_on_windows(monkeypatch):
    monkeypatch.setattr(parser_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        parser_module.subprocess,
        "CREATE_NO_WINDOW",
        _CREATE_NO_WINDOW,
        raising=False,
    )

    mock_popen = MagicMock(return_value=_ready_process())
    monkeypatch.setattr(parser_module.subprocess, "Popen", mock_popen)

    MineruParser._run_mineru_command("manual.pdf", "out")

    assert mock_popen.call_args.kwargs.get("creationflags") == _CREATE_NO_WINDOW


def test_mineru_command_omits_creationflags_on_posix(monkeypatch):
    monkeypatch.setattr(parser_module, "_IS_WINDOWS", False)

    mock_popen = MagicMock(return_value=_ready_process())
    monkeypatch.setattr(parser_module.subprocess, "Popen", mock_popen)

    MineruParser._run_mineru_command("manual.pdf", "out")

    assert "creationflags" not in mock_popen.call_args.kwargs
