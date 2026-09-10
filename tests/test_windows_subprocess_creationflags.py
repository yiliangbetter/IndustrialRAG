"""Windows parser subprocesses must hide the console window.

LibreOffice conversion and Docling CLI already have POSIX fail-closed tests.
If CREATE_NO_WINDOW is dropped, Windows ingest pops a console (or hangs in
service sessions) even when the command succeeds.
"""

from pathlib import Path
from types import SimpleNamespace

import raganything.parser as parser_module
from raganything.parser import DoclingParser, Parser

_CREATE_NO_WINDOW = 0x08000000


def _enable_windows(monkeypatch):
    monkeypatch.setattr(parser_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        parser_module.subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW, raising=False
    )


def _docx(tmp_path: Path) -> Path:
    doc = tmp_path / "memo.docx"
    doc.write_bytes(b"PK\x03\x04")
    return doc


def test_office_conversion_passes_create_no_window_on_windows(monkeypatch, tmp_path):
    _enable_windows(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "memo.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 200)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(parser_module.subprocess, "run", fake_run)

    result = Parser.convert_office_to_pdf(
        _docx(tmp_path), output_dir=str(tmp_path / "out")
    )

    assert captured.get("creationflags") == _CREATE_NO_WINDOW
    assert result.name == "memo.pdf"


def test_office_conversion_omits_creationflags_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(parser_module, "_IS_WINDOWS", False)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "memo.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 200)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(parser_module.subprocess, "run", fake_run)

    Parser.convert_office_to_pdf(_docx(tmp_path), output_dir=str(tmp_path / "out"))

    assert "creationflags" not in captured


def test_docling_command_passes_create_no_window_on_windows(monkeypatch, tmp_path):
    _enable_windows(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(parser_module.subprocess, "run", fake_run)

    pdf_path = tmp_path / "drawing.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    DoclingParser()._run_docling_command(pdf_path, tmp_path / "out", "drawing")

    assert captured.get("creationflags") == _CREATE_NO_WINDOW


def test_docling_command_omits_creationflags_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(parser_module, "_IS_WINDOWS", False)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(parser_module.subprocess, "run", fake_run)

    pdf_path = tmp_path / "drawing.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    DoclingParser()._run_docling_command(pdf_path, tmp_path / "out", "drawing")

    assert "creationflags" not in captured
