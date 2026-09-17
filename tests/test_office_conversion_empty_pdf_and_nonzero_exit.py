"""LibreOffice conversion must fail closed on empty PDFs and keep trying after a nonzero exit.

Office manuals are converted before MinerU/Docling OCR. A "successful" conversion
that writes a tiny/corrupt PDF would ingest garbage. A nonzero exit from
``libreoffice`` must still try ``soffice``. Missing sources must raise rather
than calling the converter. Distinct from open #140 (binary discovery / timeout)
and #150 (generic exception fall-through).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PARSER_PATH = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"


def _load_parser_module():
    spec = importlib.util.spec_from_file_location(
        "raganything_parser_office_fail_closed_d37f", _PARSER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parser_mod():
    return _load_parser_module()


def _docx(tmp_path: Path) -> Path:
    src = tmp_path / "manual.docx"
    src.write_bytes(b"PK\x03\x04fake")
    return src


def _ok_pdf(cmd, stem: str, size: int = 200) -> SimpleNamespace:
    outdir = Path(cmd[cmd.index("--outdir") + 1])
    (outdir / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * size)
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_missing_office_file_raises_before_subprocess(
    parser_mod, monkeypatch, tmp_path
):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise AssertionError("LibreOffice must not run for a missing source file")

    monkeypatch.setattr(parser_mod.sys, "platform", "linux")
    monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)

    missing = tmp_path / "gone.docx"
    with pytest.raises(FileNotFoundError, match="Office document does not exist"):
        parser_mod.Parser.convert_office_to_pdf(
            missing, output_dir=str(tmp_path / "out")
        )
    assert calls == []


def test_nonzero_exit_on_libreoffice_falls_through_to_soffice(
    parser_mod, monkeypatch, tmp_path
):
    src = _docx(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "libreoffice":
            return SimpleNamespace(returncode=1, stdout="", stderr="headless failed")
        return _ok_pdf(cmd, src.stem)

    monkeypatch.setattr(parser_mod.sys, "platform", "linux")
    monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)

    pdf_path = parser_mod.Parser.convert_office_to_pdf(
        src, output_dir=str(tmp_path / "pdf_out")
    )

    assert calls == ["libreoffice", "soffice"]
    assert pdf_path.exists()
    assert pdf_path.name == "manual.pdf"
    assert pdf_path.stat().st_size >= 100


def test_successful_conversion_rejects_tiny_corrupt_pdf(
    parser_mod, monkeypatch, tmp_path
):
    src = _docx(tmp_path)

    def fake_run(cmd, **kwargs):
        return _ok_pdf(cmd, src.stem, size=10)

    monkeypatch.setattr(parser_mod.sys, "platform", "linux")
    monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="empty or corrupted"):
        parser_mod.Parser.convert_office_to_pdf(
            src, output_dir=str(tmp_path / "pdf_out")
        )
