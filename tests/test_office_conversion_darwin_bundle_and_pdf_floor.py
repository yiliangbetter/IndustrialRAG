"""Office conversion must not invent a Mac binary, and must keep 100-byte PDFs.

LibreOffice is the gate before MinerU sees Word/PowerPoint manuals. On macOS the
app-bundle ``soffice`` is prepended only when that file exists; a missing bundle
has to stay on ``libreoffice``/``soffice``. A bundle that exits nonzero must
still fall through. Generated PDFs smaller than 100 bytes are rejected; a file
of exactly 100 bytes is a successful conversion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PARSER_PATH = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
_MAC_SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def _load_parser_module():
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_office_floor_73e8", _PARSER_PATH
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


def _pdf_of_size(cmd, total: int) -> SimpleNamespace:
    outdir = Path(cmd[cmd.index("--outdir") + 1])
    payload = b"%PDF" + (b"x" * (total - 4))
    assert len(payload) == total
    (outdir / "out.pdf").write_bytes(payload)
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _patch_mac_bundle(parser_mod, monkeypatch, present: bool) -> None:
    original_is_file = parser_mod.Path.is_file

    def fake_is_file(self):
        if str(self) == _MAC_SOFFICE:
            return present
        return original_is_file(self)

    monkeypatch.setattr(parser_mod.sys, "platform", "darwin")
    monkeypatch.setattr(parser_mod.Path, "is_file", fake_is_file)


def test_darwin_without_app_bundle_uses_path_commands(
    parser_mod, monkeypatch, tmp_path
):
    src = _docx(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        return _pdf_of_size(cmd, 100)

    _patch_mac_bundle(parser_mod, monkeypatch, present=False)
    monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)

    pdf_path = parser_mod.Parser.convert_office_to_pdf(
        src, output_dir=str(tmp_path / "out")
    )

    assert calls == ["libreoffice"]
    assert _MAC_SOFFICE not in calls
    assert pdf_path.name == "manual.pdf"
    assert pdf_path.stat().st_size == 100


def test_darwin_bundle_nonzero_exit_falls_through_to_libreoffice(
    parser_mod, monkeypatch, tmp_path
):
    src = _docx(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == _MAC_SOFFICE:
            return SimpleNamespace(returncode=1, stdout="", stderr="bundle failed")
        return _pdf_of_size(cmd, 120)

    _patch_mac_bundle(parser_mod, monkeypatch, present=True)
    monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)

    pdf_path = parser_mod.Parser.convert_office_to_pdf(
        src, output_dir=str(tmp_path / "out")
    )

    assert calls == [_MAC_SOFFICE, "libreoffice"]
    assert pdf_path.stat().st_size == 120


@pytest.mark.parametrize(
    ("total", "accepted"),
    [
        (100, True),
        (99, False),
    ],
)
def test_pdf_size_floor_is_one_hundred_bytes(
    parser_mod, monkeypatch, tmp_path, total, accepted
):
    src = _docx(tmp_path)

    def fake_run(cmd, **kwargs):
        return _pdf_of_size(cmd, total)

    monkeypatch.setattr(parser_mod.sys, "platform", "linux")
    monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)

    if accepted:
        pdf_path = parser_mod.Parser.convert_office_to_pdf(
            src, output_dir=str(tmp_path / "out")
        )
        assert pdf_path.stat().st_size == 100
    else:
        with pytest.raises(RuntimeError, match="empty or corrupted"):
            parser_mod.Parser.convert_office_to_pdf(
                src, output_dir=str(tmp_path / "out")
            )
