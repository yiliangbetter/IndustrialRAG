"""MinerU parse_pdf output-dir wiring and fail-closed errors.

_unique_output_dir is unit-tested elsewhere. PDF ingest is the plant-manual
path: colliding same-basename PDFs in a shared output_dir silently mix
content lists. Missing files and MinerU subprocess failures must surface as
the original exception types so ProcessorMixin can dispatch parse errors.
"""

from pathlib import Path

import pytest

from raganything.parser import MineruExecutionError, MineruParser, Parser


def test_parse_pdf_missing_file_raises(tmp_path):
    parser = MineruParser()
    missing = tmp_path / "absent.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
        parser.parse_pdf(missing)


def test_parse_pdf_uses_unique_subdir_when_output_dir_provided(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "shared_out"
    seen = {}

    monkeypatch.setattr(parser, "_run_mineru_command", lambda *a, **k: None)

    def fake_read(output_dir, file_stem, method="auto"):
        seen["output_dir"] = Path(output_dir)
        seen["stem"] = file_stem
        seen["method"] = method
        return [{"type": "text", "text": "ok"}], "md"

    monkeypatch.setattr(parser, "_read_output_files", fake_read)

    result = parser.parse_pdf(pdf, output_dir=str(out), method="ocr", lang="en")

    expected = Parser._unique_output_dir(out, pdf)
    assert seen["output_dir"] == expected
    assert expected.parent == out
    assert expected.name.startswith("manual_")
    assert seen["stem"] == "manual"
    assert seen["method"] == "ocr"
    assert result == [{"type": "text", "text": "ok"}]


def test_parse_pdf_same_basename_files_get_distinct_output_dirs(tmp_path, monkeypatch):
    parser = MineruParser()
    dir_a = tmp_path / "line_a"
    dir_b = tmp_path / "line_b"
    dir_a.mkdir()
    dir_b.mkdir()
    pdf_a = dir_a / "spec.pdf"
    pdf_b = dir_b / "spec.pdf"
    pdf_a.write_bytes(b"%PDF-1.4\na\n")
    pdf_b.write_bytes(b"%PDF-1.4\nb\n")
    out = tmp_path / "shared_out"
    captured = []

    monkeypatch.setattr(parser, "_run_mineru_command", lambda *a, **k: None)

    def fake_read(output_dir, file_stem, method="auto"):
        captured.append(Path(output_dir))
        return [{"type": "text", "text": str(output_dir)}], "md"

    monkeypatch.setattr(parser, "_read_output_files", fake_read)

    parser.parse_pdf(pdf_a, output_dir=str(out))
    parser.parse_pdf(pdf_b, output_dir=str(out))

    assert len(captured) == 2
    assert captured[0] != captured[1]
    assert captured[0].parent == out
    assert captured[1].parent == out
    assert captured[0].name.startswith("spec_")
    assert captured[1].name.startswith("spec_")


def test_parse_pdf_default_output_is_sibling_mineru_output(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    seen = {}

    monkeypatch.setattr(parser, "_run_mineru_command", lambda *a, **k: None)

    def fake_read(output_dir, file_stem, method="auto"):
        seen["output_dir"] = Path(output_dir)
        return [], "md"

    monkeypatch.setattr(parser, "_read_output_files", fake_read)

    parser.parse_pdf(pdf)

    assert seen["output_dir"] == pdf.parent / "mineru_output"


def test_parse_pdf_reraises_mineru_execution_error_unwrapped(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = tmp_path / "fail.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    original = MineruExecutionError(2, ["mineru crashed"])

    def boom(*_a, **_k):
        raise original

    monkeypatch.setattr(parser, "_run_mineru_command", boom)

    with pytest.raises(MineruExecutionError) as exc_info:
        parser.parse_pdf(pdf, output_dir=str(tmp_path / "out"))

    assert exc_info.value is original
    assert exc_info.value.return_code == 2
