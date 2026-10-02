"""Same-stem Office/text sources must not share one converted PDF."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import importlib.util


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_same_stem", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()
Parser = parser_module.Parser


def _fake_libreoffice(cmd, **kwargs):
    outdir = Path(cmd[cmd.index("--outdir") + 1])
    source = Path(cmd[-1])
    payload = b"%PDF-1.4\n" + source.read_bytes() + (b"\n" + b"x" * 120)
    (outdir / f"{source.stem}.pdf").write_bytes(payload)
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_same_stem_office_files_keep_separate_pdfs(tmp_path):
    left = tmp_path / "line_a" / "manual.docx"
    right = tmp_path / "line_b" / "manual.docx"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"line-a-manual")
    right.write_bytes(b"line-b-manual")
    output_dir = tmp_path / "output"

    with patch.object(parser_module.subprocess, "run", side_effect=_fake_libreoffice):
        left_pdf = Parser.convert_office_to_pdf(left, str(output_dir))
        right_pdf = Parser.convert_office_to_pdf(right, str(output_dir))

    assert left_pdf != right_pdf
    assert left_pdf.exists() and right_pdf.exists()
    assert b"line-a-manual" in left_pdf.read_bytes()
    assert b"line-b-manual" in right_pdf.read_bytes()

    # MinerU keys its output directory by the converted PDF path.
    left_parse_dir = Parser._unique_output_dir(output_dir, left_pdf)
    right_parse_dir = Parser._unique_output_dir(output_dir, right_pdf)
    assert left_parse_dir != right_parse_dir


def test_same_directory_doc_and_docx_do_not_clobber(tmp_path):
    doc = tmp_path / "manual.doc"
    docx = tmp_path / "manual.docx"
    doc.write_bytes(b"legacy-doc")
    docx.write_bytes(b"modern-docx")
    output_dir = tmp_path / "output"

    with patch.object(parser_module.subprocess, "run", side_effect=_fake_libreoffice):
        doc_pdf = Parser.convert_office_to_pdf(doc, str(output_dir))
        docx_pdf = Parser.convert_office_to_pdf(docx, str(output_dir))

    assert doc_pdf != docx_pdf
    assert b"legacy-doc" in doc_pdf.read_bytes()
    assert b"modern-docx" in docx_pdf.read_bytes()


def test_reconverting_same_source_reuses_pdf_path(tmp_path):
    source = tmp_path / "manual.docx"
    source.write_bytes(b"same-manual")
    output_dir = tmp_path / "output"

    with patch.object(parser_module.subprocess, "run", side_effect=_fake_libreoffice):
        first = Parser.convert_office_to_pdf(source, str(output_dir))
        second = Parser.convert_office_to_pdf(source, str(output_dir))

    assert first == second
    assert b"same-manual" in second.read_bytes()
