"""Markdown-to-PDF must escape ReportLab markup before building paragraphs.

`.txt` conversion already escapes `&`, `<`, and `>`. `.md` conversion passed
raw lines to ReportLab, so a supported markdown file containing `<br>` or
`x<y` raised during ingest and the document was dropped.
"""

from pathlib import Path

import pytest

pytest.importorskip("reportlab")


def _load_parser_class():
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location("_raganything_parser_md", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser


Parser = _load_parser_class()


def test_markdown_with_reportlab_markup_converts(tmp_path):
    md_path = tmp_path / "manual.md"
    md_path.write_text(
        "# R&D notes\n\n"
        "Break here<br>and continue.\n\n"
        "Compare x<y and if (a<b && c>d).\n",
        encoding="utf-8",
    )

    pdf_path = Parser.convert_text_to_pdf(md_path, output_dir=str(tmp_path))

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 100
    assert pdf_path.suffix == ".pdf"
