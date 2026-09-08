"""Parser.convert_text_to_pdf output identity and GBK decode order.

Industrial manuals are often GBK-encoded .txt files. If latin-1 is tried first,
Chinese text becomes mojibake in the generated PDF. Omitting output_dir must
write beside the source (reportlab_output/), matching LibreOffice/MinerU
sibling-dir contracts so parse_text_file can find the PDF.
"""

import pytest

from raganything.parser import Parser

pytest.importorskip("reportlab")

GBK_INSTALL = "安装"


def _spy_paragraphs(monkeypatch):
    from reportlab.platypus import Paragraph as RealParagraph

    seen = []

    class SpyParagraph(RealParagraph):
        def __init__(self, text, *args, **kwargs):
            seen.append(text)
            super().__init__(text, *args, **kwargs)

    monkeypatch.setattr("reportlab.platypus.Paragraph", SpyParagraph)
    return seen


def test_omitted_output_dir_writes_sibling_reportlab_output(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("Line one.\n", encoding="utf-8")

    pdf_path = Parser.convert_text_to_pdf(src)

    expected = tmp_path / "reportlab_output" / "notes.pdf"
    assert pdf_path == expected
    assert pdf_path.exists()
    assert pdf_path.stat().st_size >= 100
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_provided_output_dir_is_honored(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("Line one.\n", encoding="utf-8")
    out = tmp_path / "pdf-out"

    pdf_path = Parser.convert_text_to_pdf(src, output_dir=str(out))

    assert pdf_path == out / "notes.pdf"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size >= 100


def test_gbk_txt_is_decoded_before_latin1_fallback(tmp_path, monkeypatch):
    src = tmp_path / "manual.txt"
    src.write_bytes(GBK_INSTALL.encode("gbk"))
    seen = _spy_paragraphs(monkeypatch)

    pdf_path = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

    assert pdf_path.exists()
    assert any(GBK_INSTALL in text for text in seen), seen
    latin1_mojibake = GBK_INSTALL.encode("gbk").decode("latin-1")
    assert all(latin1_mojibake not in text for text in seen)
