"""txt-to-PDF must escape ReportLab markup and fail closed without reportlab.

Unescaped '<' / '&' in plant manuals is treated as ReportLab XML and can
drop text or crash conversion. Missing reportlab must raise a RuntimeError
with an install hint instead of a raw ImportError.
"""

import builtins
import sys

import pytest

from raganything.parser import Parser


def test_txt_escapes_reportlab_markup_characters(tmp_path, monkeypatch):
    pytest.importorskip("reportlab")
    from reportlab.platypus import Paragraph as RealParagraph

    src = tmp_path / "ratings.txt"
    src.write_text("Use <b>bold</b> & <i>italic</i> tags.\n", encoding="utf-8")
    seen = []

    class SpyParagraph(RealParagraph):
        def __init__(self, text, *args, **kwargs):
            seen.append(text)
            super().__init__(text, *args, **kwargs)

    monkeypatch.setattr("reportlab.platypus.Paragraph", SpyParagraph)

    pdf_path = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

    assert pdf_path.exists()
    assert pdf_path.stat().st_size >= 100
    assert seen, "Paragraph must receive the escaped line"
    joined = "\n".join(seen)
    assert "&lt;b&gt;" in joined
    assert "&amp;" in joined
    assert "<b>" not in joined
    assert "& <i>" not in joined


def test_missing_reportlab_raises_runtime_error_with_install_hint(
    tmp_path, monkeypatch
):
    src = tmp_path / "notes.txt"
    src.write_text("hello\n", encoding="utf-8")

    for key in list(sys.modules):
        if key == "reportlab" or key.startswith("reportlab."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "reportlab" or name.startswith("reportlab."):
            raise ImportError("simulated missing reportlab")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="reportlab is required") as excinfo:
        Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

    assert "pip install reportlab" in str(excinfo.value)
