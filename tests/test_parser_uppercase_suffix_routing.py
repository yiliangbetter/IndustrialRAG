"""Case-insensitive extension routing for MinerU, Docling, and PaddleOCR.

Windows exports often use ``.PDF`` / ``.DOCX`` / ``.TXT``. Routing compares
``suffix.lower()``. Dropping that fold sends Office and text files down the
wrong parser (MinerU's unknown-extension PDF fallback, or a Docling/PaddleOCR
reject) while lowercase tests still pass.
"""

import importlib.util
import logging
from pathlib import Path


def _load_parser_module():
    """Load parser.py without importing the raganything package (needs LightRAG)."""
    module_name = "_raganything_parser_suffix_case"
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()


def _touch(path: Path) -> Path:
    path.write_bytes(b"placeholder")
    return path


def test_mineru_uppercase_suffixes_use_typed_routes_not_pdf_fallback(
    tmp_path, monkeypatch, caplog
):
    parser = parser_module.MineruParser()
    calls = []

    def fake_pdf(file_path, output_dir, method, lang, **kwargs):
        calls.append(("pdf", Path(file_path), method, lang))
        return [{"type": "text", "text": "pdf"}]

    def fake_image(file_path, output_dir, lang, **kwargs):
        calls.append(("image", Path(file_path), lang))
        return [{"type": "text", "text": "image"}]

    def fake_office(file_path, output_dir, lang, method="auto", **kwargs):
        calls.append(("office", Path(file_path), method, lang))
        return [{"type": "text", "text": "office"}]

    def fake_text(file_path, output_dir, lang, method="auto", **kwargs):
        calls.append(("text", Path(file_path), method, lang))
        return [{"type": "text", "text": "text"}]

    monkeypatch.setattr(parser, "parse_pdf", fake_pdf)
    monkeypatch.setattr(parser, "parse_image", fake_image)
    monkeypatch.setattr(parser, "parse_office_doc", fake_office)
    monkeypatch.setattr(parser, "parse_text_file", fake_text)

    pdf = _touch(tmp_path / "Manual.PDF")
    image = _touch(tmp_path / "Scan.JPEG")
    office = _touch(tmp_path / "Procedure.DOCX")
    text = _touch(tmp_path / "Notes.TXT")

    with caplog.at_level(logging.WARNING, logger=parser_module.__name__):
        assert parser.parse_document(pdf, method="ocr", lang="en")[0]["text"] == "pdf"
        assert parser.parse_document(image, lang="ch")[0]["text"] == "image"
        assert parser.parse_document(office, method="txt", lang="en")[0]["text"] == (
            "office"
        )
        assert parser.parse_document(text, method="ocr", lang="en")[0]["text"] == "text"

    assert calls == [
        ("pdf", pdf, "ocr", "en"),
        ("image", image, "ch"),
        ("office", office, "txt", "en"),
        ("text", text, "ocr", "en"),
    ]
    # ``.PDF`` must take the PDF branch. The unknown-extension fallback also
    # calls parse_pdf, but only after this warning.
    assert not any(
        "Unsupported file extension" in message for message in caplog.messages
    )


def test_docling_uppercase_suffixes_stay_supported(tmp_path, monkeypatch):
    parser = parser_module.DoclingParser()
    calls = []

    def fake_pdf(file_path, output_dir, method, lang, **kwargs):
        calls.append("pdf")
        return [{"type": "text", "text": "pdf"}]

    def fake_office(file_path, output_dir, lang, **kwargs):
        calls.append("office")
        return [{"type": "text", "text": "office"}]

    def fake_html(file_path, output_dir, lang, **kwargs):
        calls.append("html")
        return [{"type": "text", "text": "html"}]

    monkeypatch.setattr(parser, "parse_pdf", fake_pdf)
    monkeypatch.setattr(parser, "parse_office_doc", fake_office)
    monkeypatch.setattr(parser, "parse_html", fake_html)

    assert (
        parser.parse_document(_touch(tmp_path / "Spec.PDF"), method="auto")[0]["text"]
        == "pdf"
    )
    assert parser.parse_document(_touch(tmp_path / "Sheet.XLSX"))[0]["text"] == "office"
    assert parser.parse_document(_touch(tmp_path / "Page.HTML"))[0]["text"] == "html"
    assert calls == ["pdf", "office", "html"]


def test_paddleocr_uppercase_suffixes_do_not_fail_closed(tmp_path, monkeypatch):
    parser = parser_module.PaddleOCRParser()
    calls = []

    def fake_pdf(pdf_path, output_dir=None, method="auto", lang=None, **kwargs):
        calls.append(("pdf", lang, kwargs.get("cls")))
        return [{"type": "text", "text": "pdf"}]

    def fake_image(image_path, output_dir=None, lang=None, **kwargs):
        calls.append(("image", lang))
        return [{"type": "text", "text": "image"}]

    def fake_office(doc_path, output_dir=None, lang=None, **kwargs):
        calls.append(("office", lang))
        return [{"type": "text", "text": "office"}]

    def fake_text(text_path, output_dir=None, lang=None, **kwargs):
        calls.append(("text", lang))
        return [{"type": "text", "text": "text"}]

    monkeypatch.setattr(parser, "parse_pdf", fake_pdf)
    monkeypatch.setattr(parser, "parse_image", fake_image)
    monkeypatch.setattr(parser, "parse_office_doc", fake_office)
    monkeypatch.setattr(parser, "parse_text_file", fake_text)

    assert (
        parser.parse_document(_touch(tmp_path / "Scan.PDF"), lang="en", cls=False)[0][
            "text"
        ]
        == "pdf"
    )
    assert (
        parser.parse_document(_touch(tmp_path / "Photo.JPG"), lang="ch")[0]["text"]
        == "image"
    )
    assert (
        parser.parse_document(_touch(tmp_path / "Report.DOCX"), lang="en")[0]["text"]
        == "office"
    )
    assert (
        parser.parse_document(_touch(tmp_path / "Notes.MD"), lang="en")[0]["text"]
        == "text"
    )
    assert calls == [
        ("pdf", "en", False),
        ("image", "ch"),
        ("office", "en"),
        ("text", "en"),
    ]
