"""PaddleOCR PDF page OCR must use a temp PNG for PIL pages and clean it up.

`parse_pdf` mocks `_ocr_rendered_page` on main. If saveable pages skipped the temp
file, numpy/array pages were forced through `.save`, or the PNG leaked after an
OCR error, plant-manual PDF ingest would miss pages or fill /tmp.
"""

from pathlib import Path

import pytest

from raganything.parser import PaddleOCRParser


class _SaveablePage:
    def save(self, path):
        Path(path).write_bytes(b"png-bytes")


def test_saveable_page_ocrs_temp_png_then_unlinks(monkeypatch):
    parser = PaddleOCRParser()
    seen = {}

    def fake_ocr_input(input_data, lang=None, cls_enabled=True):
        seen["path"] = input_data
        seen["lang"] = lang
        seen["cls"] = cls_enabled
        path = Path(input_data)
        seen["exists_during_ocr"] = path.exists()
        seen["suffix"] = path.suffix
        return ["line-a"]

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr_input)

    lines = parser._ocr_rendered_page(_SaveablePage(), lang="ch", cls_enabled=False)

    assert lines == ["line-a"]
    assert seen["lang"] == "ch"
    assert seen["cls"] is False
    assert seen["exists_during_ocr"] is True
    assert seen["suffix"].lower() == ".png"
    assert not Path(seen["path"]).exists()


def test_temp_png_unlinked_when_ocr_raises(monkeypatch):
    parser = PaddleOCRParser()
    seen = {}

    def fake_ocr_input(input_data, lang=None, cls_enabled=True):
        seen["path"] = input_data
        raise RuntimeError("ocr boom")

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr_input)

    with pytest.raises(RuntimeError, match="ocr boom"):
        parser._ocr_rendered_page(_SaveablePage())

    assert seen["path"]
    assert not Path(seen["path"]).exists()


def test_non_saveable_page_passed_through_to_ocr_input(monkeypatch):
    parser = PaddleOCRParser()
    rendered = object()
    seen = {}

    def fake_ocr_input(input_data, lang=None, cls_enabled=True):
        seen["data"] = input_data
        seen["lang"] = lang
        seen["cls"] = cls_enabled
        return ["from-array"]

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr_input)

    lines = parser._ocr_rendered_page(rendered, lang="en", cls_enabled=True)

    assert lines == ["from-array"]
    assert seen["data"] is rendered
    assert seen["lang"] == "en"
    assert seen["cls"] is True


def test_parse_pdf_forwards_lang_and_cls_into_rendered_ocr(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        parser,
        "_extract_pdf_page_inputs",
        lambda pdf_path: [(4, _SaveablePage())],
    )
    seen = {}

    def fake_ocr_input(input_data, lang=None, cls_enabled=True):
        seen["lang"] = lang
        seen["cls"] = cls_enabled
        return ["pump-label"]

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr_input)

    content_list = parser.parse_pdf(pdf_path, lang="ch", cls=False)

    assert content_list == [
        {"type": "text", "text": "pump-label", "page_idx": 4},
    ]
    assert seen["lang"] == "ch"
    assert seen["cls"] is False
