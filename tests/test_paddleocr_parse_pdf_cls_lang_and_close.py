"""PaddleOCR parse_pdf must forward OCR knobs and close page handles on success.

Main ``testpaddleocr_parser.py`` checks page_idx schema with a list of pages.
#141 covers missing PDFs and close-on-OCR-failure. #169 covers parse_image
``cls``/``lang``. This file locks the PDF path: ``cls``/``lang`` must reach
``_ocr_rendered_page``, and the page iterator must be closed after a successful
run so pypdfium2 handles are not leaked.
"""

from raganything.parser import PaddleOCRParser


class _CloseablePages:
    def __init__(self, items):
        self._items = list(items)
        self.closed = False

    def __iter__(self):
        return iter(self._items)

    def close(self):
        self.closed = True


def test_parse_pdf_forwards_lang_and_cls_and_closes_on_success(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    pages = _CloseablePages([(0, "page0"), (1, "page1")])
    seen = []

    monkeypatch.setattr(parser, "_extract_pdf_page_inputs", lambda pdf_path: pages)

    def fake_ocr(rendered_page, lang=None, cls_enabled=True):
        seen.append(
            {
                "page": rendered_page,
                "lang": lang,
                "cls_enabled": cls_enabled,
            }
        )
        return [f"{rendered_page}-text"]

    monkeypatch.setattr(parser, "_ocr_rendered_page", fake_ocr)

    content_list = parser.parse_pdf(pdf, lang="ch", cls=False)

    assert pages.closed is True
    assert seen == [
        {"page": "page0", "lang": "ch", "cls_enabled": False},
        {"page": "page1", "lang": "ch", "cls_enabled": False},
    ]
    assert content_list == [
        {"type": "text", "text": "page0-text", "page_idx": 0},
        {"type": "text", "text": "page1-text", "page_idx": 1},
    ]


def test_parse_pdf_defaults_cls_enabled_true(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    pages = _CloseablePages([(0, "page0")])
    seen = []

    monkeypatch.setattr(parser, "_extract_pdf_page_inputs", lambda pdf_path: pages)

    def fake_ocr(rendered_page, lang=None, cls_enabled=True):
        seen.append(cls_enabled)
        return ["ok"]

    monkeypatch.setattr(parser, "_ocr_rendered_page", fake_ocr)

    parser.parse_pdf(pdf, lang="en")

    assert seen == [True]
    assert pages.closed is True
