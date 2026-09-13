"""Office/HTML parse_document must forward a real MinerU method.

``b0c3072`` taught ``parse_document`` to omit ``method=None`` so Office
converters do not see a null method. The counterpart contract is that a
concrete method (``ocr`` / ``txt`` / ``auto``) still reaches
``parse_office_doc``. Dropping that forwarding silently forces the parser
default and skips OCR on scanned Word/HTML manuals.

Distinct from ``tests/testparser_wiring.py`` / #139 (omit ``None`` only)
and from #119 (MinerU ``parse_office_doc`` → ``parse_pdf`` passthrough).
"""

from __future__ import annotations

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_processor(tmp_path, monkeypatch, captured_kwargs):
    class FakeParser:
        def parse_office_doc(self, **kwargs):
            captured_kwargs.update(kwargs)
            return [{"type": "text", "text": "office parsed", "page_idx": 0}]

    dummy = ProcessorMixin()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = None
    dummy.doc_parser = FakeParser()
    monkeypatch.setattr(
        ProcessorMixin,
        "_generate_content_based_doc_id",
        lambda self, content_list: "doc-office",
        raising=False,
    )
    return dummy


@pytest.mark.asyncio
async def test_parse_document_office_forwards_ocr_method(monkeypatch, tmp_path):
    captured = {}
    dummy = _make_processor(tmp_path, monkeypatch, captured)
    fake_docx = tmp_path / "scanned_manual.docx"
    fake_docx.write_bytes(b"PK\x03\x04")

    await dummy.parse_document(str(fake_docx), parse_method="ocr")

    assert captured["method"] == "ocr"
    assert captured["doc_path"] == fake_docx


@pytest.mark.asyncio
async def test_parse_document_html_forwards_method_and_strips_kwargs_method(
    monkeypatch, tmp_path
):
    captured = {}
    dummy = _make_processor(tmp_path, monkeypatch, captured)
    fake_html = tmp_path / "procedure.htm"
    fake_html.write_text("<html><body>bleed</body></html>", encoding="utf-8")

    await dummy.parse_document(
        str(fake_html),
        parse_method="txt",
        method="ocr",
        lang="ch",
    )

    assert captured["method"] == "txt"
    assert captured["lang"] == "ch"
