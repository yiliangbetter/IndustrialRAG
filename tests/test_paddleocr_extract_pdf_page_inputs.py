"""Regression tests for PaddleOCR PDF page rendering adapters.

parse_pdf currently mocks this helper. The pypdfium2 to_pil / to_numpy
selection and close-on-error path decide whether OCR ingest drops pages or
leaks PDF handles.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from raganything.parser import PaddleOCRParser


class FakeRendered:
    def __init__(self, payload, *, pil=False, numpy=False):
        self.payload = payload
        if pil:
            self.to_pil = lambda: f"pil:{payload}"
        if numpy:
            self.to_numpy = lambda: f"numpy:{payload}"


class FakePage:
    def __init__(self, rendered, *, fail=False):
        self._rendered = rendered
        self.closed = False
        self.fail = fail
        self.render_scale = None

    def render(self, scale=2.0):
        self.render_scale = scale
        if self.fail:
            raise RuntimeError("render failed")
        return self._rendered

    def close(self):
        self.closed = True


class FakePdfDocument:
    def __init__(self, pages):
        self._pages = pages
        self.closed = False
        self.path = None

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        self.closed = True


def _install_fake_pdfium(monkeypatch, pages, captured_docs):
    fake_mod = types.ModuleType("pypdfium2")

    class PdfDocument:
        def __init__(self, path):
            doc = FakePdfDocument(pages)
            doc.path = path
            captured_docs.append(doc)
            self._doc = doc
            self._pages = pages
            self.closed = False
            self.path = path

        def __len__(self):
            return len(self._doc)

        def __getitem__(self, index):
            return self._doc[index]

        def close(self):
            self.closed = True
            self._doc.close()

    fake_mod.PdfDocument = PdfDocument
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_mod)
    return fake_mod


class TestExtractPdfPageInputs:
    def test_prefers_to_pil_and_closes_handles(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pages = [
            FakePage(FakeRendered("p0", pil=True, numpy=True)),
            FakePage(FakeRendered("p1", pil=True)),
        ]
        docs = []
        _install_fake_pdfium(monkeypatch, pages, docs)

        parser = PaddleOCRParser()
        results = list(parser._extract_pdf_page_inputs(pdf_path))

        assert results == [(0, "pil:p0"), (1, "pil:p1")]
        assert pages[0].render_scale == 2.0
        assert all(page.closed for page in pages)
        assert docs[0].closed is True

    def test_falls_back_to_numpy_when_no_pil(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pages = [FakePage(FakeRendered("raw", numpy=True))]
        docs = []
        _install_fake_pdfium(monkeypatch, pages, docs)

        parser = PaddleOCRParser()
        results = list(parser._extract_pdf_page_inputs(pdf_path))

        assert results == [(0, "numpy:raw")]
        assert pages[0].closed is True
        assert docs[0].closed is True

    def test_unsupported_render_format_still_closes(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pages = [FakePage(FakeRendered("raw"))]
        docs = []
        _install_fake_pdfium(monkeypatch, pages, docs)

        parser = PaddleOCRParser()
        with pytest.raises(RuntimeError, match="Unsupported rendered page format"):
            list(parser._extract_pdf_page_inputs(pdf_path))

        assert pages[0].closed is True
        assert docs[0].closed is True

    def test_render_error_still_closes_page_and_pdf(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pages = [FakePage(FakeRendered("raw", pil=True), fail=True)]
        docs = []
        _install_fake_pdfium(monkeypatch, pages, docs)

        parser = PaddleOCRParser()
        with pytest.raises(RuntimeError, match="render failed"):
            list(parser._extract_pdf_page_inputs(pdf_path))

        assert pages[0].closed is True
        assert docs[0].closed is True


class TestParsePdfFailClosed:
    def test_missing_pdf_raises(self, tmp_path):
        parser = PaddleOCRParser()
        missing = tmp_path / "gone.pdf"
        with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
            parser.parse_pdf(missing)

    def test_closes_generator_when_ocr_fails(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pages = [
            FakePage(FakeRendered("p0", pil=True)),
            FakePage(FakeRendered("p1", pil=True)),
        ]
        docs = []
        _install_fake_pdfium(monkeypatch, pages, docs)

        parser = PaddleOCRParser()
        monkeypatch.setattr(
            parser,
            "_ocr_rendered_page",
            MagicMock(side_effect=RuntimeError("ocr boom")),
        )

        with pytest.raises(RuntimeError, match="ocr boom"):
            parser.parse_pdf(pdf_path)

        assert docs[0].closed is True
        assert pages[0].closed is True
