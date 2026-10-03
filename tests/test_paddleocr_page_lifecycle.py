"""PaddleOCR page lifecycle and recognition-result contracts.

Scanned manuals depend on three behaviors that the routing tests do not lock:
initializer fallback when PaddleOCR rejects ``show_log``, harvesting a single
``(text, confidence)`` pair, and releasing the PDF plus temp image if OCR fails
mid-document.
"""

import sys
import tempfile
from pathlib import Path

import pytest


def _load_parser_module():
    """Load parser.py without importing the LightRAG-backed package."""
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_paddleocr_lifecycle", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()
PaddleOCRParser = parser_module.PaddleOCRParser


class _Rendered:
    def __init__(self, value):
        self.value = value

    def to_pil(self):
        return self.value


class _NumpyRendered:
    def __init__(self, value):
        self.value = value

    def to_numpy(self):
        return self.value


class _BothRendered:
    def to_pil(self):
        return "pil-image"

    def to_numpy(self):
        return "ndarray"


class _Page:
    def __init__(self, label, events, rendered):
        self.label = label
        self.events = events
        self.rendered = rendered

    def render(self, scale):
        self.events.append(("render", self.label, scale))
        return self.rendered

    def close(self):
        self.events.append(("page-close", self.label))


class _Pdf:
    def __init__(self, pages, events):
        self._pages = pages
        self.events = events

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        self.events.append("pdf-close")


def _install_pdfium(monkeypatch, pdf):
    class FakePdfium:
        def PdfDocument(self, path):
            self.path = path
            return pdf

    fake = FakePdfium()
    monkeypatch.setitem(sys.modules, "pypdfium2", fake)
    return fake


def test_get_ocr_skips_rejected_show_log_and_reuses_language():
    parser = PaddleOCRParser(default_lang="en")
    calls = []

    class FakePaddle:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "show_log" in kwargs:
                raise TypeError("unexpected keyword show_log")

    parser._require_paddleocr = lambda: FakePaddle

    first = parser._get_ocr(None)
    blank = parser._get_ocr("   ")
    stripped = parser._get_ocr("  en  ")
    chinese = parser._get_ocr("zh")

    assert first is blank is stripped
    assert chinese is not first
    assert calls == [
        {"lang": "en", "show_log": False},
        {"lang": "en"},
        {"lang": "zh", "show_log": False},
        {"lang": "zh"},
    ]


def test_get_ocr_raises_when_every_initializer_fails():
    parser = PaddleOCRParser()

    class Boom:
        def __init__(self, **kwargs):
            raise ValueError("no paddle")

    parser._require_paddleocr = lambda: Boom

    with pytest.raises(RuntimeError, match="language 'ch'.*no paddle"):
        parser._get_ocr("ch")


def test_single_confidence_pair_and_null_neighbors_are_harvested():
    parser = PaddleOCRParser()

    assert parser._extract_text_lines(("Torque 12 Nm", 0.93)) == ["Torque 12 Nm"]
    assert parser._extract_text_lines(("limit", 1)) == ["limit"]
    assert parser._extract_text_lines(("   ", 0.4)) == []
    assert parser._extract_text_lines([("only line", 0.4)]) == ["only line"]
    assert parser._extract_text_lines(["  alpha  ", None, "beta"]) == ["alpha", "beta"]
    assert parser._extract_text_lines({"rec_texts": [{"text": "nested"}]}) == ["nested"]


def test_to_dict_failure_does_not_abort_harvest():
    parser = PaddleOCRParser()

    class BadAdapter:
        def to_dict(self):
            raise ValueError("bad adapter")

    assert parser._extract_text_lines(BadAdapter()) == []
    assert parser._extract_text_lines(None) == []


def test_ocr_input_prefers_ocr_and_forwards_cls_flag():
    parser = PaddleOCRParser()
    seen = {}

    class FakeOCR:
        def ocr(self, data, cls=True):
            seen["data"] = data
            seen["cls"] = cls
            return ("line", 0.9)

        def predict(self, data):
            raise AssertionError("predict must not run when ocr exists")

    parser._get_ocr = lambda lang=None: FakeOCR()

    assert parser._ocr_input("img", lang="en", cls_enabled=False) == ["line"]
    assert seen == {"data": "img", "cls": False}


def test_rendered_page_deletes_temp_image_when_ocr_fails():
    parser = PaddleOCRParser()
    captured = {}

    class PilPage:
        def save(self, path):
            Path(path).write_bytes(b"png")
            captured["path"] = Path(path)

    def fail_ocr(data, lang=None, cls_enabled=True):
        assert Path(data).is_file()
        raise RuntimeError("ocr down")

    parser._ocr_input = fail_ocr

    with pytest.raises(RuntimeError, match="ocr down"):
        parser._ocr_rendered_page(PilPage(), lang="ch", cls_enabled=False)

    assert not captured["path"].exists()


def test_rendered_page_returns_text_when_temp_unlink_fails(monkeypatch):
    parser = PaddleOCRParser()
    captured = {}

    class PilPage:
        def save(self, path):
            Path(path).write_bytes(b"png")
            captured["path"] = Path(path)

    parser._ocr_input = lambda data, lang=None, cls_enabled=True: ["kept"]
    real_unlink = Path.unlink

    def boom(self, *args, **kwargs):
        if self.suffix == ".png":
            raise OSError("busy")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", boom)
    try:
        assert parser._ocr_rendered_page(PilPage()) == ["kept"]
        assert captured["path"].exists()
    finally:
        if captured.get("path") and captured["path"].exists():
            real_unlink(captured["path"])


def test_non_pil_page_is_passed_through_without_a_temp_file(monkeypatch):
    parser = PaddleOCRParser()
    page = object()
    seen = {}

    def fake_ocr(data, lang=None, cls_enabled=True):
        seen["data"] = data
        seen["lang"] = lang
        seen["cls"] = cls_enabled
        return ["array-line"]

    parser._ocr_input = fake_ocr
    created = []
    real_named = tempfile.NamedTemporaryFile

    def spy_named(*args, **kwargs):
        created.append(kwargs.get("suffix"))
        return real_named(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", spy_named)

    assert parser._ocr_rendered_page(page, lang="ch", cls_enabled=False) == [
        "array-line"
    ]
    assert seen == {"data": page, "lang": "ch", "cls": False}
    assert created == []


def test_parse_pdf_collects_pil_pages_and_closes_document(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    events = []
    pages = [
        _Page("p0", events, _Rendered("alpha")),
        _Page("p1", events, _Rendered("beta")),
    ]
    pdf = _Pdf(pages, events)
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    fake = _install_pdfium(monkeypatch, pdf)

    def ocr(rendered, lang=None, cls_enabled=True):
        assert lang == "ch"
        assert cls_enabled is True
        return [rendered]

    monkeypatch.setattr(parser, "_ocr_rendered_page", ocr)

    content = parser.parse_pdf(pdf_file, lang="ch")

    assert content == [
        {"type": "text", "text": "alpha", "page_idx": 0},
        {"type": "text", "text": "beta", "page_idx": 1},
    ]
    assert fake.path == str(pdf_file)
    assert ("render", "p0", 2.0) in events
    assert ("render", "p1", 2.0) in events
    assert ("page-close", "p0") in events
    assert ("page-close", "p1") in events
    assert events[-1] == "pdf-close"


def test_parse_pdf_closes_document_when_later_page_ocr_fails(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    events = []
    pages = [
        _Page("p0", events, _Rendered("p0")),
        _Page("p1", events, _Rendered("p1")),
    ]
    pdf = _Pdf(pages, events)
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    _install_pdfium(monkeypatch, pdf)
    calls = []

    def ocr(rendered, lang=None, cls_enabled=True):
        calls.append((rendered, lang, cls_enabled))
        if rendered == "p1":
            raise RuntimeError("ocr failed")
        return [f"{rendered}-line"]

    monkeypatch.setattr(parser, "_ocr_rendered_page", ocr)

    with pytest.raises(RuntimeError, match="ocr failed"):
        parser.parse_pdf(pdf_file, lang="ch", cls=False)

    assert calls == [("p0", "ch", False), ("p1", "ch", False)]
    assert ("page-close", "p0") in events
    assert ("page-close", "p1") in events
    assert events[-1] == "pdf-close"


def test_page_renderer_prefers_pil_and_falls_back_to_numpy(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    pdf_file = tmp_path / "scan.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    events = []
    both = _Pdf([_Page("p0", events, _BothRendered())], events)
    _install_pdfium(monkeypatch, both)
    assert list(parser._extract_pdf_page_inputs(pdf_file)) == [(0, "pil-image")]
    assert events[-1] == "pdf-close"

    events = []
    numpy_pdf = _Pdf([_Page("p0", events, _NumpyRendered("ndarray"))], events)
    _install_pdfium(monkeypatch, numpy_pdf)
    assert list(parser._extract_pdf_page_inputs(pdf_file)) == [(0, "ndarray")]
    assert ("render", "p0", 2.0) in events
    assert events[-1] == "pdf-close"


def test_unsupported_page_render_still_closes_document(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    events = []
    pdf = _Pdf([_Page("p0", events, object())], events)
    pdf_file = tmp_path / "scan.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    _install_pdfium(monkeypatch, pdf)

    with pytest.raises(RuntimeError, match="Unsupported rendered page format"):
        list(parser._extract_pdf_page_inputs(pdf_file))

    assert ("page-close", "p0") in events
    assert events[-1] == "pdf-close"


def test_parse_image_rejects_missing_and_non_image_files(tmp_path):
    parser = PaddleOCRParser()

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        parser.parse_image(tmp_path / "gone.png")

    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported image format"):
        parser.parse_image(notes)


def test_parse_pdf_missing_file_does_not_load_renderer(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "pypdfium2", raising=False)
    parser = PaddleOCRParser()

    with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
        parser.parse_pdf(tmp_path / "gone.pdf")

    assert "pypdfium2" not in sys.modules
