"""PaddleOCR office/text paths must convert to PDF then OCR with kwargs.

A regression that skips conversion or drops lang/kwargs silently OCR's the
wrong file or forces default language on plant manuals.
"""

from raganything.parser import PaddleOCRParser


def test_parse_office_doc_converts_then_forwards_kwargs(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    converted = tmp_path / "manual.pdf"
    captured = {}

    monkeypatch.setattr(
        parser,
        "convert_office_to_pdf",
        lambda doc_path, output_dir: converted,
    )

    def fake_parse_pdf(*, pdf_path, output_dir=None, lang=None, **kwargs):
        captured["pdf_path"] = pdf_path
        captured["output_dir"] = output_dir
        captured["lang"] = lang
        captured["kwargs"] = kwargs
        return [{"type": "text", "text": "office-ocr", "page_idx": 0}]

    monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

    result = parser.parse_office_doc(
        tmp_path / "manual.docx",
        output_dir=str(tmp_path / "out"),
        lang="ch",
        cls=False,
        device="cpu",
    )

    assert result == [{"type": "text", "text": "office-ocr", "page_idx": 0}]
    assert captured["pdf_path"] == converted
    assert captured["output_dir"] == str(tmp_path / "out")
    assert captured["lang"] == "ch"
    assert captured["kwargs"]["cls"] is False
    assert captured["kwargs"]["device"] == "cpu"


def test_parse_text_file_converts_then_forwards_kwargs(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    converted = tmp_path / "notes.pdf"
    captured = {}

    monkeypatch.setattr(
        parser,
        "convert_text_to_pdf",
        lambda text_path, output_dir: converted,
    )

    def fake_parse_pdf(*, pdf_path, output_dir=None, lang=None, **kwargs):
        captured["pdf_path"] = pdf_path
        captured["output_dir"] = output_dir
        captured["lang"] = lang
        captured["kwargs"] = kwargs
        return [{"type": "text", "text": "text-ocr", "page_idx": 0}]

    monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

    result = parser.parse_text_file(
        tmp_path / "notes.md",
        output_dir=str(tmp_path / "out"),
        lang="en",
        page_idx=3,
    )

    assert result == [{"type": "text", "text": "text-ocr", "page_idx": 0}]
    assert captured["pdf_path"] == converted
    assert captured["output_dir"] == str(tmp_path / "out")
    assert captured["lang"] == "en"
    assert captured["kwargs"]["page_idx"] == 3


def test_parse_office_doc_does_not_call_parse_image(monkeypatch, tmp_path):
    parser = PaddleOCRParser()
    converted = tmp_path / "slides.pdf"
    monkeypatch.setattr(
        parser, "convert_office_to_pdf", lambda doc_path, output_dir: converted
    )
    monkeypatch.setattr(
        parser,
        "parse_pdf",
        lambda **kwargs: [{"type": "text", "text": "ok", "page_idx": 0}],
    )

    def boom(*args, **kwargs):
        raise AssertionError("office docs must not be sent to parse_image")

    monkeypatch.setattr(parser, "parse_image", boom)

    parser.parse_office_doc(tmp_path / "slides.pptx", output_dir=str(tmp_path))
