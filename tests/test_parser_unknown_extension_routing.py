"""Unknown-extension routing for MinerU, Docling, and PaddleOCR.

MinerU warns and parses ``.csv``, ``.html``, and extensionless files as PDFs,
forwarding method, language, output directory, and extra kwargs. Docling and
PaddleOCR reject those same files. Docling still parses ``.html`` as HTML.
A missing file raises before any of those routes run.

Uppercase known suffixes are covered elsewhere. This file locks the fallback
and the two fail-closed parsers, which those tests never enter.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest


def _load_parser_module():
    """Load parser.py without importing the raganything package."""
    module_name = "_raganything_parser_unknown_ext_1122"
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()


def _touch(path: Path) -> Path:
    path.write_bytes(b"placeholder")
    return path


def test_mineru_unknown_extensions_parse_as_pdf(tmp_path, monkeypatch, caplog):
    parser = parser_module.MineruParser()
    calls = []

    def fake_pdf(file_path, output_dir, method, lang, **kwargs):
        calls.append(
            ("pdf", Path(file_path), output_dir, method, lang, kwargs.get("backend"))
        )
        return [{"type": "text", "text": "pdf-fallback"}]

    def unused(*args, **kwargs):
        calls.append(("unused", args, kwargs))
        return []

    monkeypatch.setattr(parser, "parse_pdf", fake_pdf)
    monkeypatch.setattr(parser, "parse_image", unused)
    monkeypatch.setattr(parser, "parse_office_doc", unused)
    monkeypatch.setattr(parser, "parse_text_file", unused)

    csv_file = _touch(tmp_path / "readings.csv")
    html_file = _touch(tmp_path / "page.html")
    bare_file = _touch(tmp_path / "README")

    with caplog.at_level(logging.WARNING, logger=parser_module.__name__):
        assert parser.parse_document(
            csv_file,
            method="ocr",
            output_dir="out/parsed",
            lang="ch",
            backend="pipeline",
        ) == [{"type": "text", "text": "pdf-fallback"}]
        assert parser.parse_document(
            html_file, method="txt", output_dir="out/html", lang="en"
        ) == [{"type": "text", "text": "pdf-fallback"}]
        assert parser.parse_document(bare_file, method="auto", lang="en") == [
            {"type": "text", "text": "pdf-fallback"}
        ]

    assert calls == [
        ("pdf", csv_file, "out/parsed", "ocr", "ch", "pipeline"),
        ("pdf", html_file, "out/html", "txt", "en", None),
        ("pdf", bare_file, None, "auto", "en", None),
    ]
    warnings = [
        message
        for message in caplog.messages
        if "Unsupported file extension" in message
    ]
    assert warnings == [
        "Warning: Unsupported file extension '.csv', attempting to parse as PDF",
        "Warning: Unsupported file extension '.html', attempting to parse as PDF",
        "Warning: Unsupported file extension '', attempting to parse as PDF",
    ]


def test_docling_rejects_unknown_extensions_and_keeps_html(tmp_path, monkeypatch):
    parser = parser_module.DoclingParser()
    calls = []

    def fake_pdf(*args, **kwargs):
        calls.append("pdf")
        return []

    def fake_office(*args, **kwargs):
        calls.append("office")
        return []

    def fake_html(file_path, output_dir, lang, **kwargs):
        calls.append(("html", Path(file_path), output_dir, lang, kwargs.get("backend")))
        return [{"type": "text", "text": "html"}]

    monkeypatch.setattr(parser, "parse_pdf", fake_pdf)
    monkeypatch.setattr(parser, "parse_office_doc", fake_office)
    monkeypatch.setattr(parser, "parse_html", fake_html)

    csv_file = _touch(tmp_path / "readings.csv")
    bare_file = _touch(tmp_path / "README")
    html_file = _touch(tmp_path / "page.html")

    with pytest.raises(ValueError, match=r"Unsupported file format: \.csv"):
        parser.parse_document(csv_file, method="ocr", lang="ch", backend="pipeline")
    with pytest.raises(ValueError, match=r"Unsupported file format: \. "):
        parser.parse_document(bare_file, lang="en")

    assert parser.parse_document(
        html_file, output_dir="out/html", lang="ch", backend="pipeline"
    ) == [{"type": "text", "text": "html"}]
    assert calls == [("html", html_file, "out/html", "ch", "pipeline")]


def test_paddleocr_rejects_unknown_extensions(tmp_path, monkeypatch):
    parser = parser_module.PaddleOCRParser()
    calls = []

    def unused(*args, **kwargs):
        calls.append("called")
        return []

    monkeypatch.setattr(parser, "parse_pdf", unused)
    monkeypatch.setattr(parser, "parse_image", unused)
    monkeypatch.setattr(parser, "parse_office_doc", unused)
    monkeypatch.setattr(parser, "parse_text_file", unused)

    expected = {
        "readings.csv": ".csv",
        "page.html": ".html",
        "README": "",
    }
    for name, ext in expected.items():
        path = _touch(tmp_path / name)
        with pytest.raises(ValueError) as exc_info:
            parser.parse_document(path, lang="ch", cls=False)
        assert str(exc_info.value) == (
            f"Unsupported file format: {ext}. "
            "PaddleOCR parser supports PDF, image, office, and text formats."
        )

    assert calls == []


def test_missing_unknown_extension_is_not_parsed(tmp_path, monkeypatch):
    missing = tmp_path / "gone.csv"
    routed = []

    def unused(*args, **kwargs):
        routed.append(args)
        return []

    mineru = parser_module.MineruParser()
    docling = parser_module.DoclingParser()
    paddle = parser_module.PaddleOCRParser()
    for parser in (mineru, docling, paddle):
        monkeypatch.setattr(parser, "parse_pdf", unused)
        monkeypatch.setattr(parser, "parse_image", unused, raising=False)
        monkeypatch.setattr(parser, "parse_office_doc", unused)
        monkeypatch.setattr(parser, "parse_text_file", unused, raising=False)
        monkeypatch.setattr(parser, "parse_html", unused, raising=False)
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            parser.parse_document(missing, method="ocr", lang="ch")

    assert routed == []
