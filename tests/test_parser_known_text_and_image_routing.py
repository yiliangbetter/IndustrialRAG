"""Known text and image routing for MinerU, Docling, and PaddleOCR.

``.txt`` and ``.md`` are real manuals, and ``.png`` scans are real inputs.
MinerU must convert text through ``parse_text_file`` and send images through
``parse_image`` (which forces OCR itself). Docling has no text or image
parser, so those suffixes must fail before PDF, Office, or HTML parsing.
PaddleOCR must use its text and image parsers rather than the PDF path.

Unknown extensions such as ``.csv`` and extensionless names are covered
elsewhere. Uppercase suffixes are covered elsewhere.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_parser_module():
    """Load parser.py without importing the raganything package."""
    module_name = "_raganything_parser_known_text_image_1126"
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


def test_mineru_routes_text_and_png_away_from_pdf(tmp_path, monkeypatch):
    parser = parser_module.MineruParser()
    calls = []

    def fake_text(text_path, output_dir=None, lang=None, method="auto", **kwargs):
        calls.append(
            (
                "text",
                Path(text_path),
                output_dir,
                lang,
                method,
                kwargs.get("backend"),
            )
        )
        return [{"type": "text", "text": "from-text"}]

    def fake_image(image_path, output_dir=None, lang=None, **kwargs):
        calls.append(
            (
                "image",
                Path(image_path),
                output_dir,
                lang,
                dict(kwargs),
            )
        )
        return [{"type": "text", "text": "from-image"}]

    def unused(*args, **kwargs):
        calls.append(("unused", args, kwargs))
        return []

    monkeypatch.setattr(parser, "parse_text_file", fake_text)
    monkeypatch.setattr(parser, "parse_image", fake_image)
    monkeypatch.setattr(parser, "parse_pdf", unused)
    monkeypatch.setattr(parser, "parse_office_doc", unused)

    notes = _touch(tmp_path / "notes.txt")
    manual = _touch(tmp_path / "manual.md")
    scan = _touch(tmp_path / "scan.png")

    assert parser.parse_document(
        notes,
        method="ocr",
        output_dir="out/txt",
        lang="ch",
        backend="pipeline",
    ) == [{"type": "text", "text": "from-text"}]
    assert parser.parse_document(
        manual,
        method="txt",
        output_dir="out/md",
        lang="en",
        backend="pipeline",
    ) == [{"type": "text", "text": "from-text"}]
    assert parser.parse_document(
        scan,
        method="ocr",
        output_dir="out/img",
        lang="ch",
        backend="pipeline",
    ) == [{"type": "text", "text": "from-image"}]

    assert calls == [
        ("text", notes, "out/txt", "ch", "ocr", "pipeline"),
        ("text", manual, "out/md", "en", "txt", "pipeline"),
        ("image", scan, "out/img", "ch", {"backend": "pipeline"}),
    ]


def test_docling_rejects_text_and_png_before_any_parser(tmp_path, monkeypatch):
    parser = parser_module.DoclingParser()
    calls = []

    def unused(*args, **kwargs):
        calls.append("called")
        return []

    monkeypatch.setattr(parser, "parse_pdf", unused)
    monkeypatch.setattr(parser, "parse_office_doc", unused)
    monkeypatch.setattr(parser, "parse_html", unused)
    monkeypatch.setattr(parser, "convert_text_to_pdf", unused)
    monkeypatch.setattr(parser, "convert_office_to_pdf", unused)

    files = {
        ".png": _touch(tmp_path / "scan.png"),
        ".txt": _touch(tmp_path / "notes.txt"),
        ".md": _touch(tmp_path / "manual.md"),
    }
    for ext, path in files.items():
        with pytest.raises(ValueError, match=rf"Unsupported file format: \{ext}"):
            parser.parse_document(
                path,
                method="ocr",
                output_dir="out/rejected",
                lang="ch",
                backend="pipeline",
            )

    assert calls == []


def test_paddleocr_routes_text_and_png_to_typed_parsers(tmp_path, monkeypatch):
    parser = parser_module.PaddleOCRParser()
    calls = []

    def fake_text(text_path, output_dir=None, lang=None, **kwargs):
        calls.append(("text", Path(text_path), output_dir, lang, dict(kwargs)))
        return [{"type": "text", "text": "ocr-text", "page_idx": 0}]

    def fake_image(image_path, output_dir=None, lang=None, **kwargs):
        calls.append(("image", Path(image_path), output_dir, lang, dict(kwargs)))
        return [{"type": "text", "text": "ocr-image", "page_idx": 0}]

    def unused(*args, **kwargs):
        calls.append(("unused", args, kwargs))
        return []

    monkeypatch.setattr(parser, "parse_text_file", fake_text)
    monkeypatch.setattr(parser, "parse_image", fake_image)
    monkeypatch.setattr(parser, "parse_pdf", unused)
    monkeypatch.setattr(parser, "parse_office_doc", unused)

    notes = _touch(tmp_path / "notes.txt")
    manual = _touch(tmp_path / "manual.md")
    scan = _touch(tmp_path / "scan.png")

    assert parser.parse_document(
        notes, method="ocr", output_dir="out/txt", lang="ch", cls=False
    ) == [{"type": "text", "text": "ocr-text", "page_idx": 0}]
    assert parser.parse_document(
        manual, output_dir="out/md", lang="en", page_idx=2
    ) == [{"type": "text", "text": "ocr-text", "page_idx": 0}]
    assert parser.parse_document(scan, output_dir="out/img", lang="ch", cls=True) == [
        {"type": "text", "text": "ocr-image", "page_idx": 0}
    ]

    assert calls == [
        ("text", notes, "out/txt", "ch", {"cls": False}),
        ("text", manual, "out/md", "en", {"page_idx": 2}),
        ("image", scan, "out/img", "ch", {"cls": True}),
    ]
