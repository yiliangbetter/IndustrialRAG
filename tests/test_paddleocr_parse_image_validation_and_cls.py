"""PaddleOCR parse_image fail-closed validation and OCR kwargs.

parse_document routing mocks parse_image. ProcessorMixin calls parse_image
directly for image extensions, so missing files, non-image suffixes, and
`cls` / `page_idx` forwarding must stay on the method itself.
"""

from pathlib import Path

import pytest

from raganything.parser import PaddleOCRParser


def test_missing_file_raises(tmp_path):
    parser = PaddleOCRParser()
    missing = tmp_path / "gone.png"
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        parser.parse_image(missing)


@pytest.mark.parametrize("name", ["notes.txt", "diagram.svg", "scan.pdf"])
def test_unsupported_suffix_raises(tmp_path, name):
    parser = PaddleOCRParser()
    path = tmp_path / name
    path.write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        parser.parse_image(path)


def test_uppercase_png_suffix_is_accepted(tmp_path, monkeypatch):
    parser = PaddleOCRParser()
    image = tmp_path / "PHOTO.PNG"
    image.write_bytes(b"png-bytes")
    seen = {}

    def fake_ocr(input_data, lang=None, cls_enabled=True):
        seen["input"] = input_data
        seen["lang"] = lang
        seen["cls"] = cls_enabled
        return ["line"]

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr)

    result = parser.parse_image(image)

    assert result == [{"type": "text", "text": "line", "page_idx": 0}]
    assert Path(seen["input"]) == image
    assert seen["cls"] is True


def test_forwards_cls_lang_and_page_idx(tmp_path, monkeypatch):
    parser = PaddleOCRParser()
    image = tmp_path / "plate.jpg"
    image.write_bytes(b"jpeg-bytes")
    seen = {}

    def fake_ocr(input_data, lang=None, cls_enabled=True):
        seen["input"] = input_data
        seen["lang"] = lang
        seen["cls"] = cls_enabled
        return ["SN-123", "SN-123"]

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr)

    result = parser.parse_image(image, lang="ch", cls=False, page_idx=4)

    assert result == [
        {"type": "text", "text": "SN-123", "page_idx": 4},
        {"type": "text", "text": "SN-123", "page_idx": 4},
    ]
    assert Path(seen["input"]) == image
    assert seen["lang"] == "ch"
    assert seen["cls"] is False
