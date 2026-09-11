"""PaddleOCR temp-PNG cleanup must not hide a successful page OCR.

Rendered PDF pages are saved to a temp .png, OCR'd, then unlinked. On
Windows the file can stay locked (antivirus, indexer). If unlink's
PermissionError escapes, a page that OCR'd correctly is dropped from
the index. Distinct from tests that assert the file is deleted on the
happy path or after OCR failure.
"""

from pathlib import Path
from unittest.mock import patch

from raganything.parser import PaddleOCRParser


class _SaveablePage:
    def save(self, path):
        Path(path).write_bytes(b"png-bytes")


def test_unlink_permission_error_still_returns_ocr_lines(monkeypatch):
    parser = PaddleOCRParser()
    seen = {}

    def fake_ocr_input(input_data, lang=None, cls_enabled=True):
        seen["path"] = input_data
        seen["existed"] = Path(input_data).exists()
        return ["pump-label"]

    monkeypatch.setattr(parser, "_ocr_input", fake_ocr_input)

    def boom_unlink(self, *args, **kwargs):
        raise PermissionError("file in use")

    with patch.object(Path, "unlink", boom_unlink):
        lines = parser._ocr_rendered_page(_SaveablePage(), lang="ch", cls_enabled=False)

    assert lines == ["pump-label"]
    assert seen["existed"] is True
    assert Path(seen["path"]).suffix.lower() == ".png"
