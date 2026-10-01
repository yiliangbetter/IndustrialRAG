"""URL downloads must keep the path suffix when Content-Type disagrees.

The suffix chooses PDF, image, Office, or text parsing. A CDN that labels a
.pdf as text/html, or an unknown type with no extension, must not retarget
the file. Schemes without a host stay local paths.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_parser_class():
    """Load Parser without importing the raganything package."""
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_url_suffix_d8a7", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser


Parser = _load_parser_class()


def _fake_response(*, body: bytes = b"%PDF-1.4 fake", content_type: str = ""):
    response = MagicMock()
    response.headers.get.return_value = content_type
    response.read = io.BytesIO(body).read
    response.close = MagicMock()
    return response


@pytest.mark.parametrize(
    "value,expected",
    [
        ("file:///etc/passwd", False),
        ("http:///missing-host/file.pdf", False),
        ("https://example.com/file.pdf", True),
    ],
)
def test_is_url_requires_a_host(value, expected):
    assert Parser._is_url(value) is expected


def test_download_keeps_url_suffix_when_content_type_disagrees():
    parser = Parser()
    response = _fake_response(content_type="text/html; charset=utf-8")

    with patch("urllib.request.urlopen", return_value=response):
        downloaded = parser._download_file(
            "https://cdn.example.com/manuals/pump.pdf?token=abc"
        )

    try:
        assert downloaded.suffix == ".pdf"
        assert downloaded.read_bytes() == b"%PDF-1.4 fake"
        response.close.assert_called_once()
    finally:
        if downloaded.exists():
            downloaded.unlink()


def test_download_unknown_content_type_keeps_empty_suffix():
    parser = Parser()
    response = _fake_response(
        body=b"raw-bytes", content_type="application/x-industrial-scan"
    )

    with patch("urllib.request.urlopen", return_value=response):
        downloaded = parser._download_file("https://example.com/download")

    try:
        assert downloaded.suffix == ""
        assert downloaded.read_bytes() == b"raw-bytes"
        response.close.assert_called_once()
    finally:
        if downloaded.exists():
            downloaded.unlink()
