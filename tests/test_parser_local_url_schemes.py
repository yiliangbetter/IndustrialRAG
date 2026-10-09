"""Docling must not treat local or scheme-relative paths as remote downloads.

``DoclingParser.parse_document`` downloads the input only when ``Parser._is_url``
is true. ``file:///…`` has no host, ``//cdn/…`` has no scheme, and ``C:/…`` is a
drive path whose scheme is empty-netloc. Fetching those would read the local
filesystem or skip a real manual. A normal https URL must still be downloaded
and the temp file removed after a successful parse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_local_url_4b43", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_parser = _load_parser_module()
Parser = _parser.Parser
DoclingParser = _parser.DoclingParser


@pytest.mark.parametrize(
    "value,expected",
    [
        ("file:///etc/passwd", False),
        ("file:///tmp/manual.pdf", False),
        ("//cdn.example.com/manual.pdf", False),
        ("C:/manuals/pump.pdf", False),
        ("https://example.com:8443/manual.pdf", True),
        ("https://user:placeholder@example.com/manual.pdf", True),
    ],
)
def test_is_url_rejects_local_schemes_and_accepts_remote_http(value, expected):
    assert Parser._is_url(value) is expected


def test_docling_does_not_download_local_or_scheme_relative_paths(monkeypatch):
    def unexpected_download(self, url):
        raise AssertionError(f"download attempted for {url}")

    monkeypatch.setattr(DoclingParser, "_download_file", unexpected_download)
    parser = DoclingParser()

    for value in (
        "file:///etc/passwd",
        "file:///tmp/manual.pdf",
        "//cdn.example.com/manual.pdf",
        "C:/manuals/pump.pdf",
    ):
        with pytest.raises(FileNotFoundError):
            parser.parse_document(value)


def test_docling_downloads_https_and_removes_temp_file(monkeypatch, tmp_path):
    pdf = tmp_path / "remote.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    seen: list[str] = []

    def fake_download(self, url):
        seen.append(url)
        return pdf

    monkeypatch.setattr(DoclingParser, "_download_file", fake_download)
    monkeypatch.setattr(
        DoclingParser,
        "parse_pdf",
        lambda self, *args, **kwargs: [{"type": "text", "text": "ok"}],
    )

    result = DoclingParser().parse_document("https://example.com/manual.pdf")

    assert seen == ["https://example.com/manual.pdf"]
    assert result == [{"type": "text", "text": "ok"}]
    assert not pdf.exists()
