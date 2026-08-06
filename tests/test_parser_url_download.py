"""Tests for the Parser URL detection and download helpers."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_parser_class():
    """Load Parser without importing the heavy raganything package."""
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location("_raganything_parser", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser


Parser = _load_parser_class()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://example.com/file.pdf", True),
        ("http://example.com/path?id=1", True),
        ("/local/path/file.pdf", False),
        ("file.pdf", False),
        ("", False),
        ("ftp://example.com/x", True),
    ],
)
def test_is_url(value, expected):
    assert Parser._is_url(value) is expected


def _fake_response(*, body: bytes = b"%PDF-1.4 fake", content_type: str = ""):
    response = MagicMock()
    response.headers.get.side_effect = lambda key, default="": (
        content_type if key == "Content-Type" else default
    )
    buf = io.BytesIO(body)

    def _read(n=-1):
        return buf.read(n)

    response.read = _read
    response.close = MagicMock()
    return response


def _patch_opener(response):
    opener = MagicMock()
    opener.open.return_value = response
    return patch("urllib.request.build_opener", return_value=opener), opener


def test_download_file_uses_extension_from_url_path(tmp_path):
    parser = Parser()
    response = _fake_response()
    opener_patch, opener = _patch_opener(response)

    with (
        opener_patch,
        patch.object(Parser, "_assert_public_hostname"),
    ):
        downloaded = parser._download_file("https://example.com/docs/report.pdf")

    try:
        assert downloaded.suffix == ".pdf"
        assert downloaded.exists()
        assert downloaded.read_bytes() == b"%PDF-1.4 fake"
        opener.open.assert_called_once()
        _, kwargs = opener.open.call_args
        assert kwargs.get("timeout") == 30, "must pass an explicit timeout"
    finally:
        if downloaded.exists():
            downloaded.unlink()


def test_download_file_infers_extension_from_content_type(tmp_path):
    parser = Parser()
    response = _fake_response(content_type="application/pdf; charset=utf-8")
    opener_patch, _ = _patch_opener(response)

    with (
        opener_patch,
        patch.object(Parser, "_assert_public_hostname"),
    ):
        downloaded = parser._download_file("https://example.com/download?id=123")

    try:
        assert downloaded.suffix == ".pdf"
        assert downloaded.exists()
    finally:
        if downloaded.exists():
            downloaded.unlink()


def test_download_file_cleans_up_temp_on_failure():
    parser = Parser()
    leaked: list[Path] = []

    real_mkstemp = __import__("tempfile").mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        leaked.append(Path(name))
        return fd, name

    response = MagicMock()
    response.headers.get.return_value = ""
    response.read.side_effect = OSError("connection reset")
    response.close = MagicMock()
    opener_patch, _ = _patch_opener(response)

    with (
        opener_patch,
        patch.object(Parser, "_assert_public_hostname"),
        patch("tempfile.mkstemp", side_effect=tracking_mkstemp),
    ):
        with pytest.raises(RuntimeError, match="Failed to download"):
            parser._download_file("https://example.com/file.pdf")

    assert leaked, "temp file should have been created"
    for p in leaked:
        assert not p.exists(), (
            f"temp file {p} leaked after failed download — exception path "
            "must clean it up"
        )
    response.close.assert_called_once()


def test_download_file_cleans_up_temp_on_urlopen_failure():
    """When opener.open itself fails, no temp file should be created or leaked."""
    parser = Parser()
    created: list[Path] = []

    real_mkstemp = __import__("tempfile").mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name))
        return fd, name

    opener = MagicMock()
    opener.open.side_effect = TimeoutError("stalled")

    with (
        patch("urllib.request.build_opener", return_value=opener),
        patch.object(Parser, "_assert_public_hostname"),
        patch("tempfile.mkstemp", side_effect=tracking_mkstemp),
    ):
        with pytest.raises(RuntimeError, match="Failed to download"):
            parser._download_file("https://slow.example.com/file.pdf")

    for p in created:
        assert not p.exists(), f"temp file {p} leaked"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x.pdf",
        "file://localhost/etc/passwd",
        "gopher://evil/x",
        "http://127.0.0.1/secret.pdf",
        "http://localhost/secret.pdf",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal.pdf",
    ],
)
def test_download_file_blocks_ssrf_targets(url):
    parser = Parser()
    with pytest.raises(RuntimeError, match="Failed to download"):
        parser._download_file(url)


def test_download_file_rejects_oversized_content_length():
    parser = Parser()
    response = MagicMock()
    response.headers.get.side_effect = lambda key, default="": (
        str(Parser.MAX_DOWNLOAD_BYTES + 1) if key == "Content-Length" else default
    )
    response.read = MagicMock(return_value=b"")
    response.close = MagicMock()
    opener_patch, _ = _patch_opener(response)

    with (
        opener_patch,
        patch.object(Parser, "_assert_public_hostname"),
    ):
        with pytest.raises(RuntimeError, match="Content-Length|max download size"):
            parser._download_file("https://example.com/huge.pdf")


def test_download_file_enforces_stream_size_cap():
    parser = Parser()
    # Stream more than the cap in chunks
    oversized = b"x" * (64 * 1024)
    chunks = [oversized] * ((Parser.MAX_DOWNLOAD_BYTES // len(oversized)) + 2)
    response = MagicMock()
    response.headers.get.return_value = ""
    response.read = MagicMock(side_effect=chunks + [b""])
    response.close = MagicMock()
    opener_patch, _ = _patch_opener(response)

    with (
        opener_patch,
        patch.object(Parser, "_assert_public_hostname"),
        patch.object(Parser, "MAX_DOWNLOAD_BYTES", 100_000),
    ):
        with pytest.raises(RuntimeError, match="exceeded max size"):
            parser._download_file("https://example.com/stream.pdf")
