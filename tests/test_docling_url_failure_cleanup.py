"""Docling URL downloads must not leak, and local files must survive parse failures."""

from pathlib import Path

import pytest


def _load_parser_module():
    """Load parser.py without importing the raganything package."""
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "docling_url_failure_cleanup_parser", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _BusyDownload:
    """Path-like download whose unlink fails after the file is on disk."""

    def __init__(self, path: Path):
        self._path = path

    def __fspath__(self) -> str:
        return str(self._path)

    def exists(self) -> bool:
        return True

    def unlink(self) -> None:
        raise OSError("device busy")


def test_unsupported_url_download_is_removed(tmp_path):
    parser = _load_parser_module().DoclingParser()
    downloaded = tmp_path / "manual.csv"
    downloaded.write_text("a,b\n", encoding="utf-8")
    parsed = {"called": False}

    def fake_download(url):
        assert url == "https://example.com/manual.csv"
        return downloaded

    def fake_parse_pdf(*args, **kwargs):
        parsed["called"] = True
        return []

    parser._download_file = fake_download
    parser.parse_pdf = fake_parse_pdf

    with pytest.raises(ValueError, match="Unsupported file format"):
        parser.parse_document("https://example.com/manual.csv")

    assert parsed["called"] is False
    assert not downloaded.exists()


def test_parse_error_removes_url_download_and_keeps_the_error(tmp_path):
    parser = _load_parser_module().DoclingParser()
    downloaded = tmp_path / "manual.pdf"
    downloaded.write_bytes(b"%PDF-1.4")
    seen = {}

    def fake_download(url):
        assert url == "https://example.com/manual.pdf"
        return downloaded

    def fake_parse_pdf(pdf_path, output_dir=None, method="auto", lang=None, **kwargs):
        seen["pdf_path"] = Path(pdf_path)
        seen["existed_during_parse"] = Path(pdf_path).exists()
        raise RuntimeError("docling failed")

    parser._download_file = fake_download
    parser.parse_pdf = fake_parse_pdf

    with pytest.raises(RuntimeError, match="docling failed"):
        parser.parse_document(
            "https://example.com/manual.pdf",
            output_dir=str(tmp_path / "out"),
        )

    assert seen["pdf_path"] == downloaded
    assert seen["existed_during_parse"] is True
    assert not downloaded.exists()


def test_unlink_error_does_not_replace_parse_error(tmp_path):
    parser = _load_parser_module().DoclingParser()
    downloaded = tmp_path / "manual.csv"
    downloaded.write_text("a,b\n", encoding="utf-8")
    busy = _BusyDownload(downloaded)

    parser._download_file = lambda url: busy

    with pytest.raises(ValueError, match="Unsupported file format"):
        parser.parse_document("https://example.com/manual.csv")

    assert downloaded.exists()


def test_local_parse_failure_does_not_delete_source(tmp_path):
    parser = _load_parser_module().DoclingParser()
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"%PDF-1.4")

    def fake_parse_pdf(*args, **kwargs):
        raise RuntimeError("docling failed")

    parser.parse_pdf = fake_parse_pdf
    download_calls = {"count": 0}

    def fake_download(url):
        download_calls["count"] += 1
        return source

    parser._download_file = fake_download

    with pytest.raises(RuntimeError, match="docling failed"):
        parser.parse_document(str(source))

    assert download_calls["count"] == 0
    assert source.exists()
    assert source.read_bytes() == b"%PDF-1.4"
