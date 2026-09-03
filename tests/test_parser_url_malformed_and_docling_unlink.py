"""Parser URL detection and Docling temp-file cleanup fail-open.

Malformed IPv6 URLs make urllib.parse.urlparse raise ValueError. _is_url
must return False instead of crashing routing. After a URL download,
parse_document's finally block must not mask a successful parse or a
parse error if unlink fails.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from raganything.parser import DoclingParser, Parser


class TestMalformedUrlDetection:
    def test_malformed_ipv6_is_not_a_url(self):
        assert Parser._is_url("http://[::1") is False
        assert Parser._is_url("http://[") is False

    def test_well_formed_ipv6_url_is_detected(self):
        assert Parser._is_url("http://[::1]/doc.pdf") is True


class TestDoclingUrlUnlinkFailure:
    @pytest.fixture
    def parser(self):
        return DoclingParser()

    def test_successful_parse_survives_temp_unlink_failure(
        self, parser, tmp_path, monkeypatch
    ):
        downloaded = tmp_path / "remote.pdf"
        downloaded.write_bytes(b"%PDF-1.4\n")

        monkeypatch.setattr(parser, "_is_url", lambda path: True)
        monkeypatch.setattr(parser, "_download_file", lambda url: downloaded)
        monkeypatch.setattr(
            parser,
            "parse_pdf",
            lambda *a, **k: [{"type": "text", "text": "url-pdf"}],
        )

        def boom_unlink(self, *args, **kwargs):
            raise PermissionError("unlink denied")

        with patch.object(Path, "unlink", boom_unlink):
            result = parser.parse_document("https://example.com/remote.pdf")

        assert result == [{"type": "text", "text": "url-pdf"}]
        assert downloaded.exists()

    def test_parse_error_not_masked_by_temp_unlink_failure(
        self, parser, tmp_path, monkeypatch
    ):
        downloaded = tmp_path / "broken.pdf"
        downloaded.write_bytes(b"%PDF-1.4\n")

        monkeypatch.setattr(parser, "_is_url", lambda path: True)
        monkeypatch.setattr(parser, "_download_file", lambda url: downloaded)

        def boom_parse(*a, **k):
            raise RuntimeError("parse failed")

        monkeypatch.setattr(parser, "parse_pdf", boom_parse)

        def boom_unlink(self, *args, **kwargs):
            raise PermissionError("unlink denied")

        with patch.object(Path, "unlink", boom_unlink):
            with pytest.raises(RuntimeError, match="parse failed"):
                parser.parse_document("https://example.com/broken.pdf")

        assert downloaded.exists()
