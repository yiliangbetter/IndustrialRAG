"""Regression tests for EnhancedMarkdownConverter backend routing.

Unknown methods and missing backends must fail closed (False) rather than
raising into ingest. File decode fallbacks must still reach convert_markdown_to_pdf.
"""

from pathlib import Path

import pytest

from raganything.enhanced_markdown import EnhancedMarkdownConverter, MarkdownConfig


class TestRecommendedBackend:
    def test_prefers_pandoc_system_then_weasyprint(self):
        conv = EnhancedMarkdownConverter(MarkdownConfig())
        conv.available_backends = {
            "weasyprint": True,
            "pandoc": True,
            "markdown": True,
            "pandoc_system": True,
        }
        assert conv._get_recommended_backend() == "pandoc"

        conv.available_backends["pandoc_system"] = False
        assert conv._get_recommended_backend() == "weasyprint"

        conv.available_backends["weasyprint"] = False
        assert conv._get_recommended_backend() == "none"


class TestConvertMarkdownToPdfRouting:
    def test_weasyprint_method_delegates(self, tmp_path):
        conv = EnhancedMarkdownConverter()
        called = {}

        def fake_weasy(content, output_path):
            called["content"] = content
            called["output_path"] = output_path
            return True

        conv.convert_with_weasyprint = fake_weasy
        out = str(tmp_path / "out.pdf")
        assert conv.convert_markdown_to_pdf("# Hi", out, method="weasyprint") is True
        assert called["content"] == "# Hi"
        assert called["output_path"] == out

    def test_unknown_method_returns_false(self, tmp_path):
        conv = EnhancedMarkdownConverter()
        assert (
            conv.convert_markdown_to_pdf(
                "# Hi", str(tmp_path / "out.pdf"), method="ghostscript"
            )
            is False
        )

    def test_auto_none_backend_returns_false(self, tmp_path):
        conv = EnhancedMarkdownConverter()
        conv.available_backends = {
            "weasyprint": False,
            "pandoc": False,
            "markdown": False,
            "pandoc_system": False,
        }
        assert (
            conv.convert_markdown_to_pdf(
                "# Hi", str(tmp_path / "out.pdf"), method="auto"
            )
            is False
        )


class TestConvertFileToPdf:
    def test_missing_input_raises(self, tmp_path):
        conv = EnhancedMarkdownConverter()
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            conv.convert_file_to_pdf(str(tmp_path / "gone.md"))

    def test_latin1_decode_fallback(self, tmp_path):
        src = tmp_path / "latin1.md"
        src.write_bytes(b"# Caf\xe9\n")
        conv = EnhancedMarkdownConverter()
        seen = {}

        def fake_convert(content, output_path, method="auto"):
            seen["content"] = content
            seen["output_path"] = output_path
            seen["method"] = method
            return True

        conv.convert_markdown_to_pdf = fake_convert
        assert conv.convert_file_to_pdf(str(src), method="weasyprint") is True
        assert "Caf" in seen["content"]
        assert seen["method"] == "weasyprint"
        assert Path(seen["output_path"]).name == "latin1.pdf"
