"""Regression tests for EnhancedMarkdownConverter.convert_file_to_pdf.

Markdown-to-PDF ingest must fail closed on missing files and decode GBK /
latin-1 plant manuals as the intended encoding rather than silent mojibake.
"""

from unittest.mock import patch

import pytest

from raganything.enhanced_markdown import EnhancedMarkdownConverter


def _converter():
    return EnhancedMarkdownConverter()


class TestConvertFileToPdfValidation:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "gone.md"
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            _converter().convert_file_to_pdf(str(missing))


class TestConvertFileToPdfEncoding:
    def test_utf8_content_and_default_output_path(self, tmp_path):
        src = tmp_path / "spec.md"
        src.write_text("# Title\n\nUTF-8 café\n", encoding="utf-8")

        with patch.object(
            EnhancedMarkdownConverter,
            "convert_markdown_to_pdf",
            return_value=True,
        ) as convert:
            ok = _converter().convert_file_to_pdf(str(src))

        assert ok is True
        convert.assert_called_once()
        content, output_path, method = convert.call_args.args
        assert "UTF-8 café" in content
        assert output_path == str(src.with_suffix(".pdf"))
        assert method == "auto"

    def test_gbk_fallback_is_not_latin1_mojibake(self, tmp_path):
        src = tmp_path / "manual.md"
        chinese = "设备说明书"
        gbk_bytes = chinese.encode("gbk")
        src.write_bytes(gbk_bytes)

        with patch.object(
            EnhancedMarkdownConverter,
            "convert_markdown_to_pdf",
            return_value=True,
        ) as convert:
            ok = _converter().convert_file_to_pdf(
                str(src), output_path=str(tmp_path / "out.pdf")
            )

        assert ok is True
        content = convert.call_args.args[0]
        assert content == chinese
        assert gbk_bytes.decode("latin-1") != chinese

    def test_latin1_fallback_when_not_utf8_or_gbk(self, tmp_path):
        src = tmp_path / "notes.md"
        src.write_bytes(b"caf\xe9 notes\n")

        with patch.object(
            EnhancedMarkdownConverter,
            "convert_markdown_to_pdf",
            return_value=True,
        ) as convert:
            ok = _converter().convert_file_to_pdf(str(src), method="weasyprint")

        assert ok is True
        content, _output_path, method = convert.call_args.args
        assert content == "café notes\n"
        assert method == "weasyprint"

    def test_custom_output_path_forwarded(self, tmp_path):
        src = tmp_path / "doc.md"
        src.write_text("hello", encoding="utf-8")
        dest = tmp_path / "nested" / "doc.pdf"

        with patch.object(
            EnhancedMarkdownConverter,
            "convert_markdown_to_pdf",
            return_value=False,
        ) as convert:
            ok = _converter().convert_file_to_pdf(
                str(src), output_path=str(dest), method="pandoc"
            )

        assert ok is False
        assert convert.call_args.args[1] == str(dest)
        assert convert.call_args.args[2] == "pandoc"
