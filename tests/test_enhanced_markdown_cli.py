"""Regression tests for EnhancedMarkdownConverter CLI exit-code contracts.

Operators and CI scripts rely on `raganything.enhanced_markdown.main`
returning 0 for `--info` / successful conversion and 1 on conversion
failure. Missing-input must fail via argparse (SystemExit 2), not silent
success. Backend conversion bodies are covered elsewhere; this file only
locks the CLI surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raganything.enhanced_markdown import main


def test_cli_info_returns_zero_without_input(monkeypatch, capsys):
    fake = MagicMock()
    fake.get_backend_info.return_value = {
        "available_backends": {
            "weasyprint": True,
            "pandoc": False,
            "markdown": True,
            "pandoc_system": False,
        },
        "recommended_backend": "weasyprint",
    }
    monkeypatch.setattr(
        "raganything.enhanced_markdown.EnhancedMarkdownConverter",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr("sys.argv", ["enhanced_markdown", "--info"])

    assert main() == 0
    out = capsys.readouterr().out
    assert "Backend Information:" in out
    assert "weasyprint" in out
    assert "Recommended backend: weasyprint" in out
    fake.convert_file_to_pdf.assert_not_called()


def test_cli_missing_input_raises_parser_error(monkeypatch):
    monkeypatch.setattr(
        "raganything.enhanced_markdown.EnhancedMarkdownConverter",
        MagicMock(),
    )
    monkeypatch.setattr("sys.argv", ["enhanced_markdown"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_cli_returns_zero_on_successful_conversion(monkeypatch, capsys):
    fake = MagicMock()
    fake.convert_file_to_pdf.return_value = True
    monkeypatch.setattr(
        "raganything.enhanced_markdown.EnhancedMarkdownConverter",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "enhanced_markdown",
            "notes.md",
            "--output",
            "notes.pdf",
            "--method",
            "weasyprint",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "Successfully converted notes.md to PDF" in out
    fake.convert_file_to_pdf.assert_called_once_with(
        input_path="notes.md",
        output_path="notes.pdf",
        method="weasyprint",
    )


def test_cli_returns_one_when_conversion_returns_false(monkeypatch, capsys):
    fake = MagicMock()
    fake.convert_file_to_pdf.return_value = False
    monkeypatch.setattr(
        "raganything.enhanced_markdown.EnhancedMarkdownConverter",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr("sys.argv", ["enhanced_markdown", "notes.md"])

    assert main() == 1
    assert "Conversion failed" in capsys.readouterr().out


def test_cli_returns_one_when_conversion_raises(monkeypatch, capsys):
    fake = MagicMock()
    fake.convert_file_to_pdf.side_effect = FileNotFoundError("Input file not found")
    monkeypatch.setattr(
        "raganything.enhanced_markdown.EnhancedMarkdownConverter",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr("sys.argv", ["enhanced_markdown", "missing.md"])

    assert main() == 1
    assert "Input file not found" in capsys.readouterr().out


def test_cli_applies_custom_css_to_config(monkeypatch):
    fake = MagicMock()
    fake.convert_file_to_pdf.return_value = True
    converter_cls = MagicMock(return_value=fake)
    monkeypatch.setattr(
        "raganything.enhanced_markdown.EnhancedMarkdownConverter",
        converter_cls,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["enhanced_markdown", "notes.md", "--css", "custom.css"],
    )

    assert main() == 0
    config = converter_cls.call_args.args[0]
    assert config.css_file == "custom.css"
