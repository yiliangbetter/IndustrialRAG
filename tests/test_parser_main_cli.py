"""Regression tests for document parser CLI exit-code contracts.

`raganything.parser.main` is the operator-facing parse entrypoint. CI and
shell wrappers depend on nonzero exits for installation check failures and
parse errors, and on `--stats` reporting content-type counts after success.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raganything.parser import main


def test_cli_check_returns_zero_when_installed(monkeypatch, capsys):
    fake = MagicMock()
    fake.check_installation.return_value = True
    monkeypatch.setattr(
        "raganything.parser.get_parser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["parser", "unused.pdf", "--check", "--parser", "mineru"],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "Mineru is properly installed" in out
    fake.parse_document.assert_not_called()


def test_cli_check_returns_one_when_not_installed(monkeypatch, capsys):
    fake = MagicMock()
    fake.check_installation.return_value = False
    monkeypatch.setattr(
        "raganything.parser.get_parser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["parser", "unused.pdf", "--check", "--parser", "docling"],
    )

    assert main() == 1
    assert "Docling installation check failed" in capsys.readouterr().out


def test_cli_returns_zero_on_successful_parse(monkeypatch, capsys):
    fake = MagicMock()
    fake.parse_document.return_value = [
        {"type": "text", "text": "hello", "page_idx": 0},
        {"type": "image", "img_path": "a.png", "page_idx": 0},
    ]
    get_parser = MagicMock(return_value=fake)
    monkeypatch.setattr("raganything.parser.get_parser", get_parser)
    monkeypatch.setattr(
        "sys.argv",
        [
            "parser",
            "doc.pdf",
            "--output",
            "/tmp/out",
            "--method",
            "ocr",
            "--parser",
            "mineru",
            "--lang",
            "en",
            "--no-formula",
            "--no-table",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "Successfully parsed: doc.pdf" in out
    assert "Extracted 2 content blocks" in out
    get_parser.assert_called_once_with("mineru")
    fake.parse_document.assert_called_once()
    kwargs = fake.parse_document.call_args.kwargs
    assert kwargs["file_path"] == "doc.pdf"
    assert kwargs["method"] == "ocr"
    assert kwargs["output_dir"] == "/tmp/out"
    assert kwargs["lang"] == "en"
    assert kwargs["formula"] is False
    assert kwargs["table"] is False


def test_cli_stats_prints_content_type_distribution(monkeypatch, capsys):
    fake = MagicMock()
    fake.parse_document.return_value = [
        {"type": "text", "text": "a", "page_idx": 0},
        {"type": "text", "text": "b", "page_idx": 1},
        {"type": "table", "table_body": "|x|", "page_idx": 1},
        "not-a-dict",
    ]
    monkeypatch.setattr(
        "raganything.parser.get_parser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["parser", "doc.pdf", "--stats"],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "Document Statistics:" in out
    assert "Total content blocks: 4" in out
    assert "• text: 2" in out
    assert "• table: 1" in out


def test_cli_returns_one_when_parse_raises(monkeypatch, capsys):
    fake = MagicMock()
    fake.parse_document.side_effect = RuntimeError("mineru failed")
    monkeypatch.setattr(
        "raganything.parser.get_parser",
        MagicMock(return_value=fake),
    )
    monkeypatch.setattr("sys.argv", ["parser", "doc.pdf"])

    assert main() == 1
    assert "mineru failed" in capsys.readouterr().out


def test_cli_returns_one_when_get_parser_raises(monkeypatch, capsys):
    monkeypatch.setattr(
        "raganything.parser.get_parser",
        MagicMock(side_effect=ValueError("Unsupported parser type: nope")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["parser", "doc.pdf", "--parser", "nope"],
    )

    assert main() == 1
    assert "Unsupported parser type: nope" in capsys.readouterr().out
