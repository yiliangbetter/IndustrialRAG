"""Regression tests for BatchParser.process_single_file success/error contract.

This is the unit of work for all batch workers; silent exception handling and
per-stem output directories directly shape reported success rates.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raganything.batch_parser import BatchParser


@pytest.fixture
def batch_parser(monkeypatch):
    fake_parser = MagicMock()
    fake_parser.OFFICE_FORMATS = {".docx"}
    fake_parser.IMAGE_FORMATS = {".png"}
    fake_parser.TEXT_FORMATS = {".txt", ".md"}
    fake_parser.check_installation.return_value = True
    monkeypatch.setattr(
        "raganything.batch_parser.get_parser",
        MagicMock(return_value=fake_parser),
    )
    bp = BatchParser(parser_type="mineru", skip_installation_check=True)
    bp.parser = fake_parser
    return bp


def test_process_single_file_success_uses_stem_output_dir(batch_parser, tmp_path):
    src = tmp_path / "reports" / "Q1 Report.pdf"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"%PDF")
    out = tmp_path / "out"
    batch_parser.parser.parse_document.return_value = [{"type": "text", "text": "hi"}]

    ok, path, err = batch_parser.process_single_file(
        str(src), str(out), parse_method="ocr", lang="en"
    )

    assert ok is True
    assert path == str(src)
    assert err is None
    expected_dir = out / "Q1 Report"
    assert expected_dir.is_dir()
    kwargs = batch_parser.parser.parse_document.call_args.kwargs
    assert kwargs["file_path"] == str(src)
    assert kwargs["output_dir"] == str(expected_dir)
    assert kwargs["method"] == "ocr"
    assert kwargs["lang"] == "en"


def test_process_single_file_returns_false_and_message_on_parser_error(
    batch_parser, tmp_path
):
    src = tmp_path / "broken.pdf"
    src.write_bytes(b"%PDF")
    out = tmp_path / "out"
    batch_parser.parser.parse_document.side_effect = RuntimeError("parse exploded")

    ok, path, err = batch_parser.process_single_file(str(src), str(out))

    assert ok is False
    assert path == str(src)
    assert err is not None
    assert "broken.pdf" in err
    assert "parse exploded" in err
    # Output dir is still created before parse; failure must not raise.
    assert (out / "broken").is_dir()
