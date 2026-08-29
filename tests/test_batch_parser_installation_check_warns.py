"""BatchParser must warn on a failed install check, not refuse to construct.

get_supported_file_extensions / filter_supported_files build a BatchParser
without skip_installation_check. Raising here would make batch listing
unusable on machines where mineru --version is flaky even though parsing
still works.
"""

import logging

from raganything.batch_parser import BatchParser


class StubParser:
    OFFICE_FORMATS = {".docx"}
    IMAGE_FORMATS = {".png"}
    TEXT_FORMATS = {".txt"}

    def __init__(self):
        self.checks = 0

    def check_installation(self):
        self.checks += 1
        return False


def test_failed_check_warns_and_still_exposes_extensions(monkeypatch, caplog):
    stub = StubParser()
    monkeypatch.setattr("raganything.batch_parser.get_parser", lambda parser_type: stub)

    with caplog.at_level(logging.WARNING):
        parser = BatchParser(parser_type="mineru", skip_installation_check=False)

    assert parser.parser is stub
    assert stub.checks == 1
    assert ".pdf" in parser.get_supported_extensions()
    assert ".docx" in parser.get_supported_extensions()
    assert any(
        "installation check failed" in record.message for record in caplog.records
    )


def test_skip_installation_check_does_not_probe_parser(monkeypatch):
    stub = StubParser()
    monkeypatch.setattr("raganything.batch_parser.get_parser", lambda parser_type: stub)

    parser = BatchParser(parser_type="paddleocr", skip_installation_check=True)

    assert parser.parser is stub
    assert stub.checks == 0
    assert parser.parser_type == "paddleocr"
