"""BatchParser must warn on a failed install check, not refuse to construct.

Unlike RAGAnything.verify_parser_installation_once, batch parsing is allowed
to proceed so callers can still dry-run or hit a working binary that the
`--version` probe missed. Raising here would block the whole folder parse.
"""

import logging

import raganything.batch_parser as batch_parser_mod
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


def test_failed_install_check_warns_and_still_constructs(monkeypatch, caplog):
    stub = StubParser()
    monkeypatch.setattr(batch_parser_mod, "get_parser", lambda name: stub)

    with caplog.at_level(logging.WARNING, logger="raganything.batch_parser"):
        parser = BatchParser(parser_type="mineru", skip_installation_check=False)

    assert parser.parser is stub
    assert stub.checks == 1
    assert any(
        "installation check failed" in record.message for record in caplog.records
    )


def test_skip_installation_check_does_not_probe(monkeypatch):
    stub = StubParser()
    monkeypatch.setattr(batch_parser_mod, "get_parser", lambda name: stub)

    parser = BatchParser(parser_type="mineru", skip_installation_check=True)

    assert parser.parser is stub
    assert stub.checks == 0
