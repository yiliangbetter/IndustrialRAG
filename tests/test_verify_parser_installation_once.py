"""Parser install probe is cached only after it succeeds.

A failed check must stay uncached so the next call probes again. A successful
check must not call the parser a second time. Caching a failure would let
later documents skip a missing MinerU/Docling install.
"""

import pytest

from raganything.raganything import RAGAnything


class _Logger:
    def info(self, *args, **kwargs):
        return None


class _Probe:
    def __init__(self, installed):
        self.installed = installed
        self.calls = 0

    def check_installation(self):
        self.calls += 1
        return self.installed


def _rag(parser_name, installed):
    rag = object.__new__(RAGAnything)
    rag.doc_parser = _Probe(installed)
    rag._parser_installation_checked = False
    rag.config = type("Config", (), {"parser": parser_name})()
    rag.logger = _Logger()
    return rag


def test_failed_install_probe_is_not_cached():
    rag = _rag("docling", installed=False)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="docling") as excinfo:
            rag.verify_parser_installation_once()
        assert "not properly installed" in str(excinfo.value)

    assert rag._parser_installation_checked is False
    assert rag.doc_parser.calls == 2


def test_successful_install_probe_is_cached():
    rag = _rag("mineru", installed=True)

    assert rag.verify_parser_installation_once() is True
    assert rag.verify_parser_installation_once() is True
    assert rag._parser_installation_checked is True
    assert rag.doc_parser.calls == 1
