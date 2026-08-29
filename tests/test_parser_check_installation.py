"""MinerU and Docling install checks must fail closed on missing binaries.

`_ensure_lightrag_initialized` and `verify_parser_installation_once` both
call these helpers. A false positive would let ingest start without a parser;
a false negative would block a working install.
"""

import subprocess
from types import SimpleNamespace

import raganything.parser as parser_mod
from raganything.parser import DoclingParser, MineruParser


def _ok_result(stdout="1.0.0\n"):
    return SimpleNamespace(stdout=stdout, returncode=0)


class TestMineruCheckInstallation:
    def test_success_returns_true(self, monkeypatch):
        monkeypatch.setattr(
            parser_mod.subprocess, "run", lambda *args, **kwargs: _ok_result()
        )
        assert MineruParser().check_installation() is True

    def test_missing_binary_returns_false(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("mineru")

        monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)
        assert MineruParser().check_installation() is False

    def test_nonzero_exit_returns_false(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1, cmd=["mineru", "--version"]
            )

        monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)
        assert MineruParser().check_installation() is False

    def test_windows_hides_console_window(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(parser_mod, "_IS_WINDOWS", True)
        monkeypatch.setattr(
            parser_mod.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
        )

        def fake_run(*args, **kwargs):
            captured.update(kwargs)
            return _ok_result()

        monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)
        assert MineruParser().check_installation() is True
        assert captured.get("creationflags") == 0x08000000


class TestDoclingCheckInstallation:
    def test_success_returns_true(self, monkeypatch):
        monkeypatch.setattr(
            parser_mod.subprocess, "run", lambda *args, **kwargs: _ok_result()
        )
        assert DoclingParser().check_installation() is True

    def test_missing_binary_returns_false(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("docling")

        monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)
        assert DoclingParser().check_installation() is False

    def test_nonzero_exit_returns_false(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1, cmd=["docling", "--version"]
            )

        monkeypatch.setattr(parser_mod.subprocess, "run", fake_run)
        assert DoclingParser().check_installation() is False
