"""Legacy MINERU_PARSE_METHOD still selects the document parse method.

Old deployments set MINERU_PARSE_METHOD instead of PARSE_METHOD. Config
construction copies that value onto parse_method only when PARSE_METHOD is
unset. A present PARSE_METHOD, or a blank legacy value, must not replace the
method that ingest already bound.
"""

import warnings

import pytest

from raganything.config import RAGAnythingConfig


def _baseline_parse_method(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("PARSE_METHOD", raising=False)
    monkeypatch.delenv("MINERU_PARSE_METHOD", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return RAGAnythingConfig().parse_method


def _legacy_value_distinct_from(baseline: str) -> str:
    return "txt" if baseline == "ocr" else "ocr"


def test_legacy_env_overrides_parse_method_when_parse_method_unset(monkeypatch):
    baseline = _baseline_parse_method(monkeypatch)
    legacy = _legacy_value_distinct_from(baseline)
    monkeypatch.setenv("MINERU_PARSE_METHOD", legacy)

    with pytest.warns(DeprecationWarning, match="MINERU_PARSE_METHOD is deprecated"):
        config = RAGAnythingConfig()

    assert config.parse_method == legacy


def test_parse_method_env_blocks_legacy_override(monkeypatch):
    baseline = _baseline_parse_method(monkeypatch)
    legacy = _legacy_value_distinct_from(baseline)
    modern = "txt" if legacy == "ocr" else "ocr"
    monkeypatch.setenv("PARSE_METHOD", modern)
    monkeypatch.setenv("MINERU_PARSE_METHOD", legacy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = RAGAnythingConfig()

    assert config.parse_method == baseline
    assert config.parse_method != legacy
    assert not any(
        "MINERU_PARSE_METHOD is deprecated" in str(item.message) for item in caught
    )


def test_blank_legacy_env_does_not_override_parse_method(monkeypatch):
    baseline = _baseline_parse_method(monkeypatch)
    monkeypatch.setenv("MINERU_PARSE_METHOD", "")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = RAGAnythingConfig()

    assert config.parse_method == baseline
    assert not any(
        "MINERU_PARSE_METHOD is deprecated" in str(item.message) for item in caught
    )
