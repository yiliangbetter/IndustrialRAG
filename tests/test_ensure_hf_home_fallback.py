"""Regression tests for local HF cache path fallback.

ensure_hf_home_from_repo_fallback sets HF_HOME from <repo>/.hf_cache when unset.
Wrong behavior breaks offline/local embedding runs that rely on a checked-in
or pre-populated cache directory without exporting HF_HOME.
"""

import os

from raganything.local_hf_embedding import ensure_hf_home_from_repo_fallback


def test_sets_hf_home_from_existing_repo_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    cache = tmp_path / ".hf_cache"
    cache.mkdir()

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert os.environ["HF_HOME"] == str(cache)


def test_does_not_override_existing_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/already/set")
    cache = tmp_path / ".hf_cache"
    cache.mkdir()

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert os.environ["HF_HOME"] == "/already/set"


def test_noop_when_repo_root_missing_or_cache_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)

    ensure_hf_home_from_repo_fallback(None)
    assert "HF_HOME" not in os.environ

    ensure_hf_home_from_repo_fallback(tmp_path)
    assert "HF_HOME" not in os.environ
