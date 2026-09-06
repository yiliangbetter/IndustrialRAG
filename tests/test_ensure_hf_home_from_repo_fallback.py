"""Regression tests for local HF cache discovery.

Every ingest CLI calls ``ensure_hf_home_from_repo_fallback`` so offline
sentence-transformers loads use ``<repo>/.hf_cache`` when ``HF_HOME`` is unset.
Overwriting a user-set ``HF_HOME`` or inventing a missing cache dir would send
embeddings to the wrong models (or to the network).
"""

import os

from raganything.local_hf_embedding import ensure_hf_home_from_repo_fallback


def test_existing_hf_home_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "/already/set")
    cache = tmp_path / ".hf_cache"
    cache.mkdir()

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert os.environ["HF_HOME"] == "/already/set"


def test_sets_hf_home_when_repo_cache_dir_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    cache = tmp_path / ".hf_cache"
    cache.mkdir()

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert os.environ["HF_HOME"] == str(cache)


def test_noop_when_repo_root_is_none(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)

    ensure_hf_home_from_repo_fallback(None)

    assert "HF_HOME" not in os.environ


def test_noop_when_repo_cache_dir_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert "HF_HOME" not in os.environ


def test_noop_when_hf_cache_is_a_file_not_a_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    (tmp_path / ".hf_cache").write_text("not a dir", encoding="utf-8")

    ensure_hf_home_from_repo_fallback(tmp_path)

    assert "HF_HOME" not in os.environ
