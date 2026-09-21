"""The response supplement pipeline must follow the active knowledge base."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import iqr_machine  # noqa: E402
import iqr_store  # noqa: E402
import query_doc_steering  # noqa: E402


def _write_store(root: Path, chunk_id: str, machine: str, *, mtime_ns: int) -> None:
    root.mkdir(parents=True)
    path = root / "kv_store_text_chunks.json"
    path.write_text(
        json.dumps(
            {
                chunk_id: {
                    "content": f"content from {machine}",
                    "file_path": f"{machine}.pdf",
                    "machine": machine,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.utime(path, ns=(mtime_ns, mtime_ns))


@pytest.fixture(autouse=True)
def _reset_storage_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iqr_store, "_kv_store_cache", None)
    monkeypatch.setattr(iqr_store, "_kv_store_revision", None)
    monkeypatch.setattr(iqr_store, "_cl_path_index", None)
    monkeypatch.setattr(iqr_machine, "_vocab_cache", None)
    monkeypatch.setattr(iqr_machine, "_vocab_revision", None)


def test_storage_resolver_prefers_active_web_kb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active = tmp_path / "selected-kb"
    monkeypatch.setenv("RAG_WEB_WORKING_DIR", str(active))
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "stale-generic-kb"))

    assert query_doc_steering._rag_storage_dir() == active.resolve()


def test_text_and_machine_caches_are_keyed_by_kb_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "rag_storage"
    second = tmp_path / "second" / "rag_storage"
    shared_mtime_ns = 1_700_000_000_000_000_000
    _write_store(first, "chunk-first", "Machine A", mtime_ns=shared_mtime_ns)
    _write_store(second, "chunk-second", "Machine B", mtime_ns=shared_mtime_ns)

    monkeypatch.setenv("RAG_WEB_WORKING_DIR", str(first))
    assert set(iqr_store._kv_text_chunks_store()) == {"chunk-first"}
    assert iqr_machine.known_machine_names() == ("Machine A",)

    monkeypatch.setenv("RAG_WEB_WORKING_DIR", str(second))
    assert set(iqr_store._kv_text_chunks_store()) == {"chunk-second"}
    assert iqr_machine.known_machine_names() == ("Machine B",)


def test_content_list_index_is_keyed_by_parser_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "pipeline_parse"
    second = tmp_path / "second" / "pipeline_parse"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "first_content_list.json").write_text("[]", encoding="utf-8")
    (second / "second_content_list.json").write_text("[]", encoding="utf-8")

    monkeypatch.setenv("RAG_WEB_PARSER_OUTPUT_DIR", str(first))
    first_entries = iqr_store._pipeline_content_list_entries()
    assert any(path.name == "first_content_list.json" for path, _, _ in first_entries)

    monkeypatch.setenv("RAG_WEB_PARSER_OUTPUT_DIR", str(second))
    second_entries = iqr_store._pipeline_content_list_entries()
    assert any(path.name == "second_content_list.json" for path, _, _ in second_entries)
    assert not any(
        path.name == "first_content_list.json" for path, _, _ in second_entries
    )
