"""Destructive knowledge-base operations require constrained, marked paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from raganything.kb_safety import (
    KB_MARKER,
    UnsafeKnowledgeBasePath,
    allowed_kb_roots,
    clear_prepared_kb_path,
    prepare_kb_paths,
)


def _roots(project: Path) -> tuple[Path, ...]:
    project.mkdir()
    return allowed_kb_roots(project, "")


def test_prepare_and_clear_only_owned_kb_children(tmp_path: Path) -> None:
    project = tmp_path / "app"
    roots = _roots(project)
    storage, parser = prepare_kb_paths(
        project / "kb" / "rag_storage",
        project / "kb" / "pipeline_parse",
        roots=roots,
    )
    (storage / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
    nested = parser / "manual"
    nested.mkdir()
    (nested / "manual_content_list.json").write_text("[]", encoding="utf-8")

    clear_prepared_kb_path(storage, role="storage", roots=roots)
    clear_prepared_kb_path(parser, role="parser", roots=roots)

    assert [path.name for path in storage.iterdir()] == [KB_MARKER]
    assert [path.name for path in parser.iterdir()] == [KB_MARKER]


def test_rejects_root_overlap_and_unrelated_nonempty_directories(
    tmp_path: Path,
) -> None:
    project = tmp_path / "app"
    roots = _roots(project)

    with pytest.raises(UnsafeKnowledgeBasePath, match="outside configured roots"):
        prepare_kb_paths(project, project / "pipeline_parse", roots=roots)

    with pytest.raises(UnsafeKnowledgeBasePath, match="must not overlap"):
        prepare_kb_paths(
            project / "kb",
            project / "kb" / "pipeline_parse",
            roots=roots,
        )

    unrelated = project / "scripts"
    unrelated.mkdir()
    (unrelated / "application.py").write_text("pass", encoding="utf-8")
    with pytest.raises(UnsafeKnowledgeBasePath, match="non-storage"):
        prepare_kb_paths(
            unrelated,
            project / "pipeline_parse",
            roots=roots,
        )


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "app"
    roots = _roots(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project / "linked-storage"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeKnowledgeBasePath, match="outside configured roots"):
        prepare_kb_paths(link, project / "pipeline_parse", roots=roots)


def test_clear_rejects_unmarked_directory(tmp_path: Path) -> None:
    project = tmp_path / "app"
    roots = _roots(project)
    storage = project / "rag_storage"
    storage.mkdir()

    with pytest.raises(UnsafeKnowledgeBasePath, match="not ownership-marked"):
        clear_prepared_kb_path(storage, role="storage", roots=roots)


def test_broad_configured_roots_are_rejected(tmp_path: Path) -> None:
    project = tmp_path / "app"
    project.mkdir()

    with pytest.raises(UnsafeKnowledgeBasePath, match="too broad"):
        allowed_kb_roots(project, str(Path.home()))
