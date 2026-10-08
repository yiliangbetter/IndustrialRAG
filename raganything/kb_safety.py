"""Guard destructive knowledge-base operations with explicit ownership markers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

KB_MARKER = ".industrialrag-kb.json"
_MARKER_VERSION = 1
_ROLES = frozenset({"storage", "parser"})


class UnsafeKnowledgeBasePath(ValueError):
    """A path is not safe enough for recursive knowledge-base cleanup."""


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def allowed_kb_roots(
    project_root: Path, configured: str | None = None
) -> tuple[Path, ...]:
    """Return roots under which KB directories may be adopted.

    ``RAG_WEB_KB_ALLOWED_ROOTS`` uses the platform path separator. The project
    root is always allowed so existing repository/client layouts keep working.
    Broad roots such as ``/`` and the user's home directory are rejected.
    """
    raw = configured
    if raw is None:
        raw = (os.getenv("RAG_WEB_KB_ALLOWED_ROOTS") or "").strip()
    candidates = [project_root]
    candidates.extend(Path(part) for part in raw.split(os.pathsep) if part.strip())

    home = Path.home().resolve()
    roots: list[Path] = []
    for candidate in candidates:
        root = _resolved(candidate)
        if root == Path(root.anchor) or root == home:
            raise UnsafeKnowledgeBasePath(
                f"KB allowed root is too broad: {root}. Choose a dedicated subdirectory."
            )
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _assert_allowed_target(path: Path, roots: Iterable[Path]) -> Path:
    target = _resolved(path)
    if target == Path(target.anchor) or target == Path.home().resolve():
        raise UnsafeKnowledgeBasePath(f"Refusing protected KB path: {target}")
    if not any(target != root and root in target.parents for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise UnsafeKnowledgeBasePath(
            f"KB path {target} is outside configured roots ({allowed})"
        )
    return target


def _read_marker(path: Path) -> dict | None:
    marker = path / KB_MARKER
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsafeKnowledgeBasePath(f"Invalid KB marker at {marker}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != _MARKER_VERSION:
        raise UnsafeKnowledgeBasePath(f"Unsupported KB marker at {marker}")
    return data


def _has_role_evidence(path: Path, role: str) -> bool:
    entries = [entry for entry in path.iterdir() if entry.name != KB_MARKER]
    if not entries:
        return True
    if role == "storage":
        prefixes = ("kv_store_", "vdb_", "graph_", "doc_status")
        return any(
            entry.is_file() and entry.name.startswith(prefixes) for entry in entries
        )
    if path.name == "pipeline_parse":
        return True
    try:
        return next(path.rglob("*_content_list*.json"), None) is not None
    except OSError:
        return False


def _validate_role_path(path: Path, role: str, roots: tuple[Path, ...]) -> Path:
    if role not in _ROLES:
        raise ValueError(f"Unknown KB path role: {role}")
    target = _assert_allowed_target(path, roots)
    if target.exists() and not target.is_dir():
        raise UnsafeKnowledgeBasePath(f"KB path is not a directory: {target}")
    if not target.exists():
        return target
    marker = _read_marker(target)
    if marker is not None:
        if marker.get("role") != role:
            raise UnsafeKnowledgeBasePath(
                f"KB marker role mismatch at {target}: expected {role}"
            )
        return target
    if not _has_role_evidence(target, role):
        raise UnsafeKnowledgeBasePath(
            f"Refusing to adopt non-{role} directory as a KB path: {target}"
        )
    return target


def prepare_kb_paths(
    storage: Path,
    parser: Path,
    *,
    roots: tuple[Path, ...],
) -> tuple[Path, Path]:
    """Validate, create, and mark a storage/parser pair before it becomes active."""
    storage = _validate_role_path(storage, "storage", roots)
    parser = _validate_role_path(parser, "parser", roots)
    if storage == parser or storage in parser.parents or parser in storage.parents:
        raise UnsafeKnowledgeBasePath(
            f"KB storage and parser paths must not overlap: {storage}, {parser}"
        )

    pair_id = hashlib.sha256(f"{storage}\0{parser}".encode()).hexdigest()
    for path, role in ((storage, "storage"), (parser, "parser")):
        path.mkdir(parents=True, exist_ok=True)
        marker = path / KB_MARKER
        marker.write_text(
            json.dumps(
                {"version": _MARKER_VERSION, "role": role, "pair_id": pair_id},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return storage, parser


def clear_prepared_kb_path(
    path: Path,
    *,
    role: str,
    roots: tuple[Path, ...],
) -> None:
    """Delete only children of a marked KB directory; retain its ownership marker."""
    target = _validate_role_path(path, role, roots)
    marker = _read_marker(target)
    if marker is None:
        raise UnsafeKnowledgeBasePath(f"KB path is not ownership-marked: {target}")

    for child in target.iterdir():
        if child.name == KB_MARKER:
            continue
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)
