"""image_query_refs submodule ``iqr_media``.

Media path resolution, safety checks, and media-token encode/decode.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from iqr_protocol import (
    _IMAGE_EXTS,
    image_document_relative_identity,
    normalize_image_path,
)


def _source_key(value: str) -> str:
    label = Path((value or "").replace("\\", "/")).stem
    return re.sub(r"\s+", "", label).casefold()


def _path_has_source(path: Path, source_hint: str) -> bool:
    expected = _source_key(source_hint)
    if not expected:
        return True
    return any(_source_key(part) == expected for part in path.parts)


def _same_file_content(paths: list[Path]) -> bool:
    """True when duplicate parser trees contain byte-identical media."""
    try:
        sizes = {path.stat().st_size for path in paths}
        if len(sizes) != 1:
            return False
        digests: set[str] = set()
        for path in paths:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digests.add(digest.hexdigest())
        return len(digests) == 1
    except OSError:
        return False


def resolve_media_path(
    path_str: str,
    media_roots: list[Path],
    *,
    source_hint: str = "",
) -> Path | None:
    """Resolve media without crossing document provenance on stale-path fallback."""
    path = Path(normalize_image_path(path_str))
    if is_safe_media_path(path, media_roots):
        resolved = path.resolve()
        return resolved if _path_has_source(resolved, source_hint) else None
    name = path.name
    if not name:
        return None
    expected_suffix = image_document_relative_identity(path_str, source_hint)
    candidates: dict[str, Path] = {}
    for root in media_roots:
        try:
            for candidate in root.rglob(name):
                if not is_safe_media_path(candidate, media_roots):
                    continue
                resolved = candidate.resolve()
                if source_hint and not _path_has_source(resolved, source_hint):
                    continue
                if source_hint and expected_suffix:
                    actual_suffix = image_document_relative_identity(
                        str(resolved), source_hint
                    )
                    if actual_suffix != expected_suffix:
                        continue
                candidates[str(resolved)] = resolved
        except OSError:
            continue
    matches = sorted(candidates.values(), key=lambda candidate: str(candidate))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and source_hint and _same_file_content(matches):
        return matches[0]
    return None


def is_safe_media_path(path: Path, allowed_roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    if resolved.suffix.lower() not in _IMAGE_EXTS:
        return False
    for root in allowed_roots:
        try:
            root_res = root.resolve()
            if resolved.is_relative_to(root_res):
                return True
        except OSError:
            continue
    return False


def encode_media_token(abs_path: Path, media_root: Path) -> str:
    rel = abs_path.resolve().relative_to(media_root.resolve())
    raw = str(rel).replace("\\", "/").encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_media_token(token: str, media_root: Path) -> Path | None:
    if not token or ".." in token or token.startswith("/"):
        return None
    try:
        pad = "=" * (-len(token) % 4)
        rel = base64.urlsafe_b64decode(token + pad).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    rel_path = Path(rel)
    if ".." in rel_path.parts or rel_path.is_absolute():
        return None
    resolved = (media_root.resolve() / rel_path).resolve()
    # Guard against path traversal: ensure result stays under media_root
    try:
        resolved.relative_to(media_root.resolve())
    except ValueError:
        return None
    return resolved
