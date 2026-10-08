"""image_query_refs submodule ``iqr_media``.

Media path resolution, safety checks, and media-token encode/decode.
"""

from __future__ import annotations

import base64
from pathlib import Path
from iqr_protocol import (
    _IMAGE_EXTS,
    normalize_image_path,
)


def resolve_media_path(path_str: str, media_roots: list[Path]) -> Path | None:
    """Resolve image path; fall back to filename search under parser output (re-ingest safe)."""
    path = Path(normalize_image_path(path_str))
    if is_safe_media_path(path, media_roots):
        return path.resolve()
    name = path.name
    if not name:
        return None
    for root in media_roots:
        try:
            for candidate in root.rglob(name):
                if is_safe_media_path(candidate, media_roots):
                    return candidate.resolve()
        except OSError:
            continue
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
