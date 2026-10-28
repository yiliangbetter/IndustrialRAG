"""Shared test-case loading for query batch / diagnostic scripts.

Parses shili17 (``docs/测试例.txt``) and canonical or legacy voice-script JSON
case sources into ``{id, query, ...}`` rows.
Extracted from a since-removed scoring script so that clarify-gate
and rerank diagnostics keep a single case-loading implementation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CANONICAL_VOICE29 = _ROOT / "tests" / "fixtures" / "evaluation" / "voice29.json"

_SOURCE_PRESETS: dict[str, Path] = {
    "shili": _ROOT / "docs" / "测试例.txt",
    "shili17": _ROOT / "docs" / "测试例.txt",
    "voice": _CANONICAL_VOICE29,
    "voice29": _CANONICAL_VOICE29,
    "voice_green8": _CANONICAL_VOICE29,
    "green8": _CANONICAL_VOICE29,
}


def _parse_shili_questions(path: Path, *, limit: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("—"):
            break
        m = re.match(r"^(\d+)\.(.+)$", line)
        if not m:
            continue
        num, rest = m.group(1), m.group(2).strip()
        if not rest or rest.startswith("是否配图"):
            continue
        rows.append({"id": num, "query": rest})
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _parse_voice_script_cases(
    path: Path, *, limit: int = 0, group: str | None = None
) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") == "rag-evaluation-dataset/v1":
        from raganything.evaluation_legacy import voice29_legacy_cases
        from raganything.evaluation_schema import load_evaluation_dataset

        cases = voice29_legacy_cases(load_evaluation_dataset(path))
    else:
        cases = data.get("cases") or []
    group_filter = (group or "").strip()
    rows: list[dict[str, str]] = []
    for case in cases:
        if case.get("skip"):
            continue
        case_group = str(case.get("group") or "").strip()
        if group_filter and case_group != group_filter:
            continue
        q = (case.get("standard_question") or "").strip()
        if not q:
            utterances = case.get("utterances") or []
            q = (utterances[0] if utterances else "").strip()
        if not q:
            continue
        rows.append(
            {
                "id": str(case.get("id", len(rows) + 1)),
                "query": q,
                "group": str(case.get("group") or ""),
                "category": str(case.get("category") or ""),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _load_cases(
    source: Path, *, limit: int, group: str | None = None
) -> list[dict[str, str]]:
    if source.suffix.lower() == ".json":
        return _parse_voice_script_cases(source, limit=limit, group=group)
    return _parse_shili_questions(source, limit=limit)


def _load_preset_cases(
    preset: str, *, limit: int = 0, group: str | None = None
) -> list[dict[str, str]]:
    """Load a named source, including the canonical Green8 subset view."""
    try:
        source = _SOURCE_PRESETS[preset]
    except KeyError:
        raise ValueError(f"Unknown query case preset: {preset}") from None
    rows = _load_cases(source, limit=0, group=group)
    if preset in {"green8", "voice_green8"}:
        from raganything.evaluation_schema import load_evaluation_dataset

        dataset = load_evaluation_dataset(_CANONICAL_VOICE29)
        wanted = {str(case_id) for case_id in dataset.metadata["green8_voice_ids"]}
        rows = [row for row in rows if row["id"] in wanted]
    return rows[:limit] if limit > 0 else rows
