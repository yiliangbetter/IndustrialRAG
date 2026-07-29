"""Shared test-case loading for query batch / diagnostic scripts.

Parses shili17 (``docs/测试例.txt``) and voice-script JSON
(``data/voice_script_*.json``) case sources into ``{id, query, ...}`` rows.
Extracted from a since-removed scoring script so that clarify-gate
and rerank diagnostics keep a single case-loading implementation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_SOURCE_PRESETS: dict[str, Path] = {
    "shili": _ROOT / "docs" / "测试例.txt",
    "shili17": _ROOT / "docs" / "测试例.txt",
    "voice": _ROOT / "data" / "voice_script_tests.json",
    "voice29": _ROOT / "data" / "voice_script_tests.json",
    "voice_green8": _ROOT / "data" / "voice_script_green8.json",
    "green8": _ROOT / "data" / "voice_script_green8.json",
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
    group_filter = (group or "").strip()
    rows: list[dict[str, str]] = []
    for case in data.get("cases") or []:
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


def _load_cases(source: Path, *, limit: int, group: str | None = None) -> list[dict[str, str]]:
    if source.suffix.lower() == ".json":
        return _parse_voice_script_cases(source, limit=limit, group=group)
    return _parse_shili_questions(source, limit=limit)
