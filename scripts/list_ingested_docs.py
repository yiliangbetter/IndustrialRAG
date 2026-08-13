#!/usr/bin/env python3
"""List unique documents indexed in LightRAG kv_store_full_docs.json."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def list_from_full_docs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for doc_id, item in data.items():
        if not isinstance(item, dict):
            continue
        fp = ""
        for key in ("file_path", "filepath", "source", "path"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                fp = val.strip()
                break
        name = fp.replace("\\", "/").rsplit("/", 1)[-1] if fp else str(doc_id)
        rows.append({"id": str(doc_id), "file_path": fp, "name": name})
    return rows


def summarize(label: str, wd: Path) -> list[str]:
    lines: list[str] = []
    fd = wd / "kv_store_full_docs.json"
    ch = wd / "vdb_chunks.json"
    rows = list_from_full_docs(fd)
    names = [r["name"] for r in rows]
    counts = Counter(names)
    uniq = sorted(counts)
    lines.append(f"=== {label} ===")
    lines.append(f"path: {wd.resolve()}")
    lines.append(f"full_docs records: {len(rows)} | unique filenames: {len(uniq)}")
    lines.append(f"vdb_chunks exists: {ch.is_file()}")
    if ch.is_file():
        try:
            chunks = json.loads(ch.read_text(encoding="utf-8"))
            lines.append(
                f"chunk records: {len(chunks) if isinstance(chunks, dict) else '?'}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"chunk read err: {exc}")
    lines.append("--- unique PDFs / docs ---")
    for i, name in enumerate(uniq, 1):
        cnt = counts[name]
        suffix = f"  (indexed {cnt}x)" if cnt > 1 else ""
        lines.append(f"{i:2}. {name}{suffix}")
    lines.append("")
    return lines


def main() -> None:
    roots = [Path(p) for p in sys.argv[1:]] or [
        _ROOT / "data" / "rag_storage",
        _ROOT / "rag_storage_run",
    ]
    lines: list[str] = []
    for wd in roots:
        if wd.is_dir():
            lines.extend(
                summarize(wd.name if wd.name != "rag_storage" else str(wd), wd)
            )

    pp = _ROOT / "data" / "pipeline_parse"
    if pp.is_dir():
        lines.append("=== data/pipeline_parse (parsed outputs) ===")
        for d in sorted(pp.iterdir()):
            if d.is_dir():
                base = d.name.rsplit("_", 1)[0] if "_" in d.name else d.name
                lines.append(f"- {base}")
        lines.append("")

    text = "\n".join(lines)
    print(text)


if __name__ == "__main__":
    main()
