#!/usr/bin/env python3
"""Remove LightRAG **query-answer** cache entries (not ingest extract cache).

The file ``kv_store_llm_response_cache.json`` also holds灌库时的 extract/summary/keywords
entries. Deleting those does **not** fix wrong Q&A answers and slows re-ingest.

Query entries are recognized by:
  - key prefix ``default:query:``, or
  - value field ``cache_type`` == ``query``

Examples::

  uv run python scripts/clear_query_llm_cache.py -w rag_storage_run
  uv run python scripts/clear_query_llm_cache.py -w rag_storage_run --stats

Stop the web server before writing the cache file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _is_query_entry(key: str, value: object) -> bool:
    if key.startswith("default:query:"):
        return True
    return isinstance(value, dict) and value.get("cache_type") == "query"


def _summarize(data: dict) -> dict[str, int]:
    by_key_prefix: Counter[str] = Counter()
    by_cache_type: Counter[str] = Counter()
    query_count = 0
    for key, value in data.items():
        if _is_query_entry(key, value):
            query_count += 1
        prefix = key.split(":", 2)[1] if key.count(":") >= 2 else key.split(":")[0]
        by_key_prefix[prefix] += 1
        if isinstance(value, dict):
            by_cache_type[str(value.get("cache_type", "?"))] += 1
        else:
            by_cache_type["?"] += 1
    return {
        "total": len(data),
        "query": query_count,
        "by_key_prefix": dict(by_key_prefix),
        "by_cache_type": dict(by_cache_type),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "rag_storage_run",
        help="LightRAG working directory",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Only print cache breakdown, do not delete",
    )
    args = p.parse_args()
    cache_path = args.working_dir / "kv_store_llm_response_cache.json"
    if not cache_path.is_file():
        print(f"未找到缓存文件: {cache_path}")
        return

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    before = _summarize(data)

    print(f"缓存文件: {cache_path}")
    print(f"  总条目: {before['total']}")
    print(f"  问答缓存 (query): {before['query']} 条")
    print(f"  按 key 段统计: {before['by_key_prefix']}")
    print(f"  按 cache_type: {before['by_cache_type']}")
    print()
    print(
        "说明: extract/summary/keywords 是灌库时图谱抽取用的缓存，"
        "不是浏览器里看到的那次「问答回复」。"
    )
    print(
        "若 query=0，说明问答缓存本来就是空的；"
        "答案仍不对时请确认 .env 中 ENABLE_LLM_CACHE=false 并已重启 Web。"
    )

    if args.stats:
        return

    kept = {k: v for k, v in data.items() if not _is_query_entry(k, v)}
    removed = before["total"] - len(kept)
    if removed == 0:
        print()
        print("未删除任何条目（没有问答 query 缓存可清）。")
        return

    cache_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    after = _summarize(kept)
    print()
    print(f"已删除 {removed} 条问答缓存；剩余 {after['total']} 条（灌库缓存）。")


if __name__ == "__main__":
    main()
