#!/usr/bin/env python3
"""Diagnose Q3 citation: answer cycle-only vs content+figure chunk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from image_query_refs import (  # noqa: E402
    _answer_body_for_citation_match,
    _chunk_citation_score,
    filter_docs_cited_by_answer,
)

QUERY = "高速智能封边机机床床身清洁，要多长时间做一次？"

ANSWERS = {
    "latest_fail": (
        "根据高速智能封边机维护保养手册的规定，机床床身清洁的保养周期为**每天一次**。\n\n"
        "手册中明确指出，“每天一次”作为保养执行的频率标准，规定了机床床身保养的执行周期。"
        "这属于机床日常清洁与保养的重要内容，旨在确保设备在日常运行中得到及时的清理与维护，"
        "保持良好的运行状态[1]。\n\n### References\n- [1] 高速智能封边机维护保养手册.pdf"
    ),
    "old_pass": (
        "根据提供的信息，高速智能封边机的机床床身清洁（即机床外部清洁、清理粉尘及木屑）"
        "属于设备的日常保养项目，其保养周期为**每天保养一次**或**每班次保养一次**。\n\n"
        "在日常保养中，主要需要对机床外部进行清洁，清理残存的粉尘及木屑。"
        "在进行此操作时，请注意切勿划伤视窗口的有机玻璃。\n\n### References\n\n"
        "- [1] 高速智能封边机维护保养手册.pdf"
    ),
    "minimal_cycle": "机床床身清洁的保养周期为每天一次。",
    "with_section": "2.1.1 机床床身清洁的保养周期为每天一次。",
}


def main() -> None:
    store = json.loads(
        (_ROOT / "data/rag_storage/kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    fig_chunk = None
    title_chunk = None
    cycle_only = None
    for rec in store.values():
        c = rec.get("content") or ""
        if "机床外部清洁" in c and "[图片]" in c:
            fig_chunk = c
        if c.strip() == "2.1.1 机床床身清洁":
            title_chunk = c
        if c.strip() == "保养内容：机床外部清洁":
            cycle_only = c

    print("=== KB chunks ===")
    print(f"figure chunk: {'yes' if fig_chunk else 'no'} len={len(fig_chunk or '')}")
    print(f"title-only 2.1.1: {'yes' if title_chunk else 'no'}")
    print(f"保养内容 line only: {'yes' if cycle_only else 'no'}")
    if fig_chunk:
        print("\n--- figure chunk preview ---")
        print(fig_chunk[:400])

    docs = []
    if fig_chunk:
        docs.append({"content": fig_chunk})
    if title_chunk:
        docs.append({"content": title_chunk})

    print("\n=== citation scores (figure chunk) ===")
    for name, ans in ANSWERS.items():
        blob = _answer_body_for_citation_match(ans)
        score = _chunk_citation_score(blob, fig_chunk or "")
        kept, meta = filter_docs_cited_by_answer(
            ans, docs, query=QUERY, pool=docs
        )
        print(f"{name}: score={score:.3f} kept={len(kept)} mode={meta.get('mode')}")
        if meta.get("kept_scores"):
            print(f"  kept_scores={meta['kept_scores']}")

    print("\n=== citation on title-only chunk ===")
    for name in ("latest_fail", "with_section"):
        blob = _answer_body_for_citation_match(ANSWERS[name])
        score = _chunk_citation_score(blob, title_chunk or "")
        print(f"{name}: title score={score:.3f}")


if __name__ == "__main__":
    main()
