#!/usr/bin/env python3
"""Audit Q15 电控板 section chunks after re-ingest."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from query_doc_steering import _text_chunks_store_path  # noqa: E402
import image_query_refs as iq  # noqa: E402

EXPECTED = {
    "高速智能封边机": {
        "section": "3.14.5",
        "cycle": "季度",
        "preferred_imgs": ["e06e7d6b", "9bee2818"],
        "bad_imgs": [],
        "caption_hint": "检查电控板设备",
    },
    "双端封边机": {
        "section": "3.2",
        "cycle": "半年",
        "preferred_imgs": ["fdf72fc8", "7cd4b93e"],
        "bad_imgs": [],
        "caption_hint": "",
    },
    "自动封边机": {
        "section": "3.2",
        "cycle": "半年",
        "preferred_imgs": ["d956986a"],
        "bad_imgs": ["e818188c"],
        "caption_hint": "",
    },
    "高速自动封边机": {
        "section": "3.2",
        "cycle": "半年",
        "preferred_imgs": ["58ebbf6c"],
        "bad_imgs": ["491ccaf0"],
        "caption_hint": "",
    },
}


def img_hash(path: object) -> str:
    return Path(str(path or "")).name[:12]


def matches_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(p[:8]) for p in prefixes)


def main() -> int:
    store = _text_chunks_store_path()
    raw = json.loads(store.read_text(encoding="utf-8"))
    print(f"KV store: {store}")
    print(f"Total chunks: {len(raw)}\n")

    issues: list[str] = []
    summary: list[tuple[str, str, int, int]] = []

    for hint, exp in EXPECTED.items():
        chunks = iq._load_manual_chunks_for_locality(hint)
        figure_hits: list[dict] = []
        heading_only: list[tuple[int | None, str, str]] = []

        for doc in chunks:
            content = iq._doc_content(doc)
            idx = iq._doc_chunk_order_index(doc)
            cid = str(doc.get("id", ""))[:22]
            refs = iq.extract_image_refs_from_context(content)

            if "3.2 电控" in content or "3.14.5" in content or (
                content.strip().startswith("3.2") and "电控" in content[:30]
            ):
                heading_only.append((idx, cid, content[:100].replace("\n", " ")))

            if "电控" in content or ("控制箱" in content and refs):
                page_m = re.search(r"页码[：:]\s*(\d+)", content)
                page = page_m.group(1) if page_m else "-"
                row = {
                    "idx": idx,
                    "id": cid,
                    "page": page,
                    "refs": [
                        (
                            img_hash(r.get("path")),
                            (iq._ref_effective_label(r) or "-")[:30],
                            iq._ref_inline_context_text(r)[:60],
                        )
                        for r in refs
                    ],
                    "head": content.split("\n")[0][:50],
                }
                if refs:
                    figure_hits.append(row)

        print("=" * 70)
        print(f"{hint} | manual chunks: {len(chunks)}")
        print(f"  Expected section: {exp['section']} | cycle: {exp['cycle']}")

        print("  --- Section heading chunks ---")
        for idx, cid, txt in heading_only[:5]:
            print(f"    idx={idx} id={cid} | {txt}")

        print("  --- Figure chunks (电控/控制箱) ---")
        found_good: set[str] = set()
        found_bad: set[str] = set()
        for row in figure_hits:
            flags: list[str] = []
            for h, _lab, _ctx in row["refs"]:
                if matches_prefix(h, exp["preferred_imgs"]):
                    found_good.add(h)
                    flags.append("OK")
                elif matches_prefix(h, exp["bad_imgs"]):
                    found_bad.add(h)
                    flags.append("BAD")
                else:
                    flags.append("?")
            print(
                f"    idx={row['idx']} page={row['page']} id={row['id']} flags={flags}"
            )
            for h, lab, ctx in row["refs"]:
                print(f"      img={h} label={lab}")
                print(f"      ctx={ctx}")

        missing_good = [
            p
            for p in exp["preferred_imgs"]
            if not any(p[:8] in g for g in found_good)
        ]
        has_bad = bool(found_bad)
        warn_in_chunk = any(
            "Warning" in lab or "警告" in lab for row in figure_hits for _h, lab, _ctx in row["refs"]
        )
        caption_ok = any(
            exp["caption_hint"] and exp["caption_hint"] in lab
            for row in figure_hits
            for _h, lab, _ctx in row["refs"]
        )

        if missing_good:
            issues.append(f"{hint}: missing expected img {missing_good}")
        if has_bad:
            issues.append(f"{hint}: still has bad img {sorted(found_bad)}")
        if hint == "自动封边机" and warn_in_chunk:
            issues.append(f"{hint}: Warning label still in 电控 chunk")
        if hint == "高速智能封边机" and exp["caption_hint"] and not caption_ok:
            issues.append(f"{hint}: caption '{exp['caption_hint']}' not found on figure ref")

        ok = not missing_good and not has_bad and not (
            hint == "自动封边机" and warn_in_chunk
        )
        summary.append((hint, "PASS" if ok else "FAIL", len(figure_hits), len(heading_only)))
        print(f"  VERDICT: {'PASS' if ok else 'FAIL'} | good={sorted(found_good)} bad={sorted(found_bad)}")
        if hint == "高速智能封边机":
            print(f"  caption '{exp['caption_hint']}': {'found' if caption_ok else 'missing'}")
        print()

    print("=" * 70)
    print("SUMMARY")
    for hint, verdict, nf, nh in summary:
        print(f"  {hint}: {verdict} (figure_chunks={nf}, heading_chunks={nh})")
    print()
    if issues:
        print("ISSUES:")
        for item in issues:
            print(f"  - {item}")
        return 1
    print("All 4 manuals ingest audit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
