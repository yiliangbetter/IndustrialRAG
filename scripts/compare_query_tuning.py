#!/usr/bin/env python3
"""Compare rerank-threshold vs image-anchor tuning on docs/测试问题.txt.

Runs each question under named profiles (baseline, high_rerank, image_anchor, both)
using ``only_need_context=True`` (no answer LLM). Reports chunk counts, context size,
clarify band, and image captions.

Examples::

  uv run python scripts/compare_query_tuning.py -w data/rag_storage
  uv run python scripts/compare_query_tuning.py --profiles baseline,both --ids 2,3,11
  uv run python scripts/compare_query_tuning.py --with-answer --ids 2
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

from lightrag import QueryParam  # noqa: E402
from client_paths import get_parser_output_dir, get_rag_storage_dir  # noqa: E402
from query_clarification import assess_query_fit, load_clarify_config  # noqa: E402
from query_progress_hooks import (  # noqa: E402
    finalize_related_images,
    get_query_debug_state,
    query_progress_hooks,
    set_query_media_roots,
)
from typing import Any

from query_tuning_profiles import (  # noqa: E402
    PROFILES,
    profile_names,
    restore_env,
    snapshot_env,
    sync_lightrag_rerank_threshold,
)


_CAPTION_HINT_RE = re.compile(
    r'[「"\u201c\u2018]([^」"\u201d\u2019]+)[」"\u201d\u2019]'
)


@dataclass
class TestCase:
    case_id: int
    query: str
    expect_images: bool | None = None
    caption_hints: list[str] = field(default_factory=list)
    note: str = ""


def parse_test_file(path: Path) -> list[TestCase]:
    if not path.is_file():
        return []
    cases: list[TestCase] = []
    current: TestCase | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("多轮问答") or line.startswith("测 "):
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if m:
            if current is not None:
                cases.append(current)
            current = TestCase(case_id=int(m.group(1)), query=m.group(2).strip())
            continue
        if current is None:
            continue
        if line.startswith("是否配图"):
            m_img = re.match(r"是否配图[：:]\s*(是|否)", line)
            if m_img:
                current.expect_images = m_img.group(1) == "是"
            elif "可有可无" in line:
                current.expect_images = None
            for hint in _CAPTION_HINT_RE.findall(line):
                if hint.strip():
                    current.caption_hints.append(hint.strip())
            continue
        if line.startswith("*") and current.note == "":
            current.note = line.lstrip("*").strip()
    if current is not None:
        cases.append(current)
    return cases


def _caption_match(hints: list[str], captions: list[str]) -> bool | None:
    if not hints:
        return None
    blob = " ".join(captions)
    return any(h in blob for h in hints)


def _image_expect_ok(case: TestCase, image_count: int, captions: list[str]) -> bool | None:
    if case.expect_images is None:
        return None
    if not case.expect_images:
        return image_count == 0
    if image_count == 0:
        return False
    match = _caption_match(case.caption_hints, captions)
    return True if match is None else match


async def _run_one(
    rag: Any,
    *,
    query: str,
    mode: str,
    parser_root: Path,
    with_answer: bool,
) -> dict[str, object]:
    from raganything.pipeline_rerank import query_param_from_env  # noqa: E402

    param = query_param_from_env(mode=mode)
    query_kwargs: dict[str, Any] = {
        "enable_rerank": param.enable_rerank,
        "top_k": param.top_k,
        "chunk_top_k": param.chunk_top_k,
        "max_entity_tokens": param.max_entity_tokens,
        "max_relation_tokens": param.max_relation_tokens,
        "max_total_tokens": param.max_total_tokens,
        "vlm_enhanced": False,
    }
    if not with_answer:
        query_kwargs["only_need_context"] = True

    set_query_media_roots([parser_root])
    async with query_progress_hooks():
        if with_answer:
            answer = await rag.aquery(query, mode=mode, **query_kwargs)
        else:
            await rag.aquery(query, mode=mode, **query_kwargs)
            answer = ""
        images = finalize_related_images()
        state = get_query_debug_state()

    docs = state.get("retrieved_docs") or []
    primary = (state.get("retrieved_docs_text") or "").strip()
    images_debug = state.get("images_debug") or {}
    anchor_scan = images_debug.get("anchor_scan") or {}
    captions = [str(img.get("caption") or "") for img in images]

    clarify = await assess_query_fit(rag, query, mode=mode)
    max_rerank = None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        raw = doc.get("rerank_score")
        if raw is None:
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        max_rerank = score if max_rerank is None else max(max_rerank, score)

    return {
        "chunk_count": len(docs),
        "primary_chars": len(primary),
        "anchor_chars": anchor_scan.get("anchor_chars", 0),
        "anchor_sections": anchor_scan.get("anchor_sections", []),
        "image_count": len(images),
        "captions": captions,
        "clarify_band": clarify.band,
        "clarify_score": round(clarify.score, 4),
        "max_rerank": max_rerank,
        "gate_ok": bool((images_debug.get("gate") or {}).get("ok")),
        "answer_preview": (answer or "")[:240],
    }


async def _async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-w", "--working-dir", type=Path, default=None)
    p.add_argument("--parser-output-dir", type=Path, default=None)
    p.add_argument(
        "--test-file",
        type=Path,
        default=_ROOT / "docs" / "测试问题.txt",
    )
    p.add_argument(
        "--profiles",
        type=str,
        default=",".join(profile_names()),
        help=f"Comma-separated profile names (default: all). Available: {', '.join(profile_names())}",
    )
    p.add_argument("--ids", type=str, default="", help="Comma-separated case ids, e.g. 2,3,11")
    p.add_argument("--mode", default=os.getenv("RAG_QUERY_MODE", "mix"))
    p.add_argument(
        "--with-answer",
        action="store_true",
        help="Also call answer LLM (slow; default is context-only).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report (default: logs/compare_tuning/TIMESTAMP.json).",
    )
    args = p.parse_args()

    profile_list = [x.strip() for x in args.profiles.split(",") if x.strip()]
    for name in profile_list:
        if name not in PROFILES:
            raise SystemExit(f"Unknown profile {name!r}; choose from {profile_names()}")

    cases = parse_test_file(args.test_file.expanduser().resolve())
    if args.ids.strip():
        wanted = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c.case_id in wanted]
    if not cases:
        raise SystemExit("No test cases selected.")

    wd = (args.working_dir or get_rag_storage_dir()).expanduser().resolve()
    pod = (args.parser_output_dir or get_parser_output_dir()).expanduser().resolve()
    rag, _, _ = await rpc._build_rag(wd, pod)

    clarify_cfg = load_clarify_config()
    saved_base = snapshot_env()
    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "working_dir": str(wd),
        "parser_output_dir": str(pod),
        "mode": args.mode.strip(),
        "clarify_thresholds": {
            "lower": clarify_cfg.threshold_lower,
            "upper": clarify_cfg.threshold_upper,
        },
        "profiles": profile_list,
        "results": [],
    }

    try:
        for profile in profile_list:
            restore_env(saved_base)
            for key, value in PROFILES[profile].items():
                os.environ[key] = value
            rerank_thr = sync_lightrag_rerank_threshold(rag.lightrag)
            print(f"\n=== profile: {profile} (min_rerank_score={rerank_thr}) ===")
            for case in cases:
                row = await _run_one(
                    rag,
                    query=case.query,
                    mode=args.mode.strip(),
                    parser_root=pod,
                    with_answer=args.with_answer,
                )
                expect_ok = _image_expect_ok(case, int(row["image_count"]), list(row["captions"]))
                entry = {
                    "profile": profile,
                    "case_id": case.case_id,
                    "query": case.query,
                    "expect_images": case.expect_images,
                    "caption_hints": case.caption_hints,
                    "image_expect_ok": expect_ok,
                    "top_k": int(os.getenv("TOP_K", "20")),
                    "chunk_top_k": int(os.getenv("CHUNK_TOP_K", "30")),
                    "max_total_tokens": int(os.getenv("MAX_TOTAL_TOKENS", "52000")),
                    **row,
                }
                report["results"].append(entry)
                ok_mark = "?" if expect_ok is None else ("OK" if expect_ok else "FAIL")
                print(
                    f"  [{case.case_id:02d}] imgs={row['image_count']} "
                    f"chunks={row['chunk_count']} primary={row['primary_chars']} "
                    f"anchor={row['anchor_chars']} clarify={row['clarify_band']} "
                    f"({row['clarify_score']}) expect={ok_mark}"
                )
                if row["captions"]:
                    print(f"       captions: {row['captions'][:3]}")
    finally:
        restore_env(saved_base)
        await rag.finalize_storages()

    out = args.out
    if out is None:
        out_dir = _ROOT / "logs" / "compare_tuning"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"{stamp}.json"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report: {out.as_posix()}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
