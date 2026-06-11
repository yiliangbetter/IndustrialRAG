#!/usr/bin/env python3
"""Web-path batch test Q1–Q13 graded against docs/测试例参考答案.md key points."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env", override=False)

# Key checks from docs/测试例参考答案.md (Q1–Q13)
REF: dict[int, dict] = {
    1: {
        "query": "高速智能封边机维护保养手册适用于哪些产品型号？",
        "text_all": ["NB9-Smart", "NB10-Smart"],
        "text_any": [],
        "want_images": False,
        "caption_any": [],
    },
    2: {
        "query": "高速智能封边机开机前我要如何检查电源开关？",
        "text_all": ["外观", "作用", "接地"],
        "text_any": [],
        "want_images": False,
        "caption_any": [],
    },
    3: {
        "query": "高速智能封边机机床床身清洁，要多长时间做一次？",
        "text_all": ["每天"],
        "text_any": [],
        "want_images": True,
        "caption_any": ["机床外部清洁"],
    },
    4: {
        "query": "高速智能封边机机床内部进行清洁，请问步骤是什么？",
        "text_all": [],
        "text_any": [],
        "text_need2": [["吸尘机", "碎布"], ["刮削", "灰尘"], ["油污"]],
        "want_images": True,
        "caption_any": ["清洁机床内部"],
    },
    5: {
        "query": "高速智能封边机清理压带轮残胶应该使用什么工具？",
        "text_all": ["刮刀"],
        "text_any": [],
        "want_images": True,
        "caption_any": ["压带轮残胶清理"],
    },
    6: {
        "query": "高速智能封边机的输送链条进行保养我要加注什么？",
        "text_all": [],
        "text_any": ["润滑脂2#", "润滑脂 2#", "润滑脂2＃"],
        "want_images": True,
        "caption_any": ["输送链条加润滑脂"],
    },
    7: {
        "query": "对高速智能封边机的进料部分保养时，需要使用什么表？表针读数需要小于多少？",
        "text_all": ["百分表"],
        "text_any": ["0.15"],
        "want_images": True,
        "caption_any": ["进料部分保养"],
    },
    8: {
        "query": "高速智能封边机更换预铣刀时，刀刃的装配方向应该如何？",
        "text_all": [],
        "text_any": ["顺铣", "逆铣"],
        "text_need2": [["顺铣"], ["逆铣"]],
        "want_images": True,
        "caption_any": ["检查预铣刀磨损情况", "预铣刀"],
    },
    9: {
        "query": "检查高速智能封边机的注油泵时，如果油位过低，需要注入哪个品牌的哪种液压油产品？",
        "text_all": [],
        "text_any": ["美孚"],
        "text_need2": [["长效液压油"], ["美孚"]],
        "want_images": True,
        "caption_any": ["检查注油泵油量", "注油泵"],
    },
    10: {
        "query": "高速智能封边机的保养中，哪些部件需要使用美孚长效液压油？",
        "text_all": [
            "自动注油泵",
            "辅助进料导轨",
            "进料靠板",
            "预铣机构导轨",
            "平切机构导轨",
            "精修导轨",
            "仿形机构导轨",
            "开槽机构丝杆",
            "刮边机构导轨",
        ],
        "text_any": [],
        "want_images": False,
        "caption_any": [],
    },
    11: {
        "query": "高速智能封边机的保养中，哪些部件需要清理残胶",
        "text_all": ["压带轮", "仿形靠板", "涂胶轴"],
        "text_any": [],
        "want_images": True,
        "caption_any": [
            "压带轮残胶清理",
            "仿形靠板上残胶清理",
            "清理胶轴老化胶水",
        ],
        "caption_min_match": 1,
    },
    12: {
        "query": "高速智能封边机的保养中，需要清理粉尘，碎屑的部件有哪些？",
        "text_all": [],
        "text_any": [],
        "text_min_len": 200,
        "want_images": True,
        "caption_min": 1,
    },
    13: {
        "query": "我将为高速智能封边机进行季度保养，请问我需要准备哪几种润滑脂？",
        "text_all": [],
        "text_any": [],
        "text_need2": [["润滑脂2#", "润滑脂 2#"], ["高温润滑脂"]],
        "want_images": True,
        "caption_min": 1,
    },
}


def _load_rpc():
    spec = importlib.util.spec_from_file_location(
        "rpc", _SCRIPTS / "rag_pipeline_parse_graph_chat.py"
    )
    rpc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rpc)
    return rpc


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or ""))


def grade_text(answer: str, spec: dict) -> tuple[bool, list[str]]:
    misses: list[str] = []
    a = answer or ""
    for t in spec.get("text_all") or []:
        if t not in a:
            misses.append(f"missing:{t}")
    for t in spec.get("text_any") or []:
        if t not in a and _norm(t) not in _norm(a):
            pass  # optional single any
    if spec.get("text_any") and not any(
        t in a or _norm(t) in _norm(a) for t in spec["text_any"]
    ):
        if not spec.get("text_need2"):
            misses.append(f"any_of:{spec['text_any']}")
    for group in spec.get("text_need2") or []:
        if not any(g in a or _norm(g) in _norm(a) for g in group):
            misses.append(f"need_one_of:{group}")
    if spec.get("text_min_len") and len(a.strip()) < int(spec["text_min_len"]):
        misses.append(f"too_short:{len(a.strip())}<{spec['text_min_len']}")
    ok = len(misses) == 0
    return ok, misses


def grade_images(result: dict, spec: dict) -> tuple[bool, list[str]]:
    imgs = result.get("images") or []
    pls = result.get("placements") or []
    caps = [str(i.get("caption") or "") for i in imgs]
    want = bool(spec.get("want_images"))
    notes: list[str] = []

    if not want:
        if imgs:
            notes.append(f"unexpected_images:{len(imgs)}")
        return len(imgs) == 0, notes

    if not pls:
        notes.append("no_placements")
        return False, notes

    expected = spec.get("caption_any") or []
    if expected:
        matched = [
            e
            for e in expected
            if any(e in c or c in e for c in caps)
        ]
        min_match = int(spec.get("caption_min_match") or 1)
        if len(matched) < min_match:
            notes.append(f"caption_match:{len(matched)}/{len(expected)}")
        wrong = [
            c
            for c in caps
            if c and not any(e in c or c in e for e in expected)
        ]
        if wrong and expected:
            notes.append(f"extra_or_wrong_caps:{wrong}")
        caps_ok = all(
            any(e in c or c in e for e in expected) for c in caps if c
        ) if caps else False
        if caps and not caps_ok:
            notes.append("caption_not_in_expected")
        ok = len(matched) >= min_match and (not caps or caps_ok)
        return ok, notes

    min_c = int(spec.get("caption_min") or 1)
    if len(imgs) < min_c:
        notes.append(f"image_count:{len(imgs)}<{min_c}")
        return False, notes
    return True, notes


async def run_cases(ids: list[int], *, mode: str, wd: Path, pod: Path) -> list[dict]:
    from query_doc_steering import strip_manual_circled_step_markers
    from query_progress_hooks import (
        finalize_inline_images,
        query_progress_hooks,
        set_query_media_roots,
        set_query_text_for_images,
    )
    from stream_cot_parser import parse_complete_cot

    rpc = _load_rpc()
    rag, _, _ = await rpc._build_rag(wd, pod)
    parser_root = pod.resolve()
    out: list[dict] = []

    try:
        for cid in ids:
            spec = REF[cid]
            query = spec["query"]
            t0 = time.perf_counter()
            async with query_progress_hooks():
                set_query_media_roots([parser_root])
                set_query_text_for_images(query.strip())
                raw = await rag.aquery(
                    query,
                    mode=mode,
                    vlm_enhanced=False,
                    **rpc._query_extras_from_env(query),
                )
                thinking, answer = parse_complete_cot(raw or "")
                answer = strip_manual_circled_step_markers(answer.strip())
                inline = finalize_inline_images(answer_text=answer)
            elapsed = int((time.perf_counter() - t0) * 1000)
            row = {
                "id": cid,
                "query": query,
                "answer": answer,
                "duration_ms": elapsed,
                "images": inline.get("images") or [],
                "placements": inline.get("placements") or [],
                "debug": inline.get("debug") or {},
            }
            text_ok, text_miss = grade_text(answer, spec)
            img_ok, img_notes = grade_images(row, spec)
            row["grade"] = {
                "text_ok": text_ok,
                "image_ok": img_ok,
                "ok": text_ok and img_ok,
                "text_miss": text_miss,
                "image_notes": img_notes,
            }
            out.append(row)
            g = row["grade"]
            print(
                f"Q{cid:02d} {'PASS' if g['ok'] else 'FAIL'} "
                f"text={g['text_ok']} img={g['image_ok']} "
                f"{elapsed}ms imgs={len(row['images'])} pl={len(row['placements'])}"
            )
            if not g["text_ok"]:
                print(f"  text: {g['text_miss']}")
            if not g["image_ok"]:
                print(f"  img: {g['image_notes']}")
            for c in [i.get("caption") for i in row["images"]]:
                print(f"  caption: {c}")
    finally:
        await rag.finalize_storages()
    return out


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Web 路径批测 Q1–Q13",
        "",
        f"- 时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 模式：mix",
        f"- 通过：{sum(1 for r in rows if r['grade']['ok'])}/{len(rows)}",
        "",
    ]
    for r in rows:
        g = r["grade"]
        lines.append(f"## Q{r['id']} {'✓' if g['ok'] else '✗'}")
        lines.append("")
        lines.append(f"*耗时 {r['duration_ms']} ms · 文字 {g['text_ok']} · 配图 {g['image_ok']}*")
        if g["text_miss"]:
            lines.append(f"- 文字缺失：{', '.join(g['text_miss'])}")
        if g["image_notes"]:
            lines.append(f"- 配图问题：{', '.join(g['image_notes'])}")
        caps = [i.get("caption") for i in r.get("images") or []]
        if caps:
            lines.append(f"- 图注：{'; '.join(str(c) for c in caps)}")
        gate = ((r.get("debug") or {}).get("gate") or {}).get("reason")
        if gate:
            lines.append(f"- gate：{gate}")
        lines.append("")
        lines.append(r.get("answer") or "（空）")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ids = list(range(1, 14))
    wd = Path(os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    mode = os.getenv("RAG_QUERY_MODE", "mix")
    print(f"Running Q1–Q13 web path, wd={wd}")
    rows = asyncio.run(run_cases(ids, mode=mode, wd=wd, pod=pod))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = _ROOT / "logs" / "web_path_q1_13"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{stamp}.json"
    md_path = report_dir / f"{stamp}.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, rows)
    passed = sum(1 for r in rows if r["grade"]["ok"])
    print(f"\nReport: {md_path}")
    print(f"Total: {passed}/{len(rows)} passed")
    if passed < len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
