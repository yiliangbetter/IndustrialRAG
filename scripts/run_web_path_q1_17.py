#!/usr/bin/env python3
"""Web-path batch test Q1–Q17 graded against docs/测试例参考答案.md key points."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env", override=False)

# Key checks from docs/测试例参考答案.md (Q1–Q17)
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
        "text_any": [],
        "text_need2": [["顺铣"], ["逆铣"]],
        "want_images": True,
        "caption_any": ["检查预铣刀磨损情况", "预铣刀"],
    },
    9: {
        "query": "检查高速智能封边机的注油泵时，如果油位过低，需要注入哪个品牌的哪种液压油产品？",
        "text_all": [],
        "text_any": [],
        "text_need2": [["美孚"], ["长效液压油"]],
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
            "电机检查清理",
        ],
        "caption_min_match": 3,
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
        "text_need2": [["润滑脂2#", "润滑脂 2#", "润滑脂2＃"], ["高温润滑脂"]],
        "want_images": True,
        "caption_min": 1,
        "caption_topic_any": ["润滑脂", "润滑", "涂胶轴", "导轨", "齿条"],
    },
    14: {
        "query": "南兴的封边机一共有多少产品型号？",
        "text_all": [],
        "text_any": [],
        "text_need2": [
            ["NB9-Smart", "NB10-Smart"],
            ["双端", "NB6S", "NB7H", "NB8C"],
            ["高速自动", "NB6P", "NB7P", "NB8P"],
            ["自动封边", "NBC", "NB5J", "NB6J", "NB7CJ"],
        ],
        "text_min_len": 80,
        "want_images": False,
        "caption_any": [],
    },
    15: {
        "query": "这四种封边机的电控板的保养周期分别是多久",
        "text_all": [],
        "text_any": [],
        "text_need2": [
            ["高速智能", "季度", "每季"],
            ["自动封边", "半年", "每半年"],
            ["双端", "半年", "每半年"],
            ["高速自动", "半年", "每半年"],
        ],
        "want_images": True,
        "caption_min": 4,
        "image_min_sources": 4,
        "caption_topic_any": ["电控"],
    },
    16: {
        "query": "1#透平油（气动油）是哪些机型的保养所需要的，是用来保养哪个部件？",
        "text_all": [],
        "text_any": [],
        "text_need2": [
            ["三联件", "油雾器", "油杯"],
            ["双端"],
            ["自动封边"],
            ["高速自动"],
        ],
        "want_images": False,
        "caption_any": [],
    },
    17: {
        "query": "所有机型中，各自哪些部件需要清除残胶",
        "text_all": [],
        "text_any": [],
        "text_need2": [
            ["高速智能"],
            ["高速自动", "自动封边", "双端"],
            ["压带轮"],
            ["仿形靠", "仿形靠模", "仿形靠板"],
        ],
        "text_min_len": 120,
        "want_images": True,
        "image_answer_pairs": True,
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


def _contains(term: str, text: str) -> bool:
    return term in text or _norm(term) in _norm(text)


def grade_text(answer: str, spec: dict) -> tuple[bool, list[str]]:
    misses: list[str] = []
    a = answer or ""
    for t in spec.get("text_all") or []:
        if not _contains(t, a):
            misses.append(f"missing:{t}")
    if spec.get("text_any") and not any(_contains(t, a) for t in spec["text_any"]):
        if not spec.get("text_need2"):
            misses.append(f"any_of:{spec['text_any']}")
    for group in spec.get("text_need2") or []:
        if not any(_contains(g, a) for g in group):
            misses.append(f"need_one_of:{group}")
    if spec.get("text_count_any") and not any(
        _contains(t, a) for t in spec["text_count_any"]
    ):
        misses.append(f"count_any:{spec['text_count_any']}")
    for bad in spec.get("text_none") or []:
        if _contains(bad, a):
            misses.append(f"forbidden:{bad}")
    if spec.get("text_min_len") and len(a.strip()) < int(spec["text_min_len"]):
        misses.append(f"too_short:{len(a.strip())}<{spec['text_min_len']}")
    ok = len(misses) == 0
    return ok, misses


def _image_matches_machine(img: dict, machine_hint: str) -> bool:
    from image_query_refs import _doc_matches_manual_hint  # noqa: WPS433

    blob = " ".join(
        str(img.get(key) or "")
        for key in ("source_key", "caption", "context", "url", "path")
    )
    if not blob.strip():
        return False
    probe = {"file_path": blob}
    if _doc_matches_manual_hint(probe, machine_hint):
        return True
    hint = machine_hint.strip()
    for token in (hint, hint.replace("封边机", "")):
        token = token.strip()
        if len(token) >= 3 and token in blob:
            return True
    return False


def _image_matches_component(img: dict, component: str) -> bool:
    from image_query_refs import (  # noqa: WPS433
        _label_matches_listing_target,
        _listing_target_head,
    )

    head = _listing_target_head(component)
    cap = str(img.get("caption") or "")
    ctx = str(img.get("context") or "")
    blob = f"{cap} {ctx}".strip()
    if not blob:
        return False
    if _label_matches_listing_target(cap, head) or _label_matches_listing_target(
        ctx, head
    ):
        return True
    return head in blob or component in blob


def _grade_images_answer_pairs(answer: str, imgs: list[dict]) -> tuple[bool, list[str]]:
    from image_query_refs import _machine_component_targets_from_answer  # noqa: WPS433

    notes: list[str] = []
    if not imgs:
        notes.append("no_images")
        return False, notes
    pairs = _machine_component_targets_from_answer(answer)
    if len(pairs) < 2:
        notes.append(f"answer_pairs:{len(pairs)}<2")
        return False, notes
    uncovered: list[str] = []
    used: set[int] = set()
    for machine, component in pairs:
        matched_idx: int | None = None
        for idx, img in enumerate(imgs):
            if idx in used:
                continue
            if _image_matches_machine(img, machine) and _image_matches_component(
                img, component
            ):
                matched_idx = idx
                break
        if matched_idx is None:
            uncovered.append(f"{machine}/{component}")
        else:
            used.add(matched_idx)
    if uncovered:
        notes.append(f"pair_missing:{uncovered}")
    if len(imgs) < len(pairs):
        notes.append(f"image_count:{len(imgs)}<{len(pairs)}")
    ok = not uncovered and len(imgs) >= len(pairs)
    return ok, notes


def grade_images(result: dict, spec: dict) -> tuple[bool, list[str]]:
    imgs = result.get("images") or []
    pls = result.get("placements") or []
    caps = [str(i.get("caption") or "") for i in imgs]
    want = spec.get("want_images")
    notes: list[str] = []

    if want is False:
        if imgs:
            notes.append(f"unexpected_images:{len(imgs)}")
        return len(imgs) == 0, notes

    if want is not True:
        return True, notes

    if spec.get("image_answer_pairs"):
        return _grade_images_answer_pairs(str(result.get("answer") or ""), imgs)

    if not pls:
        notes.append("no_placements")
        return False, notes

    expected = spec.get("caption_any") or []
    if expected:
        matched = [e for e in expected if any(e in c or c in e for c in caps)]
        min_match = int(spec.get("caption_min_match") or 1)
        if len(matched) < min_match:
            notes.append(f"caption_match:{len(matched)}/{len(expected)}")
        wrong = [
            c for c in caps if c and not any(e in c or c in e for e in expected)
        ]
        if wrong and expected:
            notes.append(f"extra_or_wrong_caps:{wrong}")
        caps_ok = (
            all(any(e in c or c in e for e in expected) for c in caps if c)
            if caps
            else False
        )
        if caps and not caps_ok:
            notes.append("caption_not_in_expected")
        ok = len(matched) >= min_match and (not caps or caps_ok)
        return ok, notes

    min_c = int(spec.get("caption_min") or spec.get("image_min") or 1)
    if len(imgs) < min_c:
        notes.append(f"image_count:{len(imgs)}<{min_c}")
        return False, notes

    topic_any = spec.get("caption_topic_any") or []
    if topic_any:
        topic_hits = sum(
            1
            for img in imgs
            if any(
                t in str(img.get("caption") or "")
                or t in str(img.get("context") or "")
                for t in topic_any
            )
        )
        if topic_hits < min(min_c, len(imgs)):
            notes.append(f"caption_topic:{topic_hits}/{min_c}")

    min_sources = int(spec.get("image_min_sources") or 0)
    if min_sources > 0:
        sources = {
            str(img.get("source_key") or "").strip()
            for img in imgs
            if str(img.get("source_key") or "").strip()
        }
        if len(sources) < min_sources:
            notes.append(f"image_sources:{len(sources)}<{min_sources}")
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
                f"{elapsed}ms imgs={len(row['images'])} pl={len(row['placements'])}",
                flush=True,
            )
            if not g["text_ok"]:
                print(f"  text: {g['text_miss']}", flush=True)
            if not g["image_ok"]:
                print(f"  img: {g['image_notes']}", flush=True)
            for c in [i.get("caption") for i in row["images"]]:
                print(f"  caption: {c}", flush=True)
    finally:
        await rag.finalize_storages()
    return out


def _resolve_image_local_path(
    img: dict, debug: dict, media_root: Path, *, index: int
) -> Path | None:
    """Resolve on-disk image path for markdown embedding in batch reports."""
    paths = (debug or {}).get("selected_paths") or []
    if index < len(paths):
        candidate = Path(str(paths[index]))
        if candidate.is_file():
            return candidate.resolve()

    url = str(img.get("url") or "")
    if url:
        token = parse_qs(urlparse(url).query).get("token", [""])[0]
        if token:
            from image_query_refs import decode_media_token  # noqa: WPS433

            decoded = decode_media_token(token, media_root)
            if decoded is not None and decoded.is_file():
                return decoded.resolve()
    return None


def _md_image_relpath(report_path: Path, image_path: Path) -> str:
    rel = os.path.relpath(image_path.resolve(), report_path.parent.resolve())
    return Path(rel).as_posix()


def _expected_image_hint(spec: dict) -> str:
    if spec.get("want_images") is False:
        return "不应配图"
    if spec.get("want_images") is not True:
        return "—"
    if spec.get("image_answer_pairs"):
        return "期望：答案每个 (机型, 部件) 各 1 张配图，图数 ≥ 答案部件条数"
    caps = spec.get("caption_any") or []
    if caps:
        min_match = int(spec.get("caption_min_match") or 1)
        return f"期望图注（≥{min_match} 命中）：{'; '.join(caps)}"
    min_c = int(spec.get("caption_min") or spec.get("image_min") or 1)
    topics = spec.get("caption_topic_any") or []
    if topics:
        return f"期望 ≥{min_c} 张，图注/上下文含：{', '.join(topics)}"
    return f"期望 ≥{min_c} 张配图"


def write_report(
    path: Path,
    rows: list[dict],
    *,
    mode: str,
    wd: Path,
    media_root: Path,
) -> None:
    passed = sum(1 for r in rows if r["grade"]["ok"])
    text_only = sum(1 for r in rows if r["grade"]["text_ok"])
    ids_label = ", ".join(f"Q{r['id']}" for r in rows)
    lines = [
        f"# Web 路径批测（{ids_label}）",
        "",
        f"- 时间：{datetime.now().astimezone().isoformat()}",
        f"- 模式：{mode}",
        f"- 工作目录：`{wd}`",
        f"- 媒体根：`{media_root}`",
        f"- 参考答案：`docs/测试例参考答案.md`",
        f"- 通过（文字+配图）：**{passed}/{len(rows)}**",
        f"- 仅文字通过：**{text_only}/{len(rows)}**",
        "",
        "## 汇总",
        "",
        "| ID | 结果 | 文字 | 配图 | 耗时(ms) | 图数 | 图注摘要 |",
        "|----|------|------|------|----------|------|----------|",
    ]
    for r in rows:
        g = r["grade"]
        caps = [str(i.get("caption") or "") for i in r.get("images") or []]
        cap_summary = "; ".join(c for c in caps if c) or "—"
        if len(cap_summary) > 48:
            cap_summary = cap_summary[:45] + "…"
        lines.append(
            f"| Q{r['id']} | {'✓' if g['ok'] else '✗'} | "
            f"{'✓' if g['text_ok'] else '✗'} | "
            f"{'✓' if g['image_ok'] else '✗'} | "
            f"{r['duration_ms']} | {len(r.get('images') or [])} | {cap_summary} |"
        )
    lines.append("")
    for r in rows:
        g = r["grade"]
        spec = REF.get(r["id"], {})
        debug = r.get("debug") or {}
        lines.append(f"## Q{r['id']} {'✓' if g['ok'] else '✗'}")
        lines.append("")
        lines.append(f"**问题**：{r['query']}")
        lines.append("")
        lines.append(
            f"*耗时 {r['duration_ms']} ms · 文字 {g['text_ok']} · 配图 {g['image_ok']}*"
        )
        lines.append(f"- 参考答案要点：{_expected_image_hint(spec)}")
        if g["text_miss"]:
            lines.append(f"- 文字缺失：{', '.join(g['text_miss'])}")
        if g["image_notes"]:
            lines.append(f"- 配图问题：{', '.join(g['image_notes'])}")
        anchor = debug.get("anchor_scan") or {}
        if anchor:
            lines.append(
                f"- 扫描模式：`{anchor.get('mode')}`"
                + (
                    f" · chunks={anchor.get('anchor_chunks')}"
                    if anchor.get("anchor_chunks") is not None
                    else ""
                )
                + (f" · {anchor.get('reason')}" if anchor.get("reason") else "")
            )
        gate = (debug.get("gate") or {}).get("reason")
        if gate:
            lines.append(f"- gate：{gate}")
        lines.append("")
        lines.append("**答案**")
        lines.append("")
        lines.append(r.get("answer") or "（空）")
        lines.append("")

        imgs = r.get("images") or []
        pls = r.get("placements") or []
        lines.append("**配图**")
        lines.append("")
        if not imgs:
            lines.append("（无配图）")
        else:
            for idx, img in enumerate(imgs):
                cap = str(img.get("caption") or "（无图注）")
                page = img.get("page")
                src = str(img.get("source_key") or "")
                meta = f"**{cap}**"
                if page is not None:
                    meta += f" · p{page}"
                if src:
                    meta += f" · {src}"
                placement = next(
                    (p for p in pls if int(p.get("image_index") or -1) == idx),
                    None,
                )
                if placement and placement.get("anchor_text"):
                    meta += f" · 锚点「{placement['anchor_text']}」"
                lines.append(f"{idx + 1}. {meta}")
                local = _resolve_image_local_path(img, debug, media_root, index=idx)
                if local is not None:
                    rel = _md_image_relpath(path, local)
                    lines.append("")
                    lines.append(f"![{cap}]({rel})")
                else:
                    lines.append("")
                    lines.append(f"- 本地路径未解析：`{img.get('url') or ''}`")
                ctx = str(img.get("context") or "").strip()
                if ctx:
                    preview = ctx if len(ctx) <= 200 else ctx[:197] + "…"
                    lines.append("")
                    lines.append(f"> {preview}")
                lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_ids(raw: str | None) -> list[int]:
    if not raw or not raw.strip():
        return list(range(1, 18))
    out: list[int] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        cid = int(part.lstrip("Qq"))
        if cid not in REF:
            raise SystemExit(f"Unknown case id Q{cid}; valid: Q1–Q17")
        out.append(cid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-path batch test vs 测试例参考答案")
    parser.add_argument(
        "--ids",
        default=os.getenv("RAG_WEB_PATH_CASE_IDS", ""),
        help="Comma-separated case ids, e.g. 2,3,8,17 or Q2,Q3 (default: all Q1–Q17)",
    )
    args = parser.parse_args()
    ids = _parse_ids(args.ids)
    wd = Path(os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    mode = os.getenv("RAG_QUERY_MODE", "mix")
    label = ", ".join(f"Q{i}" for i in ids)
    print(f"Running {label} web path, wd={wd}", flush=True)
    rows = asyncio.run(run_cases(ids, mode=mode, wd=wd, pod=pod))
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report_dir = _ROOT / "logs" / "web_path_q1_17"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"q{'_'.join(str(i) for i in ids)}" if len(ids) < 17 else "all"
    json_path = report_dir / f"{stamp}_{suffix}.json"
    md_path = report_dir / f"{stamp}_{suffix}.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, rows, mode=mode, wd=wd, media_root=pod)
    passed = sum(1 for r in rows if r["grade"]["ok"])
    print(f"\nReport: {md_path}", flush=True)
    print(f"Total: {passed}/{len(rows)} passed", flush=True)
    if passed < len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
