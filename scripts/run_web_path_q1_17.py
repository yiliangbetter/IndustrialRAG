#!/usr/bin/env python3
"""Web-path batch test Q1–Q17 (shili17) graded against docs/测试例参考答案.md.

Each case runs a clarify-gate probe (``probe_llm_retrieval_full``) to record
``final_score``. Shili17 expects **direct** gate (``final_score > CLARIFY_DIRECT_RERANK_MIN``,
default 7) on every question; sub-threshold scores are flagged prominently.

Then the full web ``aquery`` answer path runs. By default (``--web-sim direct``) the
answer reuses the gate probe's cached bundle when ``final_score > CLARIFY_DIRECT_RERANK_MIN``,
matching Web ``ClarifyBypass(direct)``. Use ``--web-sim keep_original`` to reuse the probe
bundle whenever the probe is answerable (simulates clicking「继续原问题」). ``--web-sim none``
keeps the old independent second retrieval for A/B comparison.

Query dumps go to ``logs/query_dumps/`` (prefix ``Qxx_``) unless ``--no-dump``.

Use ``--rounds N`` to run the full batch N times; each round writes its own report
(``…_r01.md``, ``…_r02.md``, … when N > 1). Multiple rounds reuse one ``_build_rag``
instance for the whole run (avoids stacking embedding models on GPU/RAM).
"""

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
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env", override=False)

# Key checks from docs/测试例参考答案.md (Q1–Q17)
_TEST_CASES_PATH = _ROOT / "tests" / "fixtures" / "test_cases_shili17.json"


def _load_test_cases(path: Path = _TEST_CASES_PATH) -> dict[int, dict]:
    """Load shili17 cases externalized to ``tests/fixtures/test_cases_shili17.json``.

    Phase-1 structural rework: cases live outside the script so the harness
    stays generic. Each case's ``text`` and ``image`` criteria are flattened
    back into one spec dict (plus ``query``) — the exact shape the former
    inline ``REF`` provided — so ``grade_text`` / ``grade_images`` are
    unchanged. The split is deliberate: when the image-matching logic is
    reworked (PR47 §4 A/B), only the ``image`` criteria + ``grade_images``
    change; text criteria and the harness stay stable.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: dict[int, dict] = {}
    for case in data["cases"]:
        spec: dict = {"query": case["query"]}
        spec.update(case.get("text") or {})
        spec.update(case.get("image") or {})
        cases[int(case["id"])] = spec
    return cases


REF: dict[int, dict] = _load_test_cases()


def _format_duration(seconds: float) -> str:
    """Human-readable duration for reports (minutes + seconds)."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes}m {remainder:.1f}s"
    return f"{seconds:.1f}s"


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


def _answer_bullets(answer: str) -> list[str]:
    bullets: list[str] = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("* "):
            bullets.append(line[2:].strip())
        elif line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets or [answer or ""]


def _grade_machine_cycle_pairs(answer: str, pairs: list[dict]) -> list[str]:
    """Require each machine mention to co-occur with the right cycle in the same bullet."""
    misses: list[str] = []
    bullets = _answer_bullets(answer)
    full = answer or ""
    for pair in pairs:
        machine_terms = pair.get("machine") or []
        cycle_any = pair.get("cycle_any") or []
        cycle_forbidden = pair.get("cycle_forbidden") or []
        matched = [b for b in bullets if any(_contains(m, b) for m in machine_terms)]
        if not matched:
            if any(_contains(m, full) for m in machine_terms):
                matched = [full]
            else:
                misses.append(f"machine_missing:{machine_terms}")
                continue
        has_cycle = any(
            any(_contains(c, line) for c in cycle_any) for line in matched
        )
        has_bad = any(
            any(_contains(c, line) for c in cycle_forbidden) for line in matched
        )
        if not has_cycle or has_bad:
            misses.append(
                f"machine_cycle:{machine_terms} need {cycle_any}"
                + (f" forbid {cycle_forbidden}" if cycle_forbidden else "")
            )
    return misses


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
    misses.extend(_grade_machine_cycle_pairs(a, spec.get("text_machine_cycles") or []))
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


def _grade_images_answer_pairs(
    answer: str,
    imgs: list[dict],
    *,
    pair_waive: list[tuple[str, str]] | None = None,
) -> tuple[bool, list[str]]:
    from image_query_refs import (  # noqa: WPS433
        _listing_target_head,
        _machine_component_targets_from_answer,
        _normalize_label_key,
    )

    notes: list[str] = []
    if not imgs:
        notes.append("no_images")
        return False, notes
    waived_keys: set[tuple[str, str]] = set()
    for machine, component in pair_waive or []:
        machine = (machine or "").strip()
        component = _listing_target_head((component or "").strip())
        if machine and component:
            waived_keys.add(
                (_normalize_label_key(machine), _normalize_label_key(component))
            )
    raw_pairs = _machine_component_targets_from_answer(answer)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for machine, component in raw_pairs:
        machine = machine.strip()
        component = _listing_target_head(component)
        if not machine or not component:
            continue
        key = (_normalize_label_key(machine), _normalize_label_key(component))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((machine, component))
    if len(pairs) < 2:
        notes.append(f"answer_pairs:{len(pairs)}<2")
        return False, notes
    required_pairs = [
        (m, c)
        for m, c in pairs
        if (_normalize_label_key(m), _normalize_label_key(c)) not in waived_keys
    ]
    waived_hit = [
        f"{m}/{c}"
        for m, c in pairs
        if (_normalize_label_key(m), _normalize_label_key(c)) in waived_keys
    ]
    if waived_hit:
        notes.append(f"pair_waived:{waived_hit}")
    uncovered: list[str] = []
    for machine, component in required_pairs:
        if not any(
            _image_matches_machine(img, machine)
            and _image_matches_component(img, component)
            for img in imgs
        ):
            uncovered.append(f"{machine}/{component}")
    machines = {_normalize_label_key(m) for m, _ in required_pairs}
    if uncovered:
        notes.append(f"pair_missing:{uncovered}")
    if machines and len(imgs) < len(machines):
        notes.append(f"image_count:{len(imgs)}<{len(machines)}")
    ok = not uncovered and (not machines or len(imgs) >= len(machines))
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
        waive_raw = spec.get("image_pair_waive") or []
        pair_waive = [
            (str(m), str(c))
            for m, c in waive_raw
            if m and c
        ]
        return _grade_images_answer_pairs(
            str(result.get("answer") or ""),
            imgs,
            pair_waive=pair_waive,
        )

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


def _gate_band(
    final_score: float | None, *, direct_min: float, answerable: bool
) -> str:
    """Map probe stats to v4 gate band (reject / offer / direct)."""
    if not answerable or final_score is None:
        return "reject"
    if final_score > direct_min:
        return "direct"
    return "offer"


_WEB_SIM_CHOICES = ("direct", "keep_original", "none")


def _normalize_web_sim(value: str | None) -> str:
    sim = (value or "direct").strip().lower()
    if sim not in _WEB_SIM_CHOICES:
        raise ValueError(f"web_sim must be one of {_WEB_SIM_CHOICES}, got {value!r}")
    return sim


def _should_reuse_probe_bundle(gate_probe: dict[str, Any], *, web_sim: str) -> bool:
    """Whether batch answer should inject probe bundle like Web clarify bypass."""
    if web_sim == "none":
        return False
    if web_sim == "direct":
        return gate_probe.get("gate_direct_ok") is True
    if web_sim == "keep_original":
        return bool(gate_probe.get("answerable"))
    return False


def _gate_direct_ok(gate_probe: dict[str, Any] | None) -> bool | None:
    """True when probe qualifies for Web direct path (final > direct_min)."""
    if not gate_probe:
        return None
    fs = gate_probe.get("final_score")
    direct_min = gate_probe.get("direct_rerank_min")
    if not gate_probe.get("answerable") or fs is None:
        return False
    try:
        return float(fs) > float(direct_min)
    except (TypeError, ValueError):
        return False


def _format_gate_final_score(gate_probe: dict[str, Any]) -> str:
    """Human-readable final_score; None only on reject (not offer)."""
    fs = gate_probe.get("final_score")
    if isinstance(fs, (int, float)):
        return f"{fs:.4f}"

    min_thr = gate_probe.get("min_rerank_threshold")
    llm_total = int(gate_probe.get("llm_chunk_total") or 0)
    max_any = gate_probe.get("max_rerank_any")
    if gate_probe.get("scores_unavailable"):
        return f"无 final（{llm_total} 个 chunk 均无 rerank_score）"
    if llm_total == 0:
        return "无 final（probe 未取到 LLM chunk）"
    if isinstance(max_any, (int, float)):
        return f"无 final（池内最高 rerank={max_any:.4f} < min_rerank {min_thr}）"
    return f"无 final（无 chunk ≥ min_rerank {min_thr}）"


def _gate_alert_message(cid: int, gate_probe: dict[str, Any]) -> str:
    direct_min = gate_probe.get("direct_rerank_min")
    band = gate_probe.get("gate_band")
    fs_txt = _format_gate_final_score(gate_probe)
    if band == "offer":
        kind = f"offer（有分但未 > {direct_min}）"
    elif band == "reject":
        kind = "reject（不可答，无 qualifying final）"
    else:
        kind = str(band)
    return (
        f"Q{cid:02d} gate 未达 direct：{kind} · final_score={fs_txt} "
        f"· qualifying_chunks={gate_probe.get('chunk_count')} · "
        f"llm_chunks={gate_probe.get('llm_chunk_total')}"
    )


def _print_gate_alert(cid: int, gate_probe: dict[str, Any]) -> None:
    banner = "!" * 72
    print(f"\n{banner}", flush=True)
    print(f"  *** GATE ALERT（shili17 期望 final > 7）***", flush=True)
    print(f"  {_gate_alert_message(cid, gate_probe)}", flush=True)
    print(f"{banner}\n", flush=True)


def _collect_gate_failures(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        gp = row.get("gate_probe")
        if not isinstance(gp, dict):
            continue
        if gp.get("gate_direct_ok") is False:
            out.append(row)
    return out


def _print_gate_failure_summary(rows: list[dict]) -> int:
    failures = _collect_gate_failures(rows)
    probed = sum(1 for r in rows if isinstance(r.get("gate_probe"), dict))
    direct_ok = probed - len(failures)
    if probed:
        print(
            f"\nGate direct（final > 7）：{direct_ok}/{probed} 达标",
            flush=True,
        )
    if not failures:
        return 0
    print(
        f"\n{'=' * 72}\n"
        f"GATE DIRECT 未达标 {len(failures)}/{probed} — shili17 期望全部 > 7\n"
        f"{'=' * 72}",
        flush=True,
    )
    for row in failures:
        print(f"  • {_gate_alert_message(int(row['id']), row['gate_probe'])}", flush=True)
        print(f"    问句：{row.get('query', '')[:60]}", flush=True)
    print(f"{'=' * 72}\n", flush=True)
    return len(failures)


async def _run_gate_probe(
    lightrag: Any, query: str, *, mode: str
) -> tuple[dict[str, Any], Any | None]:
    from raganything.clarify_gate import (  # noqa: WPS433
        clarify_direct_rerank_min,
        probe_llm_retrieval_full,
    )

    t0 = time.perf_counter()
    probe = await probe_llm_retrieval_full(lightrag, query, mode=mode)
    elapsed_s = time.perf_counter() - t0
    direct_min = clarify_direct_rerank_min()
    gate_band = _gate_band(
        probe.final_score,
        direct_min=direct_min,
        answerable=probe.answerable,
    )
    bundle = probe.bundle
    payload = {
        "final_score": probe.final_score,
        "chunk_count": probe.chunk_count,
        "llm_chunk_total": probe.llm_chunk_total,
        "answerable": probe.answerable,
        "min_rerank_threshold": probe.min_rerank_threshold,
        "direct_rerank_min": direct_min,
        "gate_band": gate_band,
        "scores_unavailable": probe.scores_unavailable,
        "max_rerank_any": probe.max_rerank_any,
        "duration_ms": int(elapsed_s * 1000),
        "duration_s": round(elapsed_s, 2),
        "duration_text": _format_duration(elapsed_s),
        "mode": probe.mode,
        "bundle_chunk_count": len(bundle.document_chunks) if bundle is not None else 0,
        "bundle_reused": False,
    }
    payload["gate_direct_ok"] = _gate_direct_ok(payload)
    return payload, bundle


async def run_cases(
    ids: list[int],
    *,
    mode: str,
    wd: Path,
    pod: Path,
    write_dumps: bool = True,
    skip_gate: bool = False,
    web_sim: str = "direct",
    rag: Any | None = None,
) -> list[dict]:
    from query_debug_dump import persist_query_debug_dump
    from query_doc_steering import strip_manual_circled_step_markers
    from query_progress_hooks import (
        finalize_inline_images,
        query_progress_hooks,
        set_clarify_context_injection,
        set_query_media_roots,
        set_query_text_for_images,
    )
    from stream_cot_parser import parse_complete_cot

    rpc = _load_rpc()
    own_rag = rag is None
    if own_rag:
        rag, _, _ = await rpc._build_rag(wd, pod)
    if rag is None:
        raise RuntimeError("run_cases: rag engine not available")
    parser_root = pod.resolve()
    out: list[dict] = []

    try:
        for cid in ids:
            spec = REF[cid]
            query = spec["query"]
            gate_probe: dict[str, Any] | None = None
            probe_bundle: Any | None = None
            if not skip_gate:
                gate_probe, probe_bundle = await _run_gate_probe(
                    rag.lightrag, query, mode=mode
                )
                fs_txt = _format_gate_final_score(gate_probe)
                print(
                    f"Q{cid:02d} gate {gate_probe.get('gate_band')} "
                    f"final={fs_txt} chunks={gate_probe.get('chunk_count')} "
                    f"{gate_probe.get('duration_text')}",
                    flush=True,
                )
                if gate_probe.get("gate_direct_ok") is False:
                    _print_gate_alert(cid, gate_probe)
                if (
                    probe_bundle is not None
                    and _should_reuse_probe_bundle(gate_probe, web_sim=web_sim)
                ):
                    set_clarify_context_injection(probe_bundle)
                    gate_probe["bundle_reused"] = True
                    print(
                        f"Q{cid:02d} web-sim={web_sim}: reusing gate probe bundle "
                        f"({gate_probe.get('bundle_chunk_count')} chunks)",
                        flush=True,
                    )
            t0 = time.perf_counter()
            thinking = ""
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
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            elapsed_s = elapsed_ms / 1000.0
            elapsed_text = _format_duration(elapsed_s)
            dump_path: Path | None = None
            if write_dumps:
                dump_path = persist_query_debug_dump(
                    query=query,
                    mode=mode,
                    parser_root=parser_root,
                    thinking=thinking or None,
                    answer=answer or None,
                    duration_ms=elapsed_ms,
                    enabled=True,
                    name_prefix=f"Q{cid:02d}",
                )
            row = {
                "id": cid,
                "query": query,
                "web_sim": web_sim if not skip_gate else None,
                "gate_probe": gate_probe,
                "answer": answer,
                "duration_ms": elapsed_ms,
                "duration_s": round(elapsed_s, 2),
                "duration_text": elapsed_text,
                "images": inline.get("images") or [],
                "placements": inline.get("placements") or [],
                "debug": inline.get("debug") or {},
                "dump_path": str(dump_path) if dump_path else None,
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
                f"{elapsed_text} imgs={len(row['images'])} pl={len(row['placements'])}",
                flush=True,
            )
            if not g["text_ok"]:
                print(f"  text: {g['text_miss']}", flush=True)
            if not g["image_ok"]:
                print(f"  img: {g['image_notes']}", flush=True)
            if dump_path:
                print(f"  dump: {dump_path}", flush=True)
            for c in [i.get("caption") for i in row["images"]]:
                try:
                    print(f"  caption: {c}", flush=True)
                except UnicodeEncodeError:
                    safe = str(c).encode("utf-8", errors="replace").decode("utf-8")
                    print(f"  caption: {safe}", flush=True)
    finally:
        if own_rag:
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
    web_sim: str | None = None,
    round_no: int | None = None,
    rounds_total: int | None = None,
) -> None:
    passed = sum(1 for r in rows if r["grade"]["ok"])
    text_only = sum(1 for r in rows if r["grade"]["text_ok"])
    gate_failures = _collect_gate_failures(rows)
    probed = sum(1 for r in rows if isinstance(r.get("gate_probe"), dict))
    direct_ok = probed - len(gate_failures)
    ids_label = ", ".join(f"Q{r['id']}" for r in rows)
    lines = [
        f"# Web 路径批测（{ids_label}）",
        "",
        f"- 时间：{datetime.now().astimezone().isoformat()}",
    ]
    if round_no is not None and rounds_total is not None and rounds_total > 1:
        lines.append(f"- 轮次：**{round_no}/{rounds_total}**")
    lines.extend(
        [
            f"- 模式：{mode}",
            f"- 工作目录：`{wd}`",
            f"- 媒体根：`{media_root}`",
            f"- 参考答案：`docs/测试例参考答案.md`",
            f"- 通过（文字+配图）：**{passed}/{len(rows)}**",
            f"- 仅文字通过：**{text_only}/{len(rows)}**",
        ]
    )
    if web_sim:
        lines.append(
            f"- Web 模拟：``{web_sim}``（answer 复用 gate probe bundle，对齐 Web clarify bypass）"
        )
    if probed:
        lines.append(
            f"- Gate direct（final > 7，shili17 期望全达标）：**{direct_ok}/{probed}**"
        )
    lines.extend(
        [
            f"- Query dumps：`logs/query_dumps/`（本批 JSON 见各题 `dump_path`）",
            "",
        ]
    )
    if gate_failures:
        lines.extend(
            [
                "## ⚠️ Gate direct 未达标（offer：有分但 ≤7；reject：无 final）",
                "",
                "shili17 期望每题门控 probe 均为 **direct**（`final_score > CLARIFY_DIRECT_RERANK_MIN`）。",
                "",
            ]
        )
        for row in gate_failures:
            gp = row["gate_probe"]
            fs_txt = _format_gate_final_score(gp)
            lines.append(
                f"- **Q{row['id']}** · {fs_txt} · band=`{gp.get('gate_band')}` · "
                f"chunks={gp.get('chunk_count')} · {row.get('query', '')}"
            )
        lines.append("")
    lines.extend(
        [
            "## 汇总",
            "",
            "| ID | 结果 | 文字 | 配图 | gate final | gate | 耗时 | 图数 | 图注摘要 |",
            "|----|------|------|------|------------|------|------|------|----------|",
        ]
    )
    for r in rows:
        g = r["grade"]
        caps = [str(i.get("caption") or "") for i in r.get("images") or []]
        cap_summary = "; ".join(c for c in caps if c) or "—"
        if len(cap_summary) > 48:
            cap_summary = cap_summary[:45] + "…"
        gp = r.get("gate_probe") or {}
        fs = gp.get("final_score")
        fs_cell = _format_gate_final_score(gp) if gp else "—"
        gate_band = str(gp.get("gate_band") or "—")
        if gp.get("gate_direct_ok") is False:
            fs_cell = f"**⚠️ {fs_cell}**"
            gate_band = f"**⚠️ {gate_band}**"
        lines.append(
            f"| Q{r['id']} | {'✓' if g['ok'] else '✗'} | "
            f"{'✓' if g['text_ok'] else '✗'} | "
            f"{'✓' if g['image_ok'] else '✗'} | "
            f"{fs_cell} | {gate_band} | "
            f"{r.get('duration_text') or _format_duration((r.get('duration_ms') or 0) / 1000)} | {len(r.get('images') or [])} | {cap_summary} |"
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
            f"*答题耗时 {r.get('duration_text') or _format_duration((r.get('duration_ms') or 0) / 1000)} · 文字 {g['text_ok']} · 配图 {g['image_ok']}*"
        )
        gp = r.get("gate_probe") or {}
        if gp:
            fs_txt = _format_gate_final_score(gp)
            prefix = "- 门控 probe："
            if gp.get("gate_direct_ok") is False:
                prefix = (
                    f"- **⚠️ GATE 未达 direct（期望 final > {gp.get('direct_rerank_min')}）** · "
                    "门控 probe："
                )
            lines.append(
                f"{prefix}band=`{gp.get('gate_band')}` · "
                f"final_score={fs_txt} · qualifying_chunks={gp.get('chunk_count')} · "
                f"llm_chunks={gp.get('llm_chunk_total')} · "
                f"probe耗时={gp.get('duration_text') or _format_duration((gp.get('duration_ms') or 0) / 1000)} · "
                f"min_rerank={gp.get('min_rerank_threshold')} · "
                f"direct_min={gp.get('direct_rerank_min')} · "
                f"bundle_reused={gp.get('bundle_reused')}"
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
        dump_path = r.get("dump_path")
        if dump_path:
            lines.append(f"- query dump：`{dump_path}`")
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


async def run_all_rounds(
    ids: list[int],
    *,
    mode: str,
    wd: Path,
    pod: Path,
    write_dumps: bool,
    skip_gate: bool,
    web_sim: str,
    rounds: int,
    report_dir: Path,
    session_stamp: str,
    suffix: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Run one or more full batches in a single event loop (required for --rounds > 1)."""
    round_summaries: list[dict[str, Any]] = []
    any_fail = False

    rpc = _load_rpc()
    rag, _, _ = await rpc._build_rag(wd, pod)
    try:
        for rnd in range(1, rounds + 1):
            if rounds > 1:
                print(f"\n{'=' * 60}\n=== Round {rnd}/{rounds} ===\n{'=' * 60}", flush=True)
            rows = await run_cases(
                ids,
                mode=mode,
                wd=wd,
                pod=pod,
                write_dumps=write_dumps,
                skip_gate=skip_gate,
                web_sim=web_sim,
                rag=rag,
            )
            round_tag = f"_r{rnd:02d}" if rounds > 1 else ""
            json_path = report_dir / f"{session_stamp}_{suffix}{round_tag}.json"
            md_path = report_dir / f"{session_stamp}_{suffix}{round_tag}.md"
            json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(
                md_path,
                rows,
                mode=mode,
                wd=wd,
                media_root=pod,
                web_sim=web_sim if not skip_gate else None,
                round_no=rnd if rounds > 1 else None,
                rounds_total=rounds if rounds > 1 else None,
            )
            passed = sum(1 for r in rows if r["grade"]["ok"])
            gate_fail_count = _print_gate_failure_summary(rows)
            print(f"\nReport: {md_path}", flush=True)
            print(f"Answer grade: {passed}/{len(rows)} passed", flush=True)
            round_fail = bool(gate_fail_count) or passed < len(rows)
            if round_fail:
                any_fail = True
            round_summaries.append(
                {
                    "round": rnd,
                    "md_path": str(md_path),
                    "json_path": str(json_path),
                    "passed": passed,
                    "total": len(rows),
                    "gate_fail_count": gate_fail_count,
                    "ok": not round_fail,
                }
            )
    finally:
        await rag.finalize_storages()

    return round_summaries, any_fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-path batch test vs 测试例参考答案")
    parser.add_argument(
        "--ids",
        default=os.getenv("RAG_WEB_PATH_CASE_IDS", ""),
        help="Comma-separated case ids, e.g. 2,3,8,17 or Q2,Q3 (default: all Q1–Q17)",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip clarify gate probe (no final_score; faster, old behavior)",
    )
    parser.add_argument(
        "--web-sim",
        default=os.getenv("RAG_WEB_PATH_WEB_SIM", "direct"),
        choices=_WEB_SIM_CHOICES,
        help=(
            "How answer aquery reuses gate probe bundle: direct (Web ClarifyBypass direct, "
            "default), keep_original (reuse whenever probe answerable), none (independent "
            "second retrieval for A/B)"
        ),
    )
    parser.add_argument(
        "--no-dump",
        action="store_true",
        help="Skip writing logs/query_dumps/*.json per case (default: write dumps)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.getenv("RAG_WEB_PATH_ROUNDS", "1")),
        help="Run the full batch N times consecutively (default: 1; env RAG_WEB_PATH_ROUNDS)",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    ids = _parse_ids(args.ids)
    wd = Path(os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    mode = os.getenv("RAG_QUERY_MODE", "mix")
    label = ", ".join(f"Q{i}" for i in ids)
    rounds_label = f", {args.rounds} round(s)" if args.rounds > 1 else ""
    print(f"Running {label} web path{rounds_label}, wd={wd}", flush=True)
    if not args.no_dump:
        print("Query dumps: logs/query_dumps/ (per case, prefix Qxx)", flush=True)
    web_sim = _normalize_web_sim(args.web_sim)
    if not args.skip_gate:
        print("Gate probe: probe_llm_retrieval_full (mix + rerank, final_score)", flush=True)
        print(f"Web sim: {web_sim} (answer bundle reuse)", flush=True)
    report_dir = _ROOT / "logs" / "web_path_q1_17"
    report_dir.mkdir(parents=True, exist_ok=True)
    session_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix = f"q{'_'.join(str(i) for i in ids)}" if len(ids) < 17 else "all"
    round_summaries, any_fail = asyncio.run(
        run_all_rounds(
            ids,
            mode=mode,
            wd=wd,
            pod=pod,
            write_dumps=not args.no_dump,
            skip_gate=args.skip_gate,
            web_sim=web_sim,
            rounds=args.rounds,
            report_dir=report_dir,
            session_stamp=session_stamp,
            suffix=suffix,
        )
    )

    if args.rounds > 1:
        print(f"\n{'=' * 60}\nMulti-round summary ({args.rounds} rounds)", flush=True)
        for s in round_summaries:
            status = "OK" if s["ok"] else "FAIL"
            gate_txt = (
                f", gate_fail={s['gate_fail_count']}" if s["gate_fail_count"] else ""
            )
            print(
                f"  r{s['round']:02d} {status} answer={s['passed']}/{s['total']}{gate_txt} "
                f"→ {s['md_path']}",
                flush=True,
            )
        ok_rounds = sum(1 for s in round_summaries if s["ok"])
        print(f"Rounds passed: {ok_rounds}/{args.rounds}", flush=True)

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
