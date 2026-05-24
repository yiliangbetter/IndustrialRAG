"""Scope retrieval to the machine type mentioned in the user query.

Rules live in code (``MACHINE_PROFILES``), not per-machine ``.env`` entries.
When a profile matches, unrelated manual PDFs are dropped after rerank and a
report is exposed for the Web UI / logs.
"""

from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Order matters: longer / more specific query phrases first.
MACHINE_PROFILES: list[dict[str, Any]] = [
    {
        "id": "high_speed_smart",
        "label": "高速智能封边机",
        "query_phrases": [
            "高速智能封边机",
            "高速智能",
            "NB9-Smart",
            "NB10-Smart",
        ],
        "deny_path_substrings": [
            "自动封边机维护保养手册",
            "高速自动封边机维护保养手册",
            "双端封边机维护保养手册",
            "数控六面钻",
            "PC封边机电气",
        ],
    },
    {
        "id": "high_speed_auto",
        "label": "高速自动封边机",
        "query_phrases": [
            "高速自动封边机",
            "高速自动",
            "NB6PG",
            "NB7PCG",
            "NB8PCHGM",
        ],
        "deny_path_substrings": [
            "自动封边机维护保养手册",
            "封边机连线项目维护保养手册",
            "双端封边机维护保养手册",
            "数控六面钻",
        ],
    },
    {
        "id": "double_end",
        "label": "双端封边机",
        "query_phrases": [
            "双端封边机",
            "双端",
            "NB6S2",
            "NB7HS2",
            "NB8CS2",
        ],
        "deny_path_substrings": [
            "自动封边机维护保养手册",
            "封边机连线项目维护保养手册",
            "高速自动封边机维护保养手册",
            "数控六面钻",
        ],
    },
    {
        "id": "auto_edge",
        "label": "自动封边机",
        "query_phrases": [
            "自动封边机",
            "NBC332",
            "NB5J",
            "NB6J",
            "NB6CJ",
            "NB7CJ",
            "NB7CJM",
            "NB557D",
        ],
        "query_exclude_if_contains": [
            "高速智能",
            "高速自动",
            "双端封边",
            "连线项目",
        ],
        "deny_path_substrings": [
            "封边机连线项目维护保养手册",
            "高速自动封边机维护保养手册",
            "双端封边机维护保养手册",
            "数控六面钻",
        ],
    },
]

_last_filter_report: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_filter_report", default=None
)


@dataclass
class FilterReport:
    active: bool
    machine_label: str = ""
    machine_id: str = ""
    kept_sources: list[str] = field(default_factory=list)
    removed_sources: list[dict[str, str]] = field(default_factory=list)

    def to_sse_payload(self) -> dict[str, Any]:
        return {
            "type": "retrieval_scope",
            "active": self.active,
            "machine_label": self.machine_label,
            "machine_id": self.machine_id,
            "kept_sources": self.kept_sources,
            "removed_sources": self.removed_sources,
            "text": self.summary_text(),
        }

    def summary_text(self) -> str:
        if not self.active:
            return ""
        parts = [f"检索范围：{self.machine_label} 相关手册"]
        if self.removed_sources:
            names = [
                r.get("title") or r.get("path", "")
                for r in self.removed_sources[:4]
            ]
            extra = len(self.removed_sources) - len(names)
            tail = f" 等{extra}份" if extra > 0 else ""
            parts.append(f"（已排除：{'、'.join(names)}{tail}）")
        return "".join(parts)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _load_extra_profiles() -> list[dict[str, Any]]:
    raw = (os.getenv("RAG_QUERY_DOC_FILTER_RULES_JSON") or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", "", (q or "").strip())


def resolve_machine_profile(query: str) -> dict[str, Any] | None:
    """Pick the best-matching machine profile from the question text."""
    qn = _normalize_query(query)
    if not qn:
        return None
    profiles = MACHINE_PROFILES + _load_extra_profiles()
    best: tuple[int, dict[str, Any]] | None = None
    for profile in profiles:
        if any(
            ex.replace(" ", "") in qn
            for ex in (profile.get("query_exclude_if_contains") or [])
        ):
            continue
        for phrase in profile.get("query_phrases") or []:
            pn = phrase.replace(" ", "")
            if pn and pn in qn and (best is None or len(pn) > best[0]):
                best = (len(pn), profile)
    return best[1] if best else None


def active_deny_substrings(query: str) -> tuple[list[str], dict[str, Any] | None]:
    if not _env_bool("RAG_QUERY_DOC_FILTER", True):
        return [], None
    profile = resolve_machine_profile(query)
    if not profile:
        return [], None
    deny = list(profile.get("deny_path_substrings") or [])
    extra = (os.getenv("RAG_QUERY_DOC_DENY_SUBSTRINGS") or "").strip()
    if extra:
        deny.extend(x.strip() for x in extra.split(",") if x.strip())
    seen: set[str] = set()
    out: list[str] = []
    for s in deny:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out, profile


def _doc_path(doc: dict) -> str:
    for key in ("file_path", "filepath", "source", "doc_id", "document_id"):
        val = doc.get(key)
        if val:
            return str(val)
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("file_path", "source", "filename"):
            val = meta.get(key)
            if val:
                return str(val)
    return ""


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1] if path else ""


def filter_retrieved_docs_by_query(query: str, docs: list[dict]) -> list[dict]:
    kept, _report = filter_retrieved_docs_with_report(query, docs)
    return kept


def filter_retrieved_docs_with_report(
    query: str, docs: list[dict]
) -> tuple[list[dict], FilterReport]:
    deny, profile = active_deny_substrings(query)
    if not deny or not docs:
        report = FilterReport(active=False)
        _last_filter_report.set(report.to_sse_payload())
        return docs, report

    label = str(profile.get("label") or profile.get("id") or "")
    mid = str(profile.get("id") or "")
    kept: list[dict] = []
    removed: list[dict[str, str]] = []
    kept_paths: list[str] = []

    for doc in docs:
        path = _doc_path(doc)
        hit = next((s for s in deny if path and s in path), None)
        if hit:
            removed.append(
                {
                    "path": path,
                    "title": _basename(path),
                    "reason": f"与「{label}」无关（命中排除规则：{hit}）",
                }
            )
            continue
        kept.append(doc)
        if path and path not in kept_paths:
            kept_paths.append(path)

    if not kept:
        report = FilterReport(active=True, machine_label=label, machine_id=mid)
        _last_filter_report.set(report.to_sse_payload())
        return docs, report

    report = FilterReport(
        active=True,
        machine_label=label,
        machine_id=mid,
        kept_sources=[_basename(p) for p in kept_paths],
        removed_sources=removed,
    )
    _last_filter_report.set(report.to_sse_payload())
    return kept, report


def consume_filter_report() -> dict[str, Any] | None:
    report = _last_filter_report.get()
    _last_filter_report.set(None)
    return report


def build_answer_fidelity_user_prompt(query: str) -> str:
    """When retrieval spans multiple manuals, keep each device's wording separate."""
    if not _env_bool("RAG_QUERY_AUTO_STEERING", True):
        return ""
    qn = _normalize_query(query)
    profile = resolve_machine_profile(query)

    parts = [
        "回答须忠实于检索到的各手册原文，不得自行归纳或重组：",
        "若内容来自多份不同设备或手册，必须按设备或手册分别列出，"
        "小节标题写明设备名称（如「双端封边机」「高速智能封边机」）或对应手册；",
        "不得把不同手册的保养周期、润滑脂型号、操作步骤合并成"
        "「常规检查与润滑」「深度清理与润滑」「日常保养」等自编分类；",
        "保留原文的保养周期、润滑剂名称与型号（如润滑脂2#、长城润滑脂3#）"
        "及步骤表述；引用标记与文末 References 须与正文实际引用一致。",
    ]

    if not profile and any(
        marker in qn for marker in ("保养", "润滑", "维护", "检查", "更换", "清理")
    ):
        parts.append(
            "用户未指定单一机型时：若检索到多台设备的同类保养条目，"
            "须分别说明各设备对应方法，不要混为一谈或只给出一条「通用」流程。"
        )

    if "传动丝杆" in qn or ("传动" in qn and "丝杆" in qn):
        parts.append(
            "传动丝杆：不同手册的保养周期与润滑脂可能不同"
            "（例如双端封边机每周长城润滑脂3#，高速智能封边机每年润滑脂2#），"
            "须分设备说明，勿合并为同一保养流程。"
        )

    return " ".join(parts)


def build_steering_user_prompt(query: str) -> str:
    """Per-query hint for the LLM (not per-machine ``.env`` entries)."""
    if not _env_bool("RAG_QUERY_KG_STEERING", True):
        return ""
    profile = resolve_machine_profile(query)
    if not profile:
        return ""
    label = str(profile.get("label") or "")
    parts = [
        f"用户问题针对「{label}」。只引用与该机型对应手册的正文 chunk；"
        "若知识图谱实体描述与其它机型手册合并后冲突，以 chunk 正文为准。"
    ]
    qn = _normalize_query(query)
    if profile.get("id") == "high_speed_smart" and "输送链条" in qn:
        parts.append(
            "输送链条保养以手册 3.1.2 及附表为准：季度/半年用手动黄油枪加注润滑脂2#；"
            "勿写「每天加注长城导轨油68#」，除非 chunk 正文明确写出该条。"
        )
    return " ".join(parts)


def build_user_prompt_for_query(query: str) -> str:
    """Merge answer-fidelity rules and machine-specific steering for LightRAG."""
    parts: list[str] = []
    fidelity = build_answer_fidelity_user_prompt(query)
    if fidelity:
        parts.append(fidelity)
    steer = build_steering_user_prompt(query)
    if steer:
        parts.append(steer)
    return "\n\n".join(parts)


def _line_is_chain_daily_oil_noise(line: str) -> bool:
    lower = line.lower()
    chain_related = (
        "conveyor chain" in lower
        or "输送链条" in line
        or ("输送" in line and "链条" in line)
    )
    if not chain_related:
        return False
    oil_markers = (
        "长城导轨油",
        "great wall guide",
        "greatwall guide",
        "centralized lubrication",
        "每天",
        "daily",
    )
    return any(m in lower or m in line for m in oil_markers)


def scrub_kg_context(query: str, context: str) -> str:
    """Remove KG lines that wrongly merge other manuals' chain lubrication."""
    if not _env_bool("RAG_QUERY_KG_SCRUB", True):
        return context
    profile = resolve_machine_profile(query)
    if not profile or profile.get("id") != "high_speed_smart":
        return context
    qn = _normalize_query(query)
    if "输送链条" not in qn:
        return context
    if not context:
        return context
    kept = [ln for ln in context.split("\n") if not _line_is_chain_daily_oil_noise(ln)]
    return "\n".join(kept)


def install_doc_filter_on_rerank() -> None:
    import lightrag.utils as ut

    orig = ut.apply_rerank_if_enabled
    if getattr(orig, "_doc_filter_wrapped", False):
        return

    async def _wrapped(
        query: str,
        retrieved_docs: list[dict],
        global_config: dict,
        enable_rerank: bool = True,
        top_n: int | None = None,
    ) -> list[dict]:
        docs = await orig(
            query, retrieved_docs, global_config, enable_rerank, top_n
        )
        kept, _report = filter_retrieved_docs_with_report(query, docs)
        return kept

    _wrapped._doc_filter_wrapped = True  # type: ignore[attr-defined]
    ut.apply_rerank_if_enabled = _wrapped  # type: ignore[method-assign]


def install_kg_context_scrub() -> None:
    import lightrag.operate as op
    from dataclasses import replace

    orig = op._build_query_context
    if getattr(orig, "_kg_scrub_wrapped", False):
        return

    async def _wrapped(*args: Any, **kwargs: Any):
        result = await orig(*args, **kwargs)
        if result is None:
            return result
        query = args[0] if args else str(kwargs.get("query") or "")
        scrubbed = scrub_kg_context(query, result.context or "")
        if scrubbed != result.context:
            result = replace(result, context=scrubbed)
        return result

    _wrapped._kg_scrub_wrapped = True  # type: ignore[attr-defined]
    op._build_query_context = _wrapped  # type: ignore[method-assign]


def install_query_steering_hooks() -> None:
    """Chunk file_path filter + KG context scrub (idempotent)."""
    install_doc_filter_on_rerank()
    install_kg_context_scrub()
