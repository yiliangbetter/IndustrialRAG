"""Scope retrieval to document sources implied by the user query.

Steering profiles are loaded from ``config/query_steering_profiles.json`` (or
``RAG_QUERY_STEERING_PROFILES``). When a profile matches, unrelated manual PDFs
are dropped after rerank and a report is exposed for the Web UI / logs.
"""

from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROFILES_PATH = _ROOT / "config" / "query_steering_profiles.json"

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


def _profiles_path() -> Path:
    raw = (os.getenv("RAG_QUERY_STEERING_PROFILES") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_PROFILES_PATH


@lru_cache(maxsize=1)
def _load_profiles_from_file() -> list[dict[str, Any]]:
    path = _profiles_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to load steering profiles from %s: %s", path, exc
        )
        return []
    return data if isinstance(data, list) else []


def _load_extra_profiles() -> list[dict[str, Any]]:
    raw = (os.getenv("RAG_QUERY_DOC_FILTER_RULES_JSON") or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def load_steering_profiles() -> list[dict[str, Any]]:
    """All active steering profiles (file + env JSON overlay)."""
    return _load_profiles_from_file() + _load_extra_profiles()


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", "", (q or "").strip())


def resolve_machine_profile(query: str) -> dict[str, Any] | None:
    """Pick the best-matching profile from the question text."""
    qn = _normalize_query(query)
    if not qn:
        return None
    profiles = load_steering_profiles()
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
    """When retrieval spans multiple manuals, keep each source's wording separate."""
    if not _env_bool("RAG_QUERY_AUTO_STEERING", True):
        return ""

    parts = [
        "回答须忠实于检索到的各文档原文，不得自行归纳或重组：",
        "若内容来自多份不同设备、产品或手册，必须按来源分别列出，"
        "小节标题写明对应设备名称或文档标题；",
        "不得把不同来源的周期、型号、操作步骤合并成自编分类或「通用流程」；",
        "保留原文中的数值、型号与步骤表述；引用标记与文末 References 须与正文实际引用一致。",
        "用户未指定单一来源时：若检索到多个来源的同类条目，须分别说明，不要混为一谈。",
    ]
    profile = resolve_machine_profile(query)
    if profile:
        label = str(profile.get("label") or "")
        if label:
            parts.append(f"用户问题已指向「{label}」；优先使用该来源对应 chunk 的原文表述。")
    return " ".join(parts)


def build_steering_user_prompt(query: str) -> str:
    """Per-query hint for the LLM when a steering profile matches."""
    if not _env_bool("RAG_QUERY_KG_STEERING", True):
        return ""
    profile = resolve_machine_profile(query)
    if not profile:
        return ""
    label = str(profile.get("label") or "")
    extra = str(profile.get("steering_prompt") or "").strip()
    parts = [
        f"用户问题针对「{label}」。只引用与该来源对应手册的正文 chunk；"
        "若知识图谱实体描述与其它文档合并后冲突，以 chunk 正文为准。"
    ]
    if extra:
        parts.append(extra)
    return " ".join(parts)


def build_user_prompt_for_query(query: str) -> str:
    """Merge answer-fidelity rules and profile steering for LightRAG."""
    parts: list[str] = []
    fidelity = build_answer_fidelity_user_prompt(query)
    if fidelity:
        parts.append(fidelity)
    steer = build_steering_user_prompt(query)
    if steer:
        parts.append(steer)
    return "\n\n".join(parts)


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


def install_query_steering_hooks() -> None:
    """Document path filter after rerank (idempotent)."""
    install_doc_filter_on_rerank()
