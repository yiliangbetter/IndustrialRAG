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
_kg_filter_query: ContextVar[str | None] = ContextVar("kg_filter_query", default=None)

_LIGHTRAG_SEP = "<SEP>"

# Manual step markers (①②③ …) — strip in user-facing answers, keep wording.
_CIRCLED_STEP_BEFORE_CJK = re.compile(
    r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    r"❶❷❸❹❺❻❼❽❾❿"
    r"](?=\s*[\u4e00-\u9fff])"
)


def strip_manual_circled_step_markers(text: str) -> str:
    """Remove circled list prefixes (①…) before Chinese text in answers."""
    if not text or not text.strip():
        return text
    return _CIRCLED_STEP_BEFORE_CJK.sub("", text)


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


# Ingest template line present on maintenance-manual forewords (not a domain phrase list).
_CATALOG_MODEL_MARKER = "本手册适用产品型号"


def _query_discriminative_terms(query: str) -> list[str]:
    from raganything.utils import discriminative_terms  # noqa: WPS433

    return discriminative_terms(query, min_len=3)


def is_catalog_product_model_query(query: str) -> bool:
    """Broad product-line / model-count questions (not single-machine maintenance)."""
    if resolve_machine_profile(query):
        return False
    q = (query or "").strip()
    if not q:
        return False
    asks_scope = bool(
        re.search(
            r"哪些|多少|一共|总共|全部|有哪些|几种|列举|清单|概况|多少个|一共有多少|多少种",
            q,
        )
    )
    asks_models = bool(re.search(r"型号|机型|产品", q))
    return asks_scope and asks_models


def _default_min_rerank_score() -> float:
    raw = os.getenv("MIN_RERANK_SCORE") or "0.28"
    try:
        return float(raw)
    except ValueError:
        return 0.28


def catalog_query_min_rerank_score() -> float:
    raw = os.getenv("RAG_CATALOG_QUERY_MIN_RERANK_SCORE") or "0.0"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _chunk_has_catalog_marker(doc: dict) -> bool:
    return _CATALOG_MODEL_MARKER in str(doc.get("content") or "")


def _catalog_chunk_relevant_to_query(query: str, doc: dict) -> bool:
    """Keep foreword catalog lines whose path/body overlap query terms (no manual name lists)."""
    if not _chunk_has_catalog_marker(doc):
        return False
    terms = _query_discriminative_terms(query)
    if not terms:
        return True
    blob = f"{_doc_path(doc)} {str(doc.get('content') or '')[:500]}"
    return any(len(term) >= 3 and term in blob for term in terms)


def _load_catalog_chunks_from_storage() -> list[dict]:
    try:
        from client_paths import get_rag_storage_dir  # noqa: WPS433

        store = get_rag_storage_dir()
    except Exception:
        store = _ROOT / "data" / "rag_storage"
    path = Path(store) / "kv_store_text_chunks.json"
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    out: list[dict] = []
    for chunk_id, row in raw.items():
        if not isinstance(row, dict):
            continue
        content = str(row.get("content") or "")
        if _CATALOG_MODEL_MARKER not in content:
            continue
        doc = dict(row)
        doc.setdefault("content", content)
        doc.setdefault("id", chunk_id)
        out.append(doc)
    return out


def supplement_catalog_product_model_chunks(
    query: str,
    docs: list[dict],
    *,
    rerank_pool: list[dict] | None = None,
) -> list[dict]:
    """Ensure each relevant manual's 「本手册适用产品型号」 foreword chunk is present."""
    if not _env_bool("RAG_CATALOG_QUERY_BOOST", True):
        return docs
    if not is_catalog_product_model_query(query):
        return docs

    seen_paths: set[str] = set()
    merged: list[dict] = []

    def add_doc(doc: dict) -> None:
        path = _doc_path(doc)
        if not path or path in seen_paths:
            return
        if not _catalog_chunk_relevant_to_query(query, doc):
            return
        seen_paths.add(path)
        boosted = dict(doc)
        boosted["rerank_score"] = max(float(boosted.get("rerank_score") or 0), 0.99)
        merged.append(boosted)

    for doc in docs:
        add_doc(doc)
    for doc in rerank_pool or []:
        add_doc(doc)
    for doc in _load_catalog_chunks_from_storage():
        add_doc(doc)

    if not merged:
        return docs
    return merged + [d for d in docs if _doc_path(d) not in seen_paths]


def _stash_catalog_boost_stats(count: int) -> None:
    if count <= 0:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["catalog_boost"] = {"chunks_added": count}
    _last_filter_report.set(payload)


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


def _path_hits_deny(path: str, deny: list[str]) -> str | None:
    if not path or not deny:
        return None
    hit = next((s for s in deny if s in path), None)
    return hit


def _split_sep_field(value: str) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    if _LIGHTRAG_SEP in raw:
        return [p.strip() for p in raw.split(_LIGHTRAG_SEP) if p.strip()]
    return [raw]


def _join_sep(parts: list[str]) -> str:
    return _LIGHTRAG_SEP.join(p for p in parts if p)


def _filter_sep_merged_kg_record(
    record: dict[str, Any],
    deny: list[str],
    *,
    path_key: str = "file_path",
    text_key: str = "description",
) -> dict[str, Any] | None:
    """Keep only file_path / description segments that are not denied (LightRAG ``<SEP>`` merge)."""
    paths = _split_sep_field(str(record.get(path_key) or ""))
    texts = _split_sep_field(str(record.get(text_key) or ""))
    if not paths:
        return record

    kept_paths: list[str] = []
    kept_texts: list[str] = []
    for i, path in enumerate(paths):
        if _path_hits_deny(path, deny):
            continue
        kept_paths.append(path)
        if i < len(texts):
            kept_texts.append(texts[i])
        elif len(texts) == 1:
            kept_texts.append(texts[0])

    if not kept_paths:
        return None

    out = dict(record)
    out[path_key] = _join_sep(kept_paths)
    if kept_texts:
        out[text_key] = _join_sep(kept_texts)
    elif texts:
        out[text_key] = texts[0] if len(kept_paths) == 1 else _join_sep(texts[: len(kept_paths)])
    return out


def _relation_endpoints(relation: dict[str, Any]) -> tuple[str | None, str | None]:
    if "src_tgt" in relation:
        pair = relation.get("src_tgt") or (None, None)
        return pair[0], pair[1]
    return relation.get("src_id"), relation.get("tgt_id")


def filter_kg_search_by_query(
    query: str, search_result: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Drop or trim KG entities/relations from denied manuals when a steering profile matches."""
    if not _env_bool("RAG_QUERY_KG_FILE_FILTER", True):
        return search_result, None

    deny, profile = active_deny_substrings(query)
    if not deny or not profile:
        return search_result, None

    entities_in = list(search_result.get("final_entities") or [])
    relations_in = list(search_result.get("final_relations") or [])

    filtered_entities: list[dict[str, Any]] = []
    dropped_entities: list[str] = []
    trimmed_entities: list[str] = []

    for ent in entities_in:
        name = str(ent.get("entity_name") or "")
        before = str(ent.get("description") or "")
        kept = _filter_sep_merged_kg_record(ent, deny)
        if kept is None:
            if name:
                dropped_entities.append(name)
            continue
        if kept.get("description") != before:
            if name:
                trimmed_entities.append(name)
        filtered_entities.append(kept)

    kept_names = {
        str(e.get("entity_name") or "")
        for e in filtered_entities
        if e.get("entity_name")
    }

    filtered_relations: list[dict[str, Any]] = []
    dropped_relations = 0
    for rel in relations_in:
        kept = _filter_sep_merged_kg_record(rel, deny)
        if kept is None:
            dropped_relations += 1
            continue
        e1, e2 = _relation_endpoints(kept)
        if kept_names:
            if (e1 and e1 not in kept_names) or (e2 and e2 not in kept_names):
                dropped_relations += 1
                continue
        filtered_relations.append(kept)

    out = dict(search_result)
    out["final_entities"] = filtered_entities
    out["final_relations"] = filtered_relations

    stats = {
        "active": True,
        "machine_label": str(profile.get("label") or ""),
        "entities_before": len(entities_in),
        "entities_after": len(filtered_entities),
        "relations_before": len(relations_in),
        "relations_after": len(filtered_relations),
        "dropped_entities": dropped_entities[:12],
        "trimmed_entities": trimmed_entities[:12],
        "dropped_relations": dropped_relations,
    }
    return out, stats


def _stash_kg_filter_stats(stats: dict[str, Any] | None) -> None:
    if not stats:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["kg_filter"] = stats
    _last_filter_report.set(payload)


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
    payload = report.to_sse_payload()
    prev = _last_filter_report.get()
    if isinstance(prev, dict) and prev.get("kg_filter"):
        payload["kg_filter"] = prev["kg_filter"]
    _last_filter_report.set(payload)
    return kept, report


def consume_filter_report() -> dict[str, Any] | None:
    report = _last_filter_report.get()
    _last_filter_report.set(None)
    return report


def build_maintenance_section_fidelity_prompt(query: str) -> str:
    """When the user asks what to add/do for a named maintenance item, anchor on section body."""
    if not _env_bool("RAG_QUERY_SECTION_FIDELITY", True):
        return ""
    q = (query or "").strip()
    if not q or not re.search(r"保养|加注|润滑|清洁|步骤|周期", q):
        return ""
    return (
        "用户问的是具体保养条目（如「输送链条保养」「3.1.x」等）。"
        "只根据检索 chunk 中该条目下「保养周期」「保养内容」「保养步骤」作答；"
        "勿把其它机型、其它部位、全书「润滑部位和周期」总表或知识图谱合并进来的说法并入；"
        "不得自编「日常润滑」等正文未出现的周期名称。"
        "若正文仅写「每季度一次、润滑脂2#」等，只答这些，不要补充正文未出现的油品型号。"
    )


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
        "若 chunk 中同一条款（如带①②③的同一行或同一句）并列写出多项检查/操作，"
        "须完整复述该条款的全部检查项，不得因用户只提及其中一项关键词而省略同句中的其它要求；"
        "知识图谱实体描述若比 chunk 正文更短，以 chunk 正文为准；"
        "图谱或附录表里出现、但该条目正文 chunk 未写明的油品/周期/方式，一律不得写入答案。",
        "输出时勿保留手册中的圈号序号（①②③等），直接写出检查/操作内容即可。",
    ]
    parts.append(build_maintenance_section_fidelity_prompt(query))
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
        "若知识图谱实体描述与其它文档合并后冲突，以 chunk 正文为准；"
        "禁止把其它手册灌库进图谱的油品/周期（如未出现在该手册 chunk 的型号）写进答案。"
    ]
    if extra:
        parts.append(extra)
    return " ".join(parts)


def build_catalog_model_listing_prompt(query: str) -> str:
    if not is_catalog_product_model_query(query):
        return ""
    return (
        "用户询问产品线/型号总览：请按检索到的每一份手册分别列出正文中"
        f"「{_CATALOG_MODEL_MARKER}」一行里的全部型号；"
        "有几份来源含该行就列几份，不得只汇总其中部分来源。"
    )


def build_user_prompt_for_query(query: str) -> str:
    """Merge answer-fidelity rules and profile steering for LightRAG."""
    parts: list[str] = []
    fidelity = build_answer_fidelity_user_prompt(query)
    if fidelity:
        parts.append(fidelity)
    catalog = build_catalog_model_listing_prompt(query)
    if catalog:
        parts.append(catalog)
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
        before = len(docs)
        docs = supplement_catalog_product_model_chunks(
            query, docs, rerank_pool=docs
        )
        if len(docs) > before:
            _stash_catalog_boost_stats(len(docs) - before)
        kept, _report = filter_retrieved_docs_with_report(query, docs)
        return kept

    _wrapped._doc_filter_wrapped = True  # type: ignore[attr-defined]
    ut.apply_rerank_if_enabled = _wrapped  # type: ignore[method-assign]


def install_catalog_rerank_threshold() -> None:
    """Lower ``min_rerank_score`` for product-line catalog queries."""
    import lightrag.utils as ut

    orig = ut.process_chunks_unified
    if getattr(orig, "_catalog_rerank_wrapped", False):
        return

    async def _wrapped(
        query: str,
        unique_chunks: list[dict],
        query_param: Any,
        global_config: dict,
        source_type: str = "mixed",
        chunk_token_limit: int | None = None,
    ):
        prev_min: float | None = None
        if is_catalog_product_model_query(query) and _env_bool(
            "RAG_CATALOG_QUERY_RERANK", True
        ):
            prev_min = float(
                global_config.get("min_rerank_score", _default_min_rerank_score())
            )
            global_config["min_rerank_score"] = catalog_query_min_rerank_score()
        try:
            return await orig(
                query,
                unique_chunks,
                query_param,
                global_config,
                source_type,
                chunk_token_limit,
            )
        finally:
            if prev_min is not None:
                global_config["min_rerank_score"] = prev_min

    _wrapped._catalog_rerank_wrapped = True  # type: ignore[attr-defined]
    ut.process_chunks_unified = _wrapped  # type: ignore[method-assign]


def install_kg_file_path_filter() -> None:
    """Filter mix/local/global KG context by ``file_path`` when a steering profile matches."""
    import lightrag.operate as op

    if getattr(op, "_kg_file_path_filter_installed", False):
        return

    orig_trunc = op._apply_token_truncation
    orig_build_ctx = op._build_query_context

    async def _apply_token_truncation(search_result, query_param, global_config):
        q = (_kg_filter_query.get() or "").strip()
        if q:
            filtered, stats = filter_kg_search_by_query(q, search_result)
            search_result = filtered
            _stash_kg_filter_stats(stats)
        return await orig_trunc(search_result, query_param, global_config)

    async def _build_query_context(query, *args, **kwargs):
        token = _kg_filter_query.set((query or "").strip())
        try:
            return await orig_build_ctx(query, *args, **kwargs)
        finally:
            _kg_filter_query.reset(token)

    op._apply_token_truncation = _apply_token_truncation  # type: ignore[method-assign]
    op._build_query_context = _build_query_context  # type: ignore[method-assign]
    op._kg_file_path_filter_installed = True  # type: ignore[attr-defined]


def install_query_steering_hooks() -> None:
    """Chunk + KG file_path filters for steered queries (idempotent)."""
    install_doc_filter_on_rerank()
    install_catalog_rerank_threshold()
    install_kg_file_path_filter()
