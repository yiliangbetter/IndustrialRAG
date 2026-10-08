"""Scope retrieval to the machine type mentioned in the user query.

Rules come from ``config/query_steering_profiles.json`` (or
``RAG_QUERY_STEERING_PROFILES``). When a profile matches, unrelated manual
PDFs can be dropped after rerank; a report is exposed for the Web UI / logs.

Rerank scores come from CrossEncoder (``pipeline_rerank``). Foreword catalog
chunks (``本手册适用产品型号``) may be merged from KV after rerank or at the
LLM batch when ``RAG_CATALOG_QUERY_BOOST`` is on. Table-matrix sibling chunks
may be merged similarly when ``table_matrix_matches_query`` agrees."""

from __future__ import annotations

import copy
import json
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from iqr_domain_schema import schema as _domain_schema

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROFILES_PATH = _ROOT / "config" / "query_steering_profiles.json"

# Manual step markers (①②…) — strip in user-facing answers, keep wording.
_CIRCLED_STEP_BEFORE_CJK = re.compile(
    r"[\u2460-\u2473\u3251-\u325f\u2776-\u277f\u24ea-\u24ff" r"](?=\s*[\u4e00-\u9fff])"
)


def strip_manual_circled_step_markers(text: str) -> str:
    """Remove circled list prefixes (①②) before Chinese text in answers."""
    if not text or not text.strip():
        return text
    return _CIRCLED_STEP_BEFORE_CJK.sub("", text)


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
                r.get("title") or r.get("path", "") for r in self.removed_sources[:4]
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


class SteeringProfileError(RuntimeError):
    """Machine steering policy is missing or malformed."""


def _validate_profiles(data: Any, source: str) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not data:
        raise SteeringProfileError(
            f"Steering profiles must be a non-empty list: {source}"
        )
    profiles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, profile in enumerate(data):
        if not isinstance(profile, dict):
            raise SteeringProfileError(
                f"Profile #{index + 1} is not an object: {source}"
            )
        profile_id = str(profile.get("id") or "").strip()
        label = str(profile.get("label") or "").strip()
        phrases = profile.get("query_phrases")
        if not profile_id or not label or not isinstance(phrases, list) or not phrases:
            raise SteeringProfileError(
                f"Profile #{index + 1} requires id, label, and query_phrases: {source}"
            )
        if profile_id in seen_ids:
            raise SteeringProfileError(f"Duplicate profile id {profile_id!r}: {source}")
        seen_ids.add(profile_id)
        for key in (
            "query_phrases",
            "query_exclude_if_contains",
            "deny_path_substrings",
        ):
            values = profile.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise SteeringProfileError(
                    f"Profile {profile_id!r} field {key!r} must be a string list: {source}"
                )
        profiles.append(dict(profile))
    return profiles


def _profiles_path() -> Path:
    raw = (os.getenv("RAG_QUERY_STEERING_PROFILES") or "").strip()
    path = Path(raw).expanduser() if raw else _DEFAULT_PROFILES_PATH
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


@lru_cache(maxsize=8)
def _load_profiles_file(path_text: str, mtime_ns: int) -> tuple[dict[str, Any], ...]:
    del mtime_ns  # part of the cache key
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SteeringProfileError(
            f"Cannot load steering profiles {path}: {exc}"
        ) from exc
    return tuple(_validate_profiles(data, str(path)))


def _load_extra_profiles() -> list[dict[str, Any]]:
    raw = (os.getenv("RAG_QUERY_DOC_FILTER_RULES_JSON") or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SteeringProfileError(
            f"RAG_QUERY_DOC_FILTER_RULES_JSON is invalid: {exc}"
        ) from exc
    return _validate_profiles(data, "RAG_QUERY_DOC_FILTER_RULES_JSON")


def load_steering_profiles() -> list[dict[str, Any]]:
    path = _profiles_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise SteeringProfileError(f"Steering profile file is missing: {path}") from exc
    combined = [*_load_profiles_file(str(path), mtime_ns), *_load_extra_profiles()]
    return _validate_profiles(combined, "combined steering profiles")


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", "", (q or "").strip())


def resolve_machine_profile(query: str) -> dict[str, Any] | None:
    """Pick the best-matching machine profile from the question text."""
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


def _path_hits_deny(path: str, deny: list[str]) -> str | None:
    if not path or not deny:
        return None
    hit = next((s for s in deny if s in path), None)
    return hit


def _catalog_model_marker() -> str:
    return _domain_schema.catalog_page_marker


_CATALOG_MODEL_MARKER = _catalog_model_marker()  # backward-compatible export


def _asks_manual_applicability_models(query: str) -> bool:
    """Foreword-style: which product models a named manual applies to."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"适用(?:于)?(?:哪些|什么|哪(?:些|种)|多少).*?(?:型号|机型)|"
            r"(?:手册|说明书).*适用.*?(?:型号|机型)|"
            r"(?:型号|机型).*适用",
            q,
        )
    )


def is_catalog_product_model_query(query: str) -> bool:
    """Broad product-line / model-count questions (not single-machine maintenance)."""
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
    if not (asks_scope and asks_models):
        return False
    if _asks_manual_applicability_models(q):
        return True
    return not resolve_machine_profile(query)


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
    return _catalog_model_marker() in str(doc.get("content") or "")


def _catalog_chunk_relevant_to_query(query: str, doc: dict) -> bool:
    """Keep foreword catalog lines whose path/body overlap query terms."""
    if not _chunk_has_catalog_marker(doc):
        return False
    path = _doc_path(doc)
    profile = resolve_machine_profile(query)
    if profile:
        deny = list(profile.get("deny_path_substrings") or [])
        if _path_hits_deny(path, deny):
            return False
        phrases = profile.get("query_phrases") or []
        if path and phrases:
            pn = path.replace(" ", "")
            if not any(
                str(p).replace(" ", "") in pn for p in phrases if str(p).strip()
            ):
                return False
    terms = _query_discriminative_terms(query)
    if not terms:
        return True
    blob = f"{path} {str(doc.get('content') or '')[:500]}"
    return any(len(term) >= 3 and term in blob for term in terms)


def _load_catalog_chunks_from_storage() -> list[dict]:
    out: list[dict] = []
    for chunk_id, row in _load_text_chunk_map().items():
        if not isinstance(row, dict):
            continue
        content = str(row.get("content") or "")
        if _catalog_model_marker() not in content:
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
    """Ensure each relevant manual's foreword ``本手册适用产品型号`` chunk is present."""
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


def supplement_llm_catalog_chunks(
    query: str,
    llm_docs: list[dict],
    *,
    rerank_pool: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Prepend missing foreword catalog chunks to the LLM batch."""
    ids_before = {_chunk_doc_id(d) for d in llm_docs if _chunk_doc_id(d)}
    merged = supplement_catalog_product_model_chunks(
        query, llm_docs, rerank_pool=rerank_pool
    )
    added = sum(
        1
        for doc in merged
        if _chunk_has_catalog_marker(doc)
        and (cid := _chunk_doc_id(doc))
        and cid not in ids_before
    )
    if added <= 0:
        return llm_docs, 0
    return merged, added


def record_catalog_boost(*, post_rerank: int = 0, llm: int = 0) -> None:
    if post_rerank <= 0 and llm <= 0:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    boost = dict(payload.get("catalog_boost") or {})
    if post_rerank > 0:
        boost["post_rerank_added"] = post_rerank
    if llm > 0:
        boost["llm_added"] = llm
    payload["catalog_boost"] = boost
    _last_filter_report.set(payload)


def supplement_clarify_injection_catalog(
    query: str, injection: dict[str, Any]
) -> dict[str, Any]:
    """Patch a cached clarify injection with missing foreword catalog chunks."""
    if not isinstance(injection, dict):
        return injection
    raw = injection.get("raw_data")
    if not isinstance(raw, dict):
        return injection
    data = raw.get("data")
    if not isinstance(data, dict):
        return injection
    chunks = [row for row in (data.get("chunks") or []) if isinstance(row, dict)]
    supplemented = supplement_catalog_product_model_chunks(
        query, chunks, rerank_pool=chunks
    )
    ids_before = {_chunk_doc_id(d) for d in chunks if _chunk_doc_id(d)}
    added = any(
        _chunk_has_catalog_marker(doc)
        and (cid := _chunk_doc_id(doc))
        and cid not in ids_before
        for doc in supplemented
    )
    if not added:
        return injection
    new_raw = copy.deepcopy(raw)
    new_raw.setdefault("data", {})["chunks"] = supplemented
    from raganything.clarify_context import scope_kg_to_llm_chunks  # noqa: WPS433

    context_str, scoped_raw = scope_kg_to_llm_chunks(
        new_raw, str(injection.get("context_str") or "")
    )
    return {
        "context_str": context_str or str(injection.get("context_str") or ""),
        "raw_data": scoped_raw or new_raw,
    }


def build_catalog_model_listing_prompt(query: str) -> str:
    if not is_catalog_product_model_query(query):
        return ""
    return (
        "用户询问产品线/型号总览：请按检索到的每一份手册分别列出正文中"
        f"「{_catalog_model_marker()}」一行里的全部型号；"
        "有几份来源含该行就列几份，不得只汇总其中部分来源。"
    )


def _query_discriminative_terms(query: str) -> list[str]:
    from raganything.utils import discriminative_terms  # noqa: WPS433

    return discriminative_terms(query, min_len=3)


def table_filter_needle(query: str) -> str | None:
    """Filter value the user wants to match in tabular rows (from the question wording)."""
    q = (query or "").strip()
    if not q:
        return None
    m = re.search(r"使用\s*([^，。？,；;\n]+?)(?:[？?]|$)", q)
    if m:
        needle = m.group(1).strip()
        if needle:
            return needle
    for hint in re.findall(r'[「"\u201c]([^」"\u201d]+)[」"\u201d]', q):
        if hint.strip():
            return hint.strip()
    return None


def _table_filter_min_matching_rows() -> int:
    raw = os.getenv("RAG_TABLE_FILTER_MIN_ROWS") or "2"
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _table_row_matches_filter(row_html: str, needle: str) -> bool:
    tds = [
        td.strip()
        for td in re.findall(r"<td[^>]*>([^<]+)</td>", row_html, re.IGNORECASE)
    ]
    if not tds or needle not in row_html:
        return False
    if needle in tds[-1]:
        return True
    if len(tds) >= 2:
        return any(needle in td for td in tds[1:])
    return False


def detect_table_filter_signal(query: str, text: str) -> bool:
    """True when retrieved context has multiple HTML table rows matching the filter value."""
    needle = table_filter_needle(query)
    if not needle or not text or "<tr" not in text.lower():
        return False
    rows = re.findall(r"<tr>.*?</tr>", text, re.IGNORECASE | re.DOTALL)
    matches = sum(1 for row in rows if _table_row_matches_filter(row, needle))
    return matches >= _table_filter_min_matching_rows()


def is_table_filter_listing_query(query: str, context: str | None = None) -> bool:
    """Confirmed table-filter listing — requires retrieval context unless explicitly passed."""
    if not (query or "").strip():
        return False
    if context is None:
        return False
    return detect_table_filter_signal(query, context)


def build_table_filter_listing_prompt(query: str) -> str:
    needle = table_filter_needle(query) or "问句中的筛选条件"
    return (
        "用户问的是对检索 context 中表格行的筛选与列举。"
        f"在 HTML 表格<table>/<tr>/<td>行中，找出与「{needle}」匹配的全部行"
        "（以各行文字所示或对应表头列为准；行内容以表头为准）。"
        "列出这些行的对象/部位/部件名称（取能识别的第一列，以表头为准）。"
        "请把 context 中所有符合条件的行列出，不要遗漏。"
        "回答用简洁列表，每行只写名称，不要展开其它列（周期、内容、方式等）。"
        "不要引用非表格的正文或其它材料来组合答案。"
    )


def _append_table_filter_user_prompt(query_param: Any, query: str) -> None:
    if query_param is None:
        return
    extra = build_table_filter_listing_prompt(query)
    if not extra:
        return
    existing = str(getattr(query_param, "user_prompt", None) or "").strip()
    if extra in existing:
        return
    query_param.user_prompt = f"{existing}\n\n{extra}".strip() if existing else extra


def _manual_chunk_min_chars() -> int:
    raw = os.getenv("RAG_QUERY_ALIGN_MIN_CHARS") or "22"
    try:
        return max(8, int(raw))
    except ValueError:
        return 22


def _query_subject_terms(query: str) -> list[str]:
    """Discriminative terms with resolved machine-profile wording de-emphasized."""
    terms = list(dict.fromkeys(_query_discriminative_terms(query)))
    profile = resolve_machine_profile(query)
    if profile:
        profile_phrases = [
            str(p).replace(" ", "")
            for p in (profile.get("query_phrases") or [])
            if str(p).strip()
        ]

        def _profile_boilerplate(term: str) -> bool:
            return any(
                (term in phrase or phrase in term)
                for phrase in profile_phrases
                if len(phrase) >= 3 and len(term) >= 3
            )

        subject = [t for t in terms if not _profile_boilerplate(t)]
        if len(subject) >= 2:
            terms = subject
    return sorted(terms, key=len, reverse=True)[:20]


def _query_focus_terms(query: str) -> list[str]:
    """Longer subject spans from the question (no domain phrase lists)."""
    terms = _query_subject_terms(query)
    if not terms:
        return []
    long_terms = [t for t in terms if len(t) >= 4]
    if long_terms:
        return sorted(long_terms, key=len, reverse=True)[:16]
    return terms[:16]


def _text_chunks_store_path() -> Path:
    return _rag_storage_dir() / "kv_store_text_chunks.json"


def _matrix_store_path() -> Path:
    return _rag_storage_dir()


def _rag_storage_dir() -> Path:
    for env_key in ("RAG_WEB_WORKING_DIR", "WORKING_DIR"):
        raw = (os.getenv(env_key) or "").strip()
        if raw:
            path = Path(raw).expanduser()
            return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()
    try:
        from client_paths import get_rag_storage_dir  # noqa: WPS433

        return Path(get_rag_storage_dir()).resolve()
    except (ImportError, OSError):
        return _ROOT / "data" / "rag_storage"


def _chunk_doc_id(doc: dict) -> str:
    for key in ("id", "chunk_id"):
        val = doc.get(key)
        if val:
            return str(val)
    return ""


def _load_table_matrix_from_storage() -> list[Any]:
    from raganything.table_matrix import load_matrix_store

    store = load_matrix_store(_matrix_store_path())
    return list(store.values())


def _allowed_manual_basenames(docs: list[dict]) -> set[str]:
    out: set[str] = set()
    for doc in docs:
        path = _doc_path(doc)
        if path:
            out.add(_basename(path))
    return out


def _table_matrix_chunk_relevant(
    query: str,
    record: Any,
    *,
    allowed_basenames: set[str],
) -> bool:
    from raganything.table_matrix import table_matrix_matches_query

    fp = str(getattr(record, "file_path", "") or "")
    if allowed_basenames:
        bp = _basename(fp)
        if not any(ab and (ab in fp or ab in bp) for ab in allowed_basenames):
            return False
    elif fp:
        terms = _query_discriminative_terms(query)
        if terms and not any(t in fp for t in terms):
            return False
    deny, _profile = active_deny_substrings(query)
    if _path_hits_deny(fp, deny):
        return False
    return table_matrix_matches_query(query, record)


def _load_text_chunk_map() -> dict[str, dict]:
    path = _text_chunks_store_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def collect_table_matrix_candidate_docs(
    query: str,
    docs: list[dict],
    *,
    context_docs: list[dict] | None = None,
    exclude_doc_ids: list[dict] | None = None,
) -> list[dict]:
    """Table chunks from ``kv_store_table_matrix`` not already in ``exclude_doc_ids``."""
    from raganything.table_matrix import table_matrix_query_boost_enabled

    if not table_matrix_query_boost_enabled():
        return []

    scope = list(docs) + list(context_docs or [])
    allowed = _allowed_manual_basenames(scope)
    seen = {
        _chunk_doc_id(d)
        for d in (exclude_doc_ids if exclude_doc_ids is not None else docs)
        if _chunk_doc_id(d)
    }
    chunk_map = _load_text_chunk_map()
    out: list[dict] = []

    for record in _load_table_matrix_from_storage():
        if not _table_matrix_chunk_relevant(
            query, record, allowed_basenames=allowed if allowed else set()
        ):
            continue
        fp = str(getattr(record, "file_path", "") or "")
        for chunk_id in getattr(record, "chunk_ids", None) or []:
            cid = str(chunk_id or "").strip()
            if not cid or cid in seen:
                continue
            row = chunk_map.get(cid)
            if not isinstance(row, dict):
                continue
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            seen.add(cid)
            out.append(
                {
                    "content": content,
                    "id": cid,
                    "file_path": fp or str(row.get("file_path") or ""),
                }
            )
    return out


def supplement_table_matrix_before_rerank(
    query: str, retrieved_docs: list[dict]
) -> tuple[list[dict], int]:
    """Merge matrix-linked table chunks into the pool before CrossEncoder rerank."""
    added = collect_table_matrix_candidate_docs(query, retrieved_docs)
    if not added:
        return retrieved_docs, 0
    return list(retrieved_docs) + added, len(added)


def supplement_llm_table_matrix_chunks(
    query: str,
    llm_docs: list[dict],
    *,
    rerank_pool: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Ensure sibling table chunks reach the LLM batch after rerank truncation."""
    added = collect_table_matrix_candidate_docs(
        query,
        llm_docs,
        context_docs=rerank_pool,
        exclude_doc_ids=llm_docs,
    )
    if not added:
        return llm_docs, 0
    seen = {_chunk_doc_id(d) for d in llm_docs if _chunk_doc_id(d)}
    prepend: list[dict] = []
    for doc in added:
        cid = _chunk_doc_id(doc)
        if cid and cid not in seen:
            seen.add(cid)
            prepend.append(doc)
    if not prepend:
        return llm_docs, 0
    return prepend + list(llm_docs), len(prepend)


def record_table_matrix_boost(*, pre_rerank: int = 0, llm: int = 0) -> None:
    if pre_rerank <= 0 and llm <= 0:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    boost = dict(payload.get("table_matrix_boost") or {})
    if pre_rerank > 0:
        boost["pre_rerank_added"] = pre_rerank
    if llm > 0:
        boost["llm_added"] = llm
    payload["table_matrix_boost"] = boost
    _last_filter_report.set(payload)


def _load_manual_chunks_for_paths(
    allowed_paths: set[str],
    deny: list[str],
) -> list[dict]:
    path = _text_chunks_store_path()
    if not path.is_file() or not allowed_paths:
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    min_chars = _manual_chunk_min_chars()
    allowed_basenames = {_basename(p) for p in allowed_paths if p}
    out: list[dict] = []
    for chunk_id, row in raw.items():
        if not isinstance(row, dict):
            continue
        fp = str(row.get("file_path") or "")
        if not fp or _path_hits_deny(fp, deny):
            continue
        bp = _basename(fp)
        if not any(
            ab and (ab in fp or ab in bp or bp in ab) for ab in allowed_basenames
        ):
            continue
        content = str(row.get("content") or "")
        if len(content.strip()) < min_chars:
            continue
        doc = dict(row)
        doc.setdefault("content", content)
        doc.setdefault("id", chunk_id)
        out.append(doc)
    return out


def _is_cross_manual_listing_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        re.search(r"部件|零件|组件", q)
        and re.search(r"哪些|有什么|有哪|各自|所有机型", q)
    )


def build_query_subject_chunk_prompt(query: str) -> str:
    """Generic LLM hint: prefer chunks that literally contain query wording."""
    if not _env_bool("RAG_QUERY_SUBJECT_STEER", True):
        return ""
    if len(_query_focus_terms(query)) < 2:
        return ""
    return (
        "若某条正文 chunk 的字面表述与用户问题中的专有名词或动作对象一致，"
        "须优先引用该 chunk 作答；不要用仅部分用词相近的其它段落替代。"
        "不得声称手册未提及，而检索 chunk 中已出现相同对象或步骤。"
    )


def build_concise_fact_answer_prompt(query: str) -> str:
    """Single-fact questions (tool/cycle/brand): avoid neighbor maintenance sections."""
    if not _env_bool("RAG_QUERY_CONCISE_FACT", True):
        return ""
    q = (query or "").strip()
    if re.search(r"步骤|哪些|几种|列举|分别", q):
        return ""
    if re.search(
        r"什么工具|用什么工具|用什么[^？?]*清|多长时间|多久|多少|哪种|哪个品牌",
        q,
    ):
        return (
            "用户只问单一事实（工具名/周期/品牌等）。"
            "只回答所问内容；不要展开同页或其它条目的保养步骤、更换流程或邻节检查"
            "（例如问压带轮残胶工具时，勿写刮刀片检查/更换等其它保养条目）。"
        )
    return ""


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


def build_maintenance_supply_listing_prompt(query: str) -> str:
    """Q13-style: list every supply category named in manual chunks (e.g. grease types)."""
    if not _env_bool("RAG_QUERY_MAINTENANCE_SUPPLY_LIST", True):
        return ""
    q = (query or "").strip()
    if not re.search(r"季度保养|半年保养|保养", q):
        return ""
    if not re.search(r"几种|哪些|准备|需要", q):
        return ""
    if not re.search(r"润滑脂|润滑油|油品|工具|材料", q):
        return ""
    return (
        "用户问保养前需准备哪些指定品类（如润滑脂等）。"
        "须据正文列举正文已出现的全部相关品类，并说明各自适用部位；"
        "若正文区分通用品类与专用品类，须一并列出，不得只答其一。"
        "仅据 chunk 正文作答；References 只列实际引用的手册。"
    )


def build_cross_manual_listing_answer_prompt(query: str) -> str:
    """Cross-manual part listings: group by machine, cite all manuals used."""
    if not _is_cross_manual_listing_query(query):
        return ""
    return (
        "跨来源列举题：按来源或机型分组，保持正文条目的原始粒度，独立条目不得合并。"
        "只列正文 chunk 明确支持的对象与要求，不得依据近义词、知识图谱或其它来源类推；"
        "正文未提及时明确写未提及。每条须引用其实际依据，References 列出全部使用来源。"
    )


def build_query_user_prompt(query: str) -> str:
    """Merge optional LLM hints for LightRAG ``user_prompt``."""
    parts: list[str] = []
    catalog = build_catalog_model_listing_prompt(query)
    if catalog:
        parts.append(catalog)
    subject = build_query_subject_chunk_prompt(query)
    if subject:
        parts.append(subject)
    concise = build_concise_fact_answer_prompt(query)
    if concise:
        parts.append(concise)
    supply = build_maintenance_supply_listing_prompt(query)
    if supply:
        parts.append(supply)
    cross_list = build_cross_manual_listing_answer_prompt(query)
    if cross_list:
        parts.append(cross_list)
    steer = build_steering_user_prompt(query)
    if steer:
        parts.append(steer)
    return "\n\n".join(parts)


def build_user_prompt_for_query(query: str) -> str:
    """Alias used by ``rag_pipeline_parse_graph_chat`` (same as ``build_query_user_prompt``)."""
    return build_query_user_prompt(query)


def build_steering_user_prompt(query: str) -> str:
    """Per-query hint for the LLM (not per-machine ``.env`` entries)."""
    if not _env_bool("RAG_QUERY_KG_STEERING", True):
        return ""
    profile = resolve_machine_profile(query)
    if not profile:
        return ""
    label = str(profile.get("label") or "")
    return (
        f"用户问题针对「{label}」。只引用与该机型对应手册的正文 chunk；"
        "若知识图谱实体描述与其它机型手册合并后冲突，以 chunk 正文为准。"
    )


def install_doc_filter_on_rerank() -> None:
    """Deprecated: rerank post-hooks removed; CrossEncoder scores are used as-is."""


def install_catalog_rerank_threshold() -> None:
    """Lower ``min_rerank_score`` for product-line catalog queries."""
    import lightrag.operate as op
    import lightrag.utils as ut

    orig = ut.process_chunks_unified
    if getattr(orig, "_catalog_rerank_wrapped", False):
        return

    async def _wrapped(*args: Any, **kwargs: Any):
        query = args[0] if args else kwargs.get("query", "")
        global_config = args[3] if len(args) > 3 else kwargs.get("global_config", {})
        prev_min: float | None = None
        if (
            isinstance(global_config, dict)
            and is_catalog_product_model_query(str(query))
            and _env_bool("RAG_CATALOG_QUERY_RERANK", True)
        ):
            prev_min = float(
                global_config.get("min_rerank_score", _default_min_rerank_score())
            )
            global_config["min_rerank_score"] = catalog_query_min_rerank_score()
        try:
            return await orig(*args, **kwargs)
        finally:
            if prev_min is not None:
                global_config["min_rerank_score"] = prev_min

    _wrapped._catalog_rerank_wrapped = True  # type: ignore[attr-defined]
    ut.process_chunks_unified = _wrapped  # type: ignore[method-assign]
    op.process_chunks_unified = _wrapped  # type: ignore[method-assign]


def install_query_context_hooks() -> None:
    """Table-filter user_prompt when retrieval context confirms a tabular listing query."""
    import lightrag.operate as op

    orig = op._build_query_context
    if getattr(orig, "_query_context_hooks_wrapped", False):
        return

    async def _wrapped(*args: Any, **kwargs: Any):
        result = await orig(*args, **kwargs)
        if result is None:
            return result
        query = args[0] if args else str(kwargs.get("query") or "")
        query_param = kwargs.get("query_param")
        if query_param is None and len(args) >= 8:
            query_param = args[7]
        if isinstance(query, str) and query.strip() and query_param is not None:
            ctx = str(getattr(result, "context", None) or "")
            if detect_table_filter_signal(query, ctx):
                _append_table_filter_user_prompt(query_param, query)
        return result

    _wrapped._query_context_hooks_wrapped = True  # type: ignore[attr-defined]
    op._build_query_context = _wrapped  # type: ignore[method-assign]


def install_query_steering_hooks() -> None:
    """Catalog rerank threshold + LLM user_prompt hooks (idempotent)."""
    install_catalog_rerank_threshold()
    install_query_context_hooks()
