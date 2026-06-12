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
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

# Manual step markers (①②…) — strip in user-facing answers, keep wording.
_CIRCLED_STEP_BEFORE_CJK = re.compile(
    r"[\u2460-\u2473\u3251-\u325f\u2776-\u277f\u24ea-\u24ff"
    r"](?=\s*[\u4e00-\u9fff])"
)


def strip_manual_circled_step_markers(text: str) -> str:
    """Remove circled list prefixes (①②) before Chinese text in answers."""
    if not text or not text.strip():
        return text
    return _CIRCLED_STEP_BEFORE_CJK.sub("", text)


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


def _path_hits_deny(path: str, deny: list[str]) -> str | None:
    if not path or not deny:
        return None
    hit = next((s for s in deny if s in path), None)
    return hit


_CATALOG_MODEL_MARKER = "本手册适用产品型号"


def _query_discriminative_terms(query: str) -> list[str]:
    from raganything.utils import discriminative_terms  # noqa: WPS433

    return discriminative_terms(query, min_len=3)


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
    if resolve_machine_profile(query):
        return False
    return True


def _default_min_rerank_score() -> float:
    if (os.getenv("RAG_USE_CLARIFY_UPPER_AS_MIN_RERANK") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        raw = os.getenv("QUERY_SCORE_THRESHOLD_UPPER") or "0.45"
        try:
            return float(raw)
        except ValueError:
            return 0.45
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
    path = _doc_path(doc)
    profile = resolve_machine_profile(query)
    if profile:
        deny = list(profile.get("deny_path_substrings") or [])
        if _path_hits_deny(path, deny):
            return False
        phrases = profile.get("query_phrases") or []
        if path and phrases:
            pn = path.replace(" ", "")
            if not any(str(p).replace(" ", "") in pn for p in phrases if str(p).strip()):
                return False
    terms = _query_discriminative_terms(query)
    if not terms:
        return True
    blob = f"{path} {str(doc.get('content') or '')[:500]}"
    return any(len(term) >= 3 and term in blob for term in terms)


def _load_catalog_chunks_from_storage() -> list[dict]:
    try:
        from client_paths import get_rag_storage_dir  # noqa: WPS433

        store = Path(get_rag_storage_dir())
    except Exception:
        store = _ROOT / "data" / "rag_storage"
    path = store / "kv_store_text_chunks.json"
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


def _stash_catalog_boost_stats(count: int) -> None:
    if count <= 0:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["catalog_boost"] = {"chunks_added": count}
    _last_filter_report.set(payload)


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
    tds = [td.strip() for td in re.findall(r"<td[^>]*>([^<]+)</td>", row_html, re.I)]
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
    rows = re.findall(r"<tr>.*?</tr>", text, re.I | re.DOTALL)
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


def _matrix_store_path() -> Path:
    try:
        from client_paths import get_rag_storage_dir  # noqa: WPS433

        return Path(get_rag_storage_dir())
    except Exception:
        return _ROOT / "data" / "rag_storage"


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
    from raganything.utils import discriminative_terms

    fp = str(getattr(record, "file_path", "") or "")
    if allowed_basenames:
        bp = _basename(fp)
        if not any(ab and (ab in fp or ab in bp) for ab in allowed_basenames):
            return False
    elif fp:
        terms = discriminative_terms(query, min_len=3)
        if terms and not any(t in fp for t in terms):
            return False
    deny, _profile = active_deny_substrings(query)
    if _path_hits_deny(fp, deny):
        return False
    return table_matrix_matches_query(query, record)


def supplement_table_matrix_chunks(
    query: str,
    docs: list[dict],
    *,
    rerank_pool: list[dict] | None = None,
) -> list[dict]:
    """Boost table chunks when row-matrix term overlap matches the query."""
    from raganything.table_matrix import table_matrix_query_boost_enabled

    if not table_matrix_query_boost_enabled():
        return docs

    allowed = _allowed_manual_basenames(docs + (rerank_pool or []))
    seen_chunk: set[str] = set()
    boosted: list[dict] = []

    def add_chunk(chunk_id: str, file_path: str) -> None:
        if not chunk_id or chunk_id in seen_chunk:
            return
        seen_chunk.add(chunk_id)
        boosted.append(
            {
                "content": "",
                "id": chunk_id,
                "file_path": file_path,
                "rerank_score": 0.98,
            }
        )

    for record in _load_table_matrix_from_storage():
        if not _table_matrix_chunk_relevant(
            query, record, allowed_basenames=allowed if allowed else set()
        ):
            continue
        fp = str(record.file_path or "")
        for chunk_id in record.chunk_ids:
            add_chunk(chunk_id, fp)

    if not boosted:
        return docs

    try:
        path = _matrix_store_path() / "kv_store_text_chunks.json"
        chunk_map = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {}
        )
        for doc in boosted:
            cid = str(doc.get("id") or "")
            row = chunk_map.get(cid)
            if isinstance(row, dict) and row.get("content"):
                doc["content"] = row["content"]
    except (OSError, json.JSONDecodeError):
        pass

    return boosted + [d for d in docs if str(d.get("id") or "") not in seen_chunk]


def _stash_matrix_boost_stats(count: int) -> None:
    if count <= 0:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["table_matrix_boost"] = {"chunks_added": count}
    _last_filter_report.set(payload)


def _query_align_boost_enabled() -> bool:
    return _env_bool("RAG_QUERY_ALIGN_BOOST", True)


def _query_align_min_score() -> float:
    raw = os.getenv("RAG_QUERY_ALIGN_MIN") or "0.12"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.12


def _query_align_rerank_score() -> float:
    raw = os.getenv("RAG_QUERY_ALIGN_RERANK") or "0.94"
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.94


def _query_align_top_n() -> int:
    raw = os.getenv("RAG_QUERY_ALIGN_TOP_N") or "2"
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _query_align_min_chars() -> int:
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


def _query_subject_anchor(query: str) -> str:
    """Longest remaining CJK span after stripping matched machine-profile phrases."""
    q = (query or "").strip()
    profile = resolve_machine_profile(query)
    if profile:
        for phrase in sorted(
            (str(p) for p in (profile.get("query_phrases") or [])),
            key=len,
            reverse=True,
        ):
            pn = phrase.replace(" ", "")
            if pn and pn in q.replace(" ", ""):
                q = q.replace(phrase, " ").replace(pn, " ")
    runs = re.findall(r"[\u4e00-\u9fff]+", q)
    return max(runs, key=len, default="")


def _query_anchor_alignment(query: str, content: str) -> float:
    from raganything.utils import text_term_alignment  # noqa: WPS433

    anchor = _query_subject_anchor(query)
    if len(anchor) < 4:
        return 1.0
    return text_term_alignment(anchor, content, min_len=3)


def _query_anchor_min_align() -> float:
    raw = os.getenv("RAG_QUERY_ALIGN_ANCHOR_MIN") or "0.18"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.18


def _chunk_query_alignment_score(query: str, content: str) -> float:
    from raganything.utils import text_term_alignment_symmetric  # noqa: WPS433

    body = content.strip()
    if len(body) < _query_align_min_chars():
        return 0.0
    sym = text_term_alignment_symmetric(query, body, min_len=3)
    short_cap = max(28, len(query) // 6)
    if len(body) < short_cap:
        sym *= len(body) / short_cap
    subject = _query_subject_terms(query)
    if not subject:
        return sym
    hits = sum(1 for term in subject if term in body)
    hit_ratio = hits / len(subject)
    longest = max((len(term) for term in subject if term in body), default=0)
    return sym + 0.55 * hit_ratio + 0.02 * longest


def _chunk_literal_focus_hit(query: str, content: str) -> bool:
    subject = _query_subject_terms(query)
    if not subject:
        return True
    return any(term in content for term in subject)


def _text_chunks_store_path() -> Path:
    try:
        from client_paths import get_rag_storage_dir  # noqa: WPS433

        store = Path(get_rag_storage_dir())
    except Exception:
        store = _ROOT / "data" / "rag_storage"
    return store / "kv_store_text_chunks.json"


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
    allowed_basenames = {_basename(p) for p in allowed_paths if p}
    out: list[dict] = []
    for chunk_id, row in raw.items():
        if not isinstance(row, dict):
            continue
        fp = str(row.get("file_path") or "")
        if not fp or _path_hits_deny(fp, deny):
            continue
        bp = _basename(fp)
        if not any(ab and (ab in fp or ab in bp or bp in ab) for ab in allowed_basenames):
            continue
        content = str(row.get("content") or "")
        if len(content.strip()) < _query_align_min_chars():
            continue
        doc = dict(row)
        doc.setdefault("content", content)
        doc.setdefault("id", chunk_id)
        out.append(doc)
    return out


def _apply_query_alignment_rerank_bonus(query: str, docs: list[dict]) -> list[dict]:
    """Nudge rerank order toward chunks whose body literally overlaps query terms."""
    if not docs:
        return docs
    rescored: list[tuple[float, dict]] = []
    for doc in docs:
        content = str(doc.get("content") or "")
        align = _chunk_query_alignment_score(query, content)
        boosted = dict(doc)
        base = float(boosted.get("rerank_score") or 0)
        boosted["rerank_score"] = base + min(0.3, align * 0.45)
        rescored.append((align, boosted))
    rescored.sort(
        key=lambda pair: (float(pair[1].get("rerank_score") or 0), pair[0]),
        reverse=True,
    )
    return [doc for _, doc in rescored]


def supplement_query_aligned_chunks(
    query: str,
    docs: list[dict],
    *,
    rerank_pool: list[dict] | None = None,
) -> list[dict]:
    """Promote chunks whose body literally matches query discriminative terms."""
    if not _query_align_boost_enabled():
        return docs
    focus = _query_focus_terms(query)
    if len(focus) < 2:
        return docs

    deny, _profile = active_deny_substrings(query)
    allowed_paths = {
        p for p in (_doc_path(d) for d in list(docs) + list(rerank_pool or [])) if p
    }
    if not allowed_paths:
        return _apply_query_alignment_rerank_bonus(query, docs)

    candidates: dict[str, dict] = {}
    for doc in list(docs) + list(rerank_pool or []):
        cid = str(doc.get("id") or "")
        if cid:
            candidates[cid] = doc

    for doc in _load_manual_chunks_for_paths(
        allowed_paths,
        deny,
    ):
        cid = str(doc.get("id") or "")
        if cid and cid not in candidates:
            candidates[cid] = doc

    min_score = _query_align_min_score()
    scored: list[tuple[float, dict]] = []
    for doc in candidates.values():
        content = str(doc.get("content") or "")
        score = _chunk_query_alignment_score(query, content)
        if score < min_score:
            continue
        if not _chunk_literal_focus_hit(query, content) and score < min_score * 2:
            continue
        if _query_anchor_alignment(query, content) < _query_anchor_min_align():
            continue
        scored.append((score, doc))

    if not scored:
        return _apply_query_alignment_rerank_bonus(query, docs)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_n = _query_align_top_n()
    boost_score = _query_align_rerank_score()
    boosted_ids: set[str] = set()
    merged: list[dict] = []
    for score, doc in scored[:top_n]:
        cid = str(doc.get("id") or "")
        if cid in boosted_ids:
            continue
        boosted_ids.add(cid)
        boosted = dict(doc)
        boosted["rerank_score"] = max(
            float(boosted.get("rerank_score") or 0),
            boost_score,
        )
        merged.append(boosted)

    if not merged:
        return _apply_query_alignment_rerank_bonus(query, docs)

    rest = [
        d for d in _apply_query_alignment_rerank_bonus(query, docs)
        if str(d.get("id") or "") not in boosted_ids
    ]
    return merged + rest


def _stash_query_align_boost_stats(count: int) -> None:
    if count <= 0:
        return
    prev = _last_filter_report.get()
    payload = dict(prev) if isinstance(prev, dict) else {}
    payload["query_align_boost"] = {"chunks_added": count}
    _last_filter_report.set(payload)


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


def build_catalog_model_listing_prompt(query: str) -> str:
    if not is_catalog_product_model_query(query):
        return ""
    return (
        "用户询问产品线/型号总览：请按检索到的每一份手册分别列出正文中"
        f"「{_CATALOG_MODEL_MARKER}」一行里的全部型号；"
        "有几份来源含该行就列几份，不得只汇总其中部分来源。"
    )


def build_query_user_prompt(query: str) -> str:
    """Merge catalog listing + profile steering hints for LightRAG ``user_prompt``."""
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
        before_catalog = len(docs)
        docs = supplement_catalog_product_model_chunks(
            query, docs, rerank_pool=docs
        )
        if len(docs) > before_catalog:
            _stash_catalog_boost_stats(len(docs) - before_catalog)
        before_matrix = len(docs)
        docs = supplement_table_matrix_chunks(query, docs, rerank_pool=docs)
        if len(docs) > before_matrix:
            _stash_matrix_boost_stats(len(docs) - before_matrix)
        before_align = len(docs)
        docs = supplement_query_aligned_chunks(query, docs, rerank_pool=docs)
        if len(docs) > before_align:
            _stash_query_align_boost_stats(len(docs) - before_align)
        kept, _report = filter_retrieved_docs_with_report(query, docs)
        return kept

    _wrapped._doc_filter_wrapped = True  # type: ignore[attr-defined]
    ut.apply_rerank_if_enabled = _wrapped  # type: ignore[method-assign]


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
        if (
            isinstance(query, str)
            and query.strip()
            and query_param is not None
        ):
            ctx = str(getattr(result, "context", None) or "")
            if detect_table_filter_signal(query, ctx):
                _append_table_filter_user_prompt(query_param, query)
        return result

    _wrapped._query_context_hooks_wrapped = True  # type: ignore[attr-defined]
    op._build_query_context = _wrapped  # type: ignore[method-assign]


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


def install_query_steering_hooks() -> None:
    """Chunk file_path filter + catalog/table query hooks (idempotent)."""
    install_doc_filter_on_rerank()
    install_catalog_rerank_threshold()
    install_query_context_hooks()
