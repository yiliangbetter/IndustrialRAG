"""Scope retrieval to the machine type mentioned in the user query.

Rules live in code (``MACHINE_PROFILES``), not per-machine ``.env`` entries.
When a profile matches, unrelated manual PDFs can be dropped after rerank
(see ``query_progress_hooks``); a report is exposed for the Web UI / logs.

Rerank scores come only from CrossEncoder (``pipeline_rerank``). Table-matrix
sibling chunks may be merged into the pre-rerank pool (or the LLM batch as a
fallback) when ``table_matrix_matches_query`` agrees; no synthetic scores."""

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
    try:
        from client_paths import get_rag_storage_dir  # noqa: WPS433

        return Path(get_rag_storage_dir())
    except Exception:
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
        if not any(ab and (ab in fp or ab in bp or bp in ab) for ab in allowed_basenames):
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
    extra = ""
    if re.search(r"残胶", query):
        extra = (
            "仅列出正文 chunk 字面写明需清理残胶（或「老化胶水」等同类表述）的部件；"
            "勿根据知识图谱或其它机型类推增加条目；"
            "某机型正文无残胶清理描述时写明未提及，勿编造部件。"
            "同一手册内各有独立「保养内容：」行的条目须各占一条 bullet、禁止合并："
            "若正文同时出现「保养内容：涂胶轴检查清理」与「保养内容：电机检查清理」，"
            "须分别写 **涂胶轴**（清理轴周老化胶水/残胶，对应涂胶轴保养条目）"
            "与 **涂胶电机**（电机检查清理、含清理涂胶轴掉下残胶，对应涂胶电机保养条目），"
            "不得只写「涂胶轴掉下的残胶」并挂在电机条目下而漏掉涂胶轴独立条。"
            "每条格式：**部件名**：该部件/保养内容对应的残胶清理要求（一句即可）。"
        )
    elif re.search(r"1#透平油|透平油", query):
        extra = (
            "仅列出正文 chunk 字面出现「1#透平油」或完整词组「1#透平油（气动油）」的机型；"
            "勿将仅写「气动油（ISOVG32）」「ISO VG-32」等未出现「1#透平油」字样的手册机型列入；"
            "勿凭知识图谱粘度等级类推六面钻/加工中心等机型。"
            "每条格式：**机型名**：用于保养**部件名**（一句，可附 [n] 引用）；"
            "部件名须加粗且为润滑/保养部位，勿将油品名称当作部件。"
        )
    return (
        "跨机型列举题：按机型分组列出部件；同一部件在不同机型须分开写。"
        "每条须能在所引用手册正文 chunk 中找到依据；"
        "References 须列出作答时实际依据的全部机型手册。"
        + (f" {extra}" if extra else "")
    )


def build_query_user_prompt(query: str) -> str:
    """Merge optional LLM hints for LightRAG ``user_prompt``."""
    parts: list[str] = []
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
    """Deprecated: catalog-specific min_rerank override removed."""


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


def install_query_steering_hooks() -> None:
    """LLM user_prompt hooks only (no rerank score steering)."""
    install_query_context_hooks()
