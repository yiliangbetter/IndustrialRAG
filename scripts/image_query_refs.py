"""Lightweight PDF image refs for text-only ingest + Q&A display (Plan B)."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from raganything.utils import (
    _bbox_center,
    build_image_ref_block,
    flatten_image_refs_for_skip_multimodal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_image_ref_block",
    "flatten_image_refs_for_skip_multimodal",
    "extract_image_refs_from_context",
    "images_for_api",
    "query_wants_kb_images",
    "retrieval_supports_images",
    "encode_media_token",
    "decode_media_token",
    "is_safe_media_path",
    "normalize_context_for_image_parse",
    "supplement_refs_from_content_lists",
]

_IMAGE_EXT_GROUP = r"(?:jpg|jpeg|png|gif|webp|bmp|tif|tiff)"
_IMAGE_PATH_RE = re.compile(
    rf"图片路径[：:]\s*([^\n]+?\.{_IMAGE_EXT_GROUP})"
    rf"|Image Path:\s*([^\n]+?\.{_IMAGE_EXT_GROUP})",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"页码[：:]\s*(\d+)", re.IGNORECASE)
_CAPTION_RE = re.compile(r"图注[：:]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"脚注[：:]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CONTEXT_RE = re.compile(r"关联正文[：:]\s*(.+?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL)

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})

_MAINTENANCE_ANCHOR_PATTERNS = (
    r"传动丝杆检查润滑[^\n]{0,24}",
    r"传动丝杆保养",
    r"传动丝杆加润滑(?:脂)?",
    r"3\.1\.5\s*传动丝杆保养",
    r"保养内容：传动丝杆加润滑脂",
)

_NEG_FOR_TRANSMISSION_SCREW = (
    "进料丝杆",
    "滚珠丝杆",
    "开槽刀",
    "开槽",
    "整机清洁",
    "输送电机",
    "辅助进料",
    "抛光",
    "涂胶",
    "进料部分",
    "链条打黄油",
    "伺服电机",
    "变速箱",
)

_BBOX_MATCH_MAX_DIST = 900.0

_CHITCHAT_ONLY_PATTERNS = (
    r"你好啊?",
    r"您好啊?",
    r"嗨+",
    r"hello+",
    r"hi+",
    r"hey+",
    r"在吗",
    r"在不在",
    r"早上好",
    r"下午好",
    r"晚上好",
    r"谢谢|感谢|多谢",
    r"你是谁",
    r"你是啥",
    r"你叫什么",
    r"介绍一下你自己",
    r"介绍你自己",
    r"自我介绍",
    r"你能做什么",
    r"你会什么",
    r"你能帮我什么",
    r"帮助",
    r"怎么用你",
    r"如何使用",
)

_QUERY_TERM_STOP = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "为什么",
        "为何",
        "请问",
        "是否",
        "可以",
        "有没有",
        "哪些",
        "那个",
        "这个",
        "一下",
        "告诉",
        "介绍",
        "关于",
        "问题",
        "答案",
        "多少",
        "多久",
        "需要",
        "哪些",
        "请问",
        "一下",
        "吗",
        "呢",
        "啊",
        "呀",
    }
)


def _normalize_query_for_match(query: str) -> str:
    q = (query or "").strip()
    q = re.sub(r"[\s!！?？。.，,~、；;：:""''\"']+", "", q, flags=re.I)
    return q


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        if len(term) < 2 or term in _QUERY_TERM_STOP or term in seen:
            return
        seen.add(term)
        terms.append(term)

    for run in re.findall(r"[\u4e00-\u9fff]+", query or ""):
        if 2 <= len(run) <= 12:
            add(run)
        for i in range(len(run) - 1):
            add(run[i : i + 2])
        for i in range(len(run) - 2):
            add(run[i : i + 3])

    for term in re.findall(r"[a-zA-Z0-9]{3,}", (query or "").lower()):
        add(term)
    return terms


def query_wants_kb_images(query: str | None) -> bool:
    """False for greetings/chitchat that should never trigger doc images."""
    q = (query or "").strip()
    if not q:
        return False
    qn = _normalize_query_for_match(q)
    if not qn:
        return False
    for pat in _CHITCHAT_ONLY_PATTERNS:
        if re.fullmatch(pat, qn, re.I):
            return False
    if len(qn) <= 2 and not _query_terms(q):
        return False
    return True


def _max_rerank_score(retrieved_docs: list[dict[str, Any]] | None) -> float | None:
    scores: list[float] = []
    for doc in retrieved_docs or []:
        raw = doc.get("rerank_score")
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else None


def _term_overlap_ratio(query: str, text: str) -> float:
    terms = _query_terms(query)
    if not terms or not text.strip():
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def _image_min_rerank_score() -> float:
    raw = (
        os.getenv("RAG_IMAGE_MIN_RERANK_SCORE")
        or os.getenv("MIN_RERANK_SCORE")
        or "0.28"
    )
    try:
        return float(raw)
    except ValueError:
        return 0.28


def _image_min_term_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_MIN_TERM_OVERLAP") or "0.34"
    try:
        return float(raw)
    except ValueError:
        return 0.34


def retrieval_supports_images(
    query: str | None,
    *,
    retrieved_docs: list[dict[str, Any]] | None = None,
    context_text: str | None = None,
) -> bool:
    """True only when retrieval looks substantively relevant to the query."""
    if not query_wants_kb_images(query):
        logger.info("Skip related images: non-KB / chitchat query")
        return False

    q = (query or "").strip()
    max_score = _max_rerank_score(retrieved_docs)
    if max_score is not None:
        threshold = _image_min_rerank_score()
        if max_score < threshold:
            logger.info(
                "Skip related images: max rerank_score %.3f < %.3f",
                max_score,
                threshold,
            )
            return False
        return True

    text = (context_text or "").strip()
    if not text:
        logger.info("Skip related images: empty retrieval context")
        return False

    terms = _query_terms(q)
    if not terms:
        logger.info("Skip related images: no substantive query terms")
        return False

    overlap = _term_overlap_ratio(q, text)
    min_overlap = _image_min_term_overlap()
    if overlap < min_overlap:
        logger.info(
            "Skip related images: term overlap %.2f < %.2f",
            overlap,
            min_overlap,
        )
        return False
    return True


def normalize_context_for_image_parse(text: str) -> str:
    """LightRAG context may contain literal ``\\n`` instead of real newlines."""
    if not text:
        return ""
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    if "\\t" in text:
        text = text.replace("\\t", "\t")
    return text


def normalize_image_path(path: str) -> str:
    p = path.strip().strip('"').strip("'")
    while "\\\\" in p:
        p = p.replace("\\\\", "\\")
    return p


def _image_block_for_path(context: str, path_match_start: int) -> str:
    """Isolate the ``[图片]`` block that owns this path marker."""
    block_start = context.rfind("[图片]", 0, path_match_start)
    if block_start < 0:
        block_start = max(0, path_match_start - 200)
    next_block = context.find("\n[图片]", path_match_start)
    if next_block < 0:
        next_block = context.find("[图片]", path_match_start + 1)
    if next_block < 0:
        return context[block_start:]
    return context[block_start:next_block]


def _metadata_from_block(block: str) -> dict[str, Any]:
    page = None
    pm = _PAGE_RE.search(block)
    if pm:
        try:
            page = int(pm.group(1))
        except ValueError:
            page = None

    caption = ""
    cm = _CAPTION_RE.search(block)
    if cm:
        caption = cm.group(1).strip()
    else:
        fm = _FOOTNOTE_RE.search(block)
        if fm:
            caption = fm.group(1).strip()
        else:
            cap_m = re.search(r"标注[：:]\s*(.+?)(?:\n|$)", block)
            if cap_m:
                caption = cap_m.group(1).strip()

    ctx_m = _CONTEXT_RE.search(block)
    context_snippet = ctx_m.group(1).strip()[:300] if ctx_m else ""
    return {"page": page, "caption": caption, "context": context_snippet}


def _query_wants_transmission_screw(query: str | None) -> bool:
    if not query:
        return False
    q = query.strip()
    return "传动丝杆" in q or ("传动" in q and "丝杆" in q)


def _extract_maintenance_anchors(text: str) -> list[str]:
    if not text.strip():
        return []
    anchors: list[str] = []
    for pattern in _MAINTENANCE_ANCHOR_PATTERNS:
        for match in re.finditer(pattern, text):
            phrase = match.group(0).strip()
            if len(phrase) >= 4:
                anchors.append(phrase)
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in anchors:
        if phrase not in seen:
            seen.add(phrase)
            deduped.append(phrase)
    return deduped


def _manual_key_from_path(path_str: str) -> str:
    path = Path(normalize_image_path(path_str))
    for part in path.parts:
        if "手册" in part or "维护保养" in part:
            return part.split("_")[0]
    if len(path.parts) >= 2:
        return path.parts[-3] if path.parts[-2] == "images" else path.parts[-2]
    return path.name


def _manual_hints_from_text(text: str) -> set[str]:
    hints: set[str] = set()
    for part in re.split(r"[\n\\n/\\\\]+", text):
        part = part.strip()
        if ("手册" in part or "维护保养" in part) and len(part) >= 6:
            hints.add(part.split("_")[0])
    return hints


def _score_ref_for_query(
    ref: dict[str, Any],
    query: str | None,
    *,
    anchor_phrases: list[str] | None = None,
    manual_hints: set[str] | None = None,
) -> int:
    if not query:
        return 0
    blob = " ".join(
        str(ref.get(key) or "")
        for key in ("caption", "context", "path")
    )
    q = query.strip()
    score = 0
    wants_ts = _query_wants_transmission_screw(q)

    if wants_ts:
        if "传动丝杆" not in blob:
            if "丝杆" in blob:
                return 0
            return 0
        score += 45
        for neg in _NEG_FOR_TRANSMISSION_SCREW:
            if neg in blob and "传动丝杆" not in blob:
                return 0

    for anchor in anchor_phrases or _extract_maintenance_anchors(q):
        if anchor in blob:
            score += 30

    for neg in _NEG_FOR_TRANSMISSION_SCREW:
        if neg in blob and wants_ts and "传动丝杆" not in blob:
            score -= 40

    if len(q) >= 2 and q in blob:
        score += 20
    for term in re.findall(r"[\u4e00-\u9fff]{2,}", q):
        if term in blob:
            score += 3
    if "保养" in q and "保养" in blob:
        score += 2

    if manual_hints:
        path = str(ref.get("path") or "")
        for hint in manual_hints:
            if hint and hint in path:
                score += 18

    return max(0, score)


def extract_image_refs_from_context(context: str) -> list[dict[str, Any]]:
    """Parse image metadata from retrieved LightRAG context text."""
    context = normalize_context_for_image_parse(context)
    if not context.strip():
        return []

    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in _IMAGE_PATH_RE.finditer(context):
        path = normalize_image_path(m.group(1) or m.group(2) or "")
        if not path:
            continue
        dedupe_key = Path(path).name
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        block = _image_block_for_path(context, m.start())
        meta = _metadata_from_block(block)
        refs.append({"path": path, **meta})

    return refs


def _merge_refs(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for ref in group:
            key = Path(str(ref.get("path") or "")).name
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(ref)
    return merged


def _closest_image_for_text(
    items: list[dict[str, Any]], text_item: dict[str, Any]
) -> dict[str, Any] | None:
    page_idx = text_item.get("page_idx")
    text_center = _bbox_center(text_item.get("bbox"))
    if page_idx is None or text_center is None:
        return None

    best: dict[str, Any] | None = None
    best_dist = float("inf")
    for item in items:
        if item.get("type") != "image" or item.get("page_idx") != page_idx:
            continue
        img_center = _bbox_center(item.get("bbox"))
        if img_center is None:
            continue
        dist = math.hypot(img_center[0] - text_center[0], img_center[1] - text_center[1])
        if dist < best_dist:
            best_dist = dist
            best = item
    if best is None or best_dist > _BBOX_MATCH_MAX_DIST:
        return None
    return best


def supplement_refs_from_content_lists(
    text: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Match maintenance phrases in retrieved text to same-page images via MinerU bbox."""
    anchors = _extract_maintenance_anchors(text)
    if query:
        anchors = _merge_anchor_lists(anchors, _extract_maintenance_anchors(query))
    if not anchors:
        return []

    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in media_roots:
        try:
            content_lists = list(root.rglob("*_content_list.json"))
        except OSError:
            continue
        for cl_path in content_lists:
            try:
                items = json.loads(cl_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(items, list):
                continue
            auto_dir = cl_path.parent

            for anchor in anchors:
                for item in items:
                    if item.get("type") != "text":
                        continue
                    body = item.get("text")
                    if not isinstance(body, str) or anchor not in body:
                        continue
                    image_item = _closest_image_for_text(items, item)
                    if image_item is None:
                        continue
                    rel_path = (image_item.get("img_path") or "").strip()
                    if not rel_path:
                        continue
                    full_path = (auto_dir / rel_path).resolve()
                    key = full_path.name
                    if key in seen:
                        continue
                    seen.add(key)
                    page_idx = image_item.get("page_idx")
                    refs.append(
                        {
                            "path": str(full_path),
                            "page": page_idx if isinstance(page_idx, int) else None,
                            "caption": anchor[:80],
                            "context": body.strip()[:300],
                        }
                    )
    return refs


def _merge_anchor_lists(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for phrase in [*a, *b]:
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out


def text_from_retrieved_docs(docs: list[dict[str, Any]] | None) -> str:
    """Concatenate chunk bodies returned by LightRAG rerank / retrieval."""
    parts: list[str] = []
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        for key in ("content", "text", "chunk_content", "page_content"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return "\n\n".join(parts)


def merge_context_for_images(*sources: str | None) -> str:
    parts = [s.strip() for s in sources if isinstance(s, str) and s.strip()]
    return "\n\n".join(parts)


def resolve_media_path(path_str: str, media_roots: list[Path]) -> Path | None:
    """Resolve image path; fall back to filename search under parser output (re-ingest safe)."""
    path = Path(normalize_image_path(path_str))
    if is_safe_media_path(path, media_roots):
        return path.resolve()
    name = path.name
    if not name:
        return None
    for root in media_roots:
        try:
            for candidate in root.rglob(name):
                if is_safe_media_path(candidate, media_roots):
                    return candidate.resolve()
        except OSError:
            continue
    return None


def is_safe_media_path(path: Path, allowed_roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    if resolved.suffix.lower() not in _IMAGE_EXTS:
        return False
    for root in allowed_roots:
        try:
            root_res = root.resolve()
            if resolved.is_relative_to(root_res):
                return True
        except OSError:
            continue
    return False


def encode_media_token(abs_path: Path, media_root: Path) -> str:
    rel = abs_path.resolve().relative_to(media_root.resolve())
    raw = str(rel).replace("\\", "/").encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_media_token(token: str, media_root: Path) -> Path | None:
    if not token or ".." in token or token.startswith("/"):
        return None
    try:
        pad = "=" * (-len(token) % 4)
        rel = base64.urlsafe_b64decode(token + pad).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if ".." in Path(rel).parts:
        return None
    return (media_root.resolve() / rel).resolve()


def _select_scored_refs(
    scored: list[tuple[int, dict[str, Any]]],
    *,
    limit: int,
    min_relative: float = 0.45,
    min_absolute: int = 20,
) -> list[dict[str, Any]]:
    if not scored:
        return []
    scored = [(score, ref) for score, ref in scored if score > 0]
    if not scored:
        return []

    best_by_manual: dict[str, tuple[int, dict[str, Any]]] = {}
    for score, ref in scored:
        manual = _manual_key_from_path(str(ref.get("path") or ""))
        prev = best_by_manual.get(manual)
        if prev is None or score > prev[0]:
            best_by_manual[manual] = (score, ref)

    candidates = sorted(best_by_manual.values(), key=lambda pair: pair[0], reverse=True)
    max_score = candidates[0][0]
    threshold = max(min_absolute, int(max_score * min_relative))

    selected: list[dict[str, Any]] = []
    for score, ref in candidates:
        if score < min_absolute:
            continue
        if not selected or score >= threshold:
            selected.append(ref)
        if len(selected) >= limit:
            break
    return selected


def images_for_api(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    extra_context: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Turn retrieved context into Web-safe image descriptors."""
    primary_text = context or ""
    merged_for_gate = merge_context_for_images(primary_text, extra_context)
    if not retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=merged_for_gate,
    ):
        return []

    anchor_phrases = _extract_maintenance_anchors(primary_text)
    if query:
        anchor_phrases = _merge_anchor_lists(
            anchor_phrases, _extract_maintenance_anchors(query)
        )
    manual_hints = _manual_hints_from_text(primary_text)

    refs = _merge_refs(
        extract_image_refs_from_context(primary_text),
        extract_image_refs_from_context(extra_context or ""),
        supplement_refs_from_content_lists(
            primary_text, media_roots, query=query
        ),
    )
    if not refs:
        return []

    scored = [
        (
            _score_ref_for_query(
                ref,
                query,
                anchor_phrases=anchor_phrases,
                manual_hints=manual_hints,
            ),
            ref,
        )
        for ref in refs
    ]
    selected = _select_scored_refs(scored, limit=limit)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for ref in selected:
        path_str = ref.get("path") or ""
        path_key = Path(path_str).name
        if not path_str or path_key in seen:
            continue
        resolved = resolve_media_path(path_str, media_roots)
        if resolved is None:
            continue
        root_for_token: Path | None = None
        for root in media_roots:
            try:
                if resolved.is_relative_to(root.resolve()):
                    root_for_token = root.resolve()
                    break
            except OSError:
                continue
        if root_for_token is None:
            continue
        seen.add(path_key)
        token = encode_media_token(resolved, root_for_token)
        caption = ref.get("caption") or ""
        context_snippet = ref.get("context") or ""
        if not caption and context_snippet:
            caption = context_snippet[:60]
        item: dict[str, Any] = {
            "url": f"/api/media/image?token={token}",
            "caption": caption,
        }
        if ref.get("page") is not None:
            item["page"] = ref["page"]
        if context_snippet:
            item["context"] = context_snippet
        out.append(item)

    if out:
        logger.info(
            "Resolved %d related image(s) for query (%d candidate path(s))",
            len(out),
            len(refs),
        )
    return out
