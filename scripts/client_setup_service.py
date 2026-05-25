"""Setup wizard checks: runtime deps, bundled models, knowledge base."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from client_env_manager import load_env_dict
from client_paths import (
    get_app_root,
    get_env_path,
    get_models_dir,
    get_rag_storage_dir,
    is_client_mode,
    is_setup_complete,
    read_setup_marker,
)

# Expected bundled HF hub folder names (under HF_HOME/hub/).
BUNDLED_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "id": "embedding",
        "label": "向量模型 BAAI/bge-m3",
        "hub_dir": "models--BAAI--bge-m3",
        "weight_files": ("pytorch_model.bin", "model.safetensors"),
    },
    {
        "id": "rerank",
        "label": "Rerank 模型 BAAI/bge-reranker-base",
        "hub_dir": "models--BAAI--bge-reranker-base",
        "weight_files": ("pytorch_model.bin", "model.safetensors"),
    },
    {
        "id": "mineru",
        "label": "PDF 解析 MinerU (PDF-Extract-Kit-1.0)",
        "hub_dir": "models--opendatalab--PDF-Extract-Kit-1.0",
        "weight_files": ("README.md",),
    },
]


def resolve_multimodal_enabled() -> bool:
    """True when ingest should process images/tables/equations via LLM."""
    enable = (os.getenv("RAG_WEB_ENABLE_MULTIMODAL") or "").strip().lower()
    if enable in ("1", "true", "yes", "on"):
        return True
    if enable in ("0", "false", "no", "off"):
        return False
    skip = (os.getenv("RAG_WEB_SKIP_MULTIMODAL", "true") or "true").strip().lower()
    return skip not in ("1", "true", "yes", "on")


def _hub_model_ready(hub_root: Path, hub_dir: str, weight_names: tuple[str, ...]) -> bool:
    base = hub_root / "hub" / hub_dir
    if not base.is_dir():
        return False
    for snap in base.glob("snapshots/*"):
        if not snap.is_dir():
            continue
        if any((snap / name).is_file() for name in weight_names):
            return True
        if (snap / "config.json").is_file() and any(
            (snap / name).is_file() for name in weight_names
        ):
            return True
        if any(snap.rglob("*.pth")) or any(snap.rglob("*.safetensors")):
            return True
    return False


def check_runtime_deps() -> list[dict[str, Any]]:
    specs = [
        ("lightrag", "LightRAG 核心"),
        ("fastapi", "Web 服务"),
        ("uvicorn", "Web 服务"),
        ("sentence_transformers", "本地向量模型"),
        ("mineru", "PDF 解析 (MinerU)"),
    ]
    rows: list[dict[str, Any]] = []
    for module, label in specs:
        ok = importlib.util.find_spec(module) is not None
        rows.append({"id": module, "label": label, "ok": ok})

    cv2_ok = importlib.util.find_spec("cv2") is not None
    rows.append(
        {
            "id": "cv2",
            "label": "OpenCV (MinerU PDF 解析依赖 cv2)",
            "ok": cv2_ok,
        }
    )
    return rows


def check_bundled_models() -> list[dict[str, Any]]:
    candidates = [get_models_dir()]
    dev_cache = get_app_root() / ".hf_cache"
    if dev_cache.is_dir() and dev_cache not in candidates:
        candidates.append(dev_cache)
    rows: list[dict[str, Any]] = []
    for spec in BUNDLED_MODEL_SPECS:
        weight_files = tuple(spec.get("weight_files", ("pytorch_model.bin", "model.safetensors")))
        ok = False
        found_path = ""
        for root in candidates:
            if _hub_model_ready(root, spec["hub_dir"], weight_files):
                ok = True
                found_path = str(root / "hub" / spec["hub_dir"])
                break
        rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "ok": ok,
                "path": found_path or str(get_models_dir() / "hub" / spec["hub_dir"]),
            }
        )
    return rows


def resolve_hf_home_for_runtime() -> Path:
    """Prefer data/models; fall back to .hf_cache if PDF kit only exists there."""
    primary = get_models_dir()
    dev_cache = get_app_root() / ".hf_cache"
    if not dev_cache.is_dir():
        return primary
    kit = "models--opendatalab--PDF-Extract-Kit-1.0"
    primary_has_kit = _hub_model_ready(primary, kit, ("README.md",))
    cache_has_kit = _hub_model_ready(dev_cache, kit, ("README.md",))
    if cache_has_kit and not primary_has_kit:
        return dev_cache
    return primary


def _json_kv_entry_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 2:
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return len(data)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0
    return 0


def _rag_storage_has_data(wd: Path) -> bool:
    """True when KB has indexed chunks usable for retrieval.

    ``kv_store_full_docs.json`` alone may exist after partial / cancelled ingest;
    require non-empty vector chunks as well.
    """
    if not wd.is_dir():
        return False
    doc_count = _json_kv_entry_count(wd / "kv_store_full_docs.json")
    chunk_count = _json_kv_entry_count(wd / "vdb_chunks.json")
    return doc_count > 0 and chunk_count > 0


def _rag_storage_status(wd: Path) -> dict[str, Any]:
    doc_count = _json_kv_entry_count(wd / "kv_store_full_docs.json")
    chunk_count = _json_kv_entry_count(wd / "vdb_chunks.json")
    unique_docs = _unique_doc_paths(wd / "kv_store_full_docs.json")
    ok = doc_count > 0 and chunk_count > 0
    partial = doc_count > 0 and chunk_count == 0
    if partial:
        if unique_docs and unique_docs < doc_count:
            message = (
                f"检测到 {doc_count} 条文档记录（{unique_docs} 个不同 PDF），"
                "但未完成向量索引（请清空后重新灌库）"
            )
        else:
            message = f"检测到 {doc_count} 篇文档残留，但未完成向量索引（请重新灌库）"
    elif ok:
        if unique_docs and unique_docs < doc_count:
            message = f"已灌库 {unique_docs} 个 PDF（索引记录 {doc_count} 条，含重复灌库）"
        else:
            message = f"已灌库 {doc_count} 篇文档"
    else:
        message = "尚未灌库"
    return {
        "ok": ok,
        "partial": partial,
        "doc_count": doc_count,
        "unique_doc_count": unique_docs,
        "chunk_count": chunk_count,
        "message": message,
    }


def _unique_doc_paths(full_docs_path: Path) -> int:
    if not full_docs_path.is_file():
        return 0
    try:
        data = json.loads(full_docs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    names: set[str] = set()
    for item in data.values():
        if not isinstance(item, dict):
            continue
        for key in ("file_path", "filepath", "source"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                names.add(val.replace("\\", "/").rsplit("/", 1)[-1])
                break
    return len(names)


def check_env_config() -> dict[str, Any]:
    if not get_env_path().is_file():
        return {"ok": False, "message": "尚未保存 .env 配置"}
    data = load_env_dict()
    missing: list[str] = []
    if not (data.get("LLM_BINDING_API_KEY") or data.get("OPENAI_API_KEY")):
        missing.append("LLM API Key")
    if not data.get("LLM_BINDING_HOST"):
        missing.append("LLM API 地址")
    if not data.get("LLM_MODEL"):
        missing.append("LLM 模型")
    if missing:
        return {"ok": False, "message": "缺少：" + "、".join(missing)}
    return {"ok": True, "message": "配置已保存"}


def get_setup_status() -> dict[str, Any]:
    deps = check_runtime_deps()
    models = check_bundled_models()
    env_status = check_env_config()
    wd = get_rag_storage_dir()
    kb = _rag_storage_status(wd)
    kb_ok = kb["ok"]
    hf_home = resolve_hf_home_for_runtime()

    deps_ok = all(r["ok"] for r in deps)
    models_ok = all(r["ok"] for r in models)
    env_data = load_env_dict()
    summary_lang = (env_data.get("SUMMARY_LANGUAGE") or "English").strip()
    prompt_lang = (env_data.get("RAG_PROMPT_LANGUAGE") or "").strip()
    chinese_ingest = summary_lang.lower() in ("chinese", "zh", "cn", "中文")
    multimodal_enabled = resolve_multimodal_enabled()
    vision_model = (env_data.get("VISION_MODEL") or "").strip()

    return {
        "client_mode": is_client_mode(),
        "setup_complete": is_setup_complete(),
        "setup_marker": read_setup_marker(),
        "app_root": str(get_app_root()),
        "env_path": str(get_env_path()),
        "working_dir": str(wd),
        "models_dir": str(get_models_dir()),
        "hf_home_effective": str(hf_home),
        "env": env_status,
        "runtime_deps": deps,
        "runtime_deps_ok": deps_ok,
        "bundled_models": models,
        "bundled_models_ok": models_ok,
        "knowledge_base_ok": kb_ok,
        "knowledge_base": kb,
        "kb_doc_count": kb["doc_count"],
        "kb_unique_doc_count": kb.get("unique_doc_count", kb["doc_count"]),
        "kb_chunk_count": kb["chunk_count"],
        "kb_partial": kb["partial"],
        "ingest_language": summary_lang,
        "prompt_language": prompt_lang or ("zh" if chinese_ingest else "en"),
        "chinese_ingest": chinese_ingest,
        "multimodal_enabled": multimodal_enabled,
        "skip_multimodal": not multimodal_enabled,
        "vision_model": vision_model or None,
        "llm_model": (env_data.get("LLM_MODEL") or "").strip() or None,
        "python": sys.version.split()[0],
        "disk_free_gb": _disk_free_gb(get_app_root()),
        "can_enter_chat": env_status["ok"] and models_ok and deps_ok,
        "can_finish_setup": env_status["ok"] and models_ok and deps_ok and kb_ok,
    }


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 1)
    except OSError:
        return None
