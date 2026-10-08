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

# Static bundled specs (MinerU). Embedding/rerank paths come from saved .env — see _bundled_model_specs().
_STATIC_BUNDLED_SPECS: list[dict[str, Any]] = [
    {
        "id": "mineru",
        "label": "PDF 解析 MinerU (PDF-Extract-Kit-1.0)",
        "hub_dir": "models--opendatalab--PDF-Extract-Kit-1.0",
        "weight_files": ("README.md",),
    },
]


def _hf_hub_dir(repo_id: str) -> str:
    """``BAAI/bge-m3`` → ``models--BAAI--bge-m3`` (HF hub cache folder name)."""
    return "models--" + repo_id.strip().replace("/", "--")


def _hub_cache_roots() -> list[Path]:
    """HF hub cache directories to search (aligned with runtime HF_HOME resolution)."""
    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str | None) -> None:
        if not path:
            return
        p = Path(path).expanduser()
        try:
            p = p.resolve()
        except OSError:
            p = p.absolute()
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        if p.is_dir():
            roots.append(p)

    data = load_env_dict()
    hf_raw = (data.get("HF_HOME") or "").strip().strip('"').strip("'")
    if hf_raw:
        add(hf_raw)
    add(resolve_hf_home_for_runtime())
    add(get_models_dir())
    dev_cache = get_app_root() / ".hf_cache"
    if dev_cache.is_dir():
        add(dev_cache)
    return roots or [get_models_dir()]


def _requires_local_hf_embedding() -> bool:
    backend = (load_env_dict().get("EMBEDDING_BACKEND") or "hf").strip().lower()
    return backend in ("hf", "local", "sentence_transformers")


def _requires_local_hf_rerank() -> bool:
    data = load_env_dict()
    binding = (data.get("RERANK_BINDING") or "hf").strip().lower()
    if binding in ("", "none", "off", "false", "0", "disabled"):
        return False
    return binding in ("hf", "local", "cross_encoder", "sentence_transformers")


def _bundled_model_specs() -> list[dict[str, Any]]:
    """Resolve embedding/rerank check targets from saved env (setup form values)."""
    data = load_env_dict()
    embedding = (data.get("EMBEDDING_MODEL") or "BAAI/bge-m3").strip()
    rerank = (data.get("RERANK_MODEL") or "BAAI/bge-reranker-base").strip()
    rerank_binding = (data.get("RERANK_BINDING") or "hf").strip().lower()
    specs: list[dict[str, Any]] = [
        {
            "id": "embedding",
            "label": f"向量模型 {embedding}",
            "hub_dir": _hf_hub_dir(embedding),
            "weight_files": ("pytorch_model.bin", "model.safetensors"),
            "requires_local": _requires_local_hf_embedding(),
            "remote_binding": (data.get("EMBEDDING_BACKEND") or "hf").strip().lower(),
        },
        {
            "id": "rerank",
            "label": f"Rerank 模型 {rerank}",
            "hub_dir": _hf_hub_dir(rerank),
            "weight_files": ("pytorch_model.bin", "model.safetensors"),
            "requires_local": _requires_local_hf_rerank(),
            "remote_binding": rerank_binding,
        },
        *_STATIC_BUNDLED_SPECS,
    ]
    for spec in specs:
        if spec.get("requires_local") is not False:
            continue
        binding = spec.get("remote_binding") or "api"
        if spec["id"] == "embedding":
            spec["label"] = f"向量模型 {embedding}（{binding} API，无需本地缓存）"
        elif spec["id"] == "rerank":
            spec["label"] = f"Rerank 模型 {rerank}（{binding} API，无需本地缓存）"
    return specs


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
    candidates = _hub_cache_roots()
    rows: list[dict[str, Any]] = []
    for spec in _bundled_model_specs():
        if spec.get("requires_local") is False:
            rows.append(
                {
                    "id": spec["id"],
                    "label": spec["label"],
                    "ok": True,
                    "path": "",
                    "remote": True,
                }
            )
            continue
        weight_files = tuple(spec.get("weight_files", ("pytorch_model.bin", "model.safetensors")))
        ok = False
        found_path = ""
        for root in candidates:
            if _hub_model_ready(root, spec["hub_dir"], weight_files):
                ok = True
                found_path = str(root / "hub" / spec["hub_dir"])
                break
        default_path = str(candidates[0] / "hub" / spec["hub_dir"])
        rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "ok": ok,
                "path": found_path or default_path,
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


def _load_doc_status_map(wd: Path) -> dict[str, Any]:
    path = wd / "kv_store_doc_status.json"
    if not path.is_file() or path.stat().st_size <= 2:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _vector_chunk_count(wd: Path) -> int:
    path = wd / "vdb_chunks.json"
    if not path.is_file() or path.stat().st_size <= 2:
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return len(data["data"])
    return _json_kv_entry_count(path)


def _doc_status_display_name(meta: dict[str, Any], doc_id: str) -> str:
    for key in ("file_path", "filepath", "source"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.replace("\\", "/").rsplit("/", 1)[-1]
    return doc_id


def _normalize_doc_status(raw: Any) -> str:
    if raw is None:
        return ""
    val = getattr(raw, "value", None)
    if isinstance(val, str) and val.strip():
        return val.strip().lower()
    text = str(raw).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _rag_storage_has_data(wd: Path) -> bool:
    """True when KB has successfully indexed chunks usable for retrieval."""
    kb = _rag_storage_status(wd)
    return kb["success_count"] > 0 and kb["chunk_count"] > 0


def _rag_storage_status(wd: Path) -> dict[str, Any]:
    doc_status = _load_doc_status_map(wd)
    chunk_count = _vector_chunk_count(wd)
    full_doc_count = _json_kv_entry_count(wd / "kv_store_full_docs.json")

    success_docs: list[dict[str, str]] = []
    failed_docs: list[dict[str, str]] = []
    in_progress = 0

    for doc_id, meta in doc_status.items():
        if not isinstance(meta, dict):
            continue
        status = _normalize_doc_status(meta.get("status"))
        name = _doc_status_display_name(meta, doc_id)
        fp = meta.get("file_path") if isinstance(meta.get("file_path"), str) else ""
        err = str(meta.get("error_msg") or meta.get("error") or "").strip()
        entry = {"doc_id": doc_id, "name": name, "file_path": fp, "error": err}
        if status in ("processed", "completed", "done"):
            success_docs.append(entry)
        elif status in ("failed", "error"):
            failed_docs.append(entry)
        elif status in ("processing", "pending", "handling"):
            in_progress += 1

    success_count = len(success_docs)
    failed_count = len(failed_docs)
    record_count = len(doc_status)

    partial_index = full_doc_count > 0 and chunk_count == 0
    has_failures = failed_count > 0 or in_progress > 0
    ok = success_count > 0 and chunk_count > 0 and not has_failures
    partial = partial_index or (success_count > 0 and has_failures) or (
        success_count == 0 and (failed_count > 0 or in_progress > 0)
    )

    if success_count == 0 and failed_count == 0 and record_count == 0:
        message = "尚未灌库"
    elif partial_index and success_count == 0:
        message = (
            f"检测到 {full_doc_count} 条文档残留，但未完成向量索引（请清空后重新灌库）"
        )
    elif success_count > 0 and failed_count > 0:
        message = (
            f"已成功灌库 {success_count} 篇，{failed_count} 篇失败"
            "（请查看灌库日志后追加灌库重试失败文件）"
        )
    elif failed_count > 0 and success_count == 0:
        message = f"灌库失败 {failed_count} 篇（请检查 LLM 配额或配置后重试）"
    elif in_progress > 0:
        message = f"有 {in_progress} 篇文档仍在处理中，请稍候或查看日志"
    elif ok:
        message = f"已成功灌库 {success_count} 篇文档"
    elif success_count > 0 and chunk_count == 0:
        message = f"已写入 {success_count} 篇文档记录，但向量索引未完成"
    else:
        message = "尚未灌库"

    unique_success = len({d["name"] for d in success_docs if d.get("name")})

    return {
        "ok": ok,
        "partial": partial,
        "success_count": success_count,
        "failed_count": failed_count,
        "in_progress_count": in_progress,
        "doc_count": record_count or full_doc_count,
        "unique_doc_count": unique_success or success_count,
        "chunk_count": chunk_count,
        "success_docs": success_docs[:30],
        "failed_docs": failed_docs[:30],
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
        "kb_doc_count": kb["success_count"],
        "kb_success_count": kb["success_count"],
        "kb_failed_count": kb["failed_count"],
        "kb_unique_doc_count": kb.get("unique_doc_count", kb["success_count"]),
        "kb_chunk_count": kb["chunk_count"],
        "kb_partial": kb["partial"],
        "ingest_language": summary_lang,
        "prompt_language": prompt_lang or ("zh" if chinese_ingest else "en"),
        "chinese_ingest": chinese_ingest,
        "multimodal_enabled": multimodal_enabled,
        "skip_multimodal": not multimodal_enabled,
        "vision_model": vision_model or None,
        "llm_model": (env_data.get("LLM_MODEL") or "").strip() or None,
        "embedding_model": (env_data.get("EMBEDDING_MODEL") or "").strip() or None,
        "embedding_backend": (env_data.get("EMBEDDING_BACKEND") or "").strip() or None,
        "rerank_model": (env_data.get("RERANK_MODEL") or "").strip() or None,
        "rerank_binding": (env_data.get("RERANK_BINDING") or "").strip() or None,
        "rag_query_mode": (env_data.get("RAG_QUERY_MODE") or "").strip() or None,
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
