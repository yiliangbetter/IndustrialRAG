"""Client install layout: config / data / bundled models (portable green package)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_SETUP_MARKER_VERSION = 1


def is_client_mode() -> bool:
    v = (os.getenv("RAG_CLIENT_MODE") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_app_root() -> Path:
    raw = (os.getenv("RAG_CLIENT_APP_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def get_config_dir() -> Path:
    d = get_app_root() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    d = get_app_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    d = get_app_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_tiktoken_cache_dir() -> Path:
    """Local tiktoken cache for fully offline LightRAG tokenization."""
    d = get_config_dir() / "tiktoken_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir() -> Path:
    """Bundled Hugging Face cache root (HF_HOME)."""
    d = get_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rag_storage_dir() -> Path:
    d = get_data_dir() / "rag_storage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_parser_output_dir() -> Path:
    d = get_data_dir() / "pipeline_parse"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_env_path() -> Path:
    client_env = get_config_dir() / ".env"
    if client_env.is_file():
        return client_env
    root_env = get_app_root() / ".env"
    if root_env.is_file():
        return root_env
    return client_env


def get_env_example_path() -> Path:
    p = get_config_dir() / "env.example"
    if p.is_file():
        return p
    return get_app_root() / "env.example"


def get_setup_marker_path() -> Path:
    return get_config_dir() / "setup_complete.json"


def read_setup_marker() -> dict | None:
    p = get_setup_marker_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_setup_complete() -> bool:
    if not is_client_mode():
        return True
    return read_setup_marker() is not None


def write_setup_complete() -> dict:
    payload = {
        "version": _SETUP_MARKER_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    p = get_setup_marker_path()
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def clear_setup_complete() -> None:
    p = get_setup_marker_path()
    if p.is_file():
        p.unlink()


def apply_client_env_defaults() -> None:
    """Push portable paths into os.environ before loading dotenv overrides."""
    if not is_client_mode():
        return
    os.environ.setdefault("RAG_CLIENT_APP_ROOT", str(get_app_root()))
    try:
        from client_setup_service import resolve_hf_home_for_runtime  # noqa: WPS433

        os.environ["HF_HOME"] = str(resolve_hf_home_for_runtime())
    except Exception:
        os.environ["HF_HOME"] = str(get_models_dir())
    os.environ.setdefault("RAG_WEB_WORKING_DIR", str(get_rag_storage_dir()))
    os.environ.setdefault("RAG_WEB_PARSER_OUTPUT_DIR", str(get_parser_output_dir()))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(get_tiktoken_cache_dir()))
    os.environ.setdefault("EMBEDDING_BACKEND", "hf")
    os.environ.setdefault("HF_EMBED_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RERANK_BINDING", "hf")
