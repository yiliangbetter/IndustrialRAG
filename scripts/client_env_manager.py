"""Read / write client .env for the setup wizard."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from client_paths import get_env_example_path, get_env_path, is_client_mode
from client_paths import (
    get_models_dir,
    get_parser_output_dir,
    get_rag_storage_dir,
    get_tiktoken_cache_dir,
)

# Keys shown in setup UI (order matters).
SETUP_FIELDS: list[dict[str, Any]] = [
    {
        "key": "LLM_BINDING_HOST",
        "label": "LLM API 地址",
        "hint": "OpenAI 兼容网关，例如 DashScope 兼容地址",
        "type": "text",
        "required": True,
        "placeholder": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    {
        "key": "LLM_BINDING_API_KEY",
        "label": "LLM API Key",
        "hint": "必填，用于灌库实体抽取与问答",
        "type": "password",
        "required": True,
        "placeholder": "",
    },
    {
        "key": "LLM_MODEL",
        "label": "LLM 模型名称",
        "hint": "与网关支持的模型名一致",
        "type": "text",
        "required": True,
        "placeholder": "qwen-plus",
    },
    {
        "key": "EMBEDDING_MODEL",
        "label": "向量模型",
        "hint": "安装包内置 BAAI/bge-m3，一般无需修改",
        "type": "text",
        "required": True,
        "default": "BAAI/bge-m3",
    },
    {
        "key": "EMBEDDING_DIM",
        "label": "向量维度",
        "hint": "bge-m3 为 1024",
        "type": "text",
        "required": True,
        "default": "1024",
    },
    {
        "key": "RERANK_MODEL",
        "label": "Rerank 模型",
        "hint": "安装包内置 BAAI/bge-reranker-base",
        "type": "text",
        "required": True,
        "default": "BAAI/bge-reranker-base",
    },
    {
        "key": "RAG_QUERY_MODE",
        "label": "默认 RAG 模式",
        "hint": "推荐 mix",
        "type": "select",
        "required": True,
        "default": "mix",
        "options": ["mix", "hybrid", "local", "global", "naive"],
    },
]

# Written on every save in client mode (not shown as separate form rows).
CLIENT_AUTO_KEYS: dict[str, str] = {
    "EMBEDDING_BACKEND": "hf",
    "HF_EMBED_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "RERANK_BY_DEFAULT": "true",
    "RERANK_BINDING": "hf",
    "RAG_QUERY_DOC_FILTER": "true",
    "ENABLE_LLM_CACHE": "false",
    "PARSER": "mineru",
    "SUMMARY_LANGUAGE": "Chinese",
}


def _client_tiktoken_cache_dir() -> str:
    return str(get_tiktoken_cache_dir())


def _client_hf_home() -> str:
    try:
        from client_setup_service import resolve_hf_home_for_runtime  # noqa: WPS433

        return str(resolve_hf_home_for_runtime())
    except Exception:
        return str(get_models_dir())

_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _ENV_LINE.match(s)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _quote_env_value(value: str) -> str:
    if not value:
        return ""
    if any(c in value for c in " #\"'\\"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def load_env_dict() -> dict[str, str]:
    merged: dict[str, str] = {}
    example = get_env_example_path()
    if example.is_file():
        merged.update(_parse_env_file(example))
    env_path = get_env_path()
    if env_path.is_file():
        merged.update(_parse_env_file(env_path))
    for item in SETUP_FIELDS:
        key = item["key"]
        if key not in merged and item.get("default") is not None:
            merged[key] = str(item["default"])
    if is_client_mode():
        merged["HF_HOME"] = _client_hf_home()
        merged["RAG_WEB_WORKING_DIR"] = str(get_rag_storage_dir())
        merged["RAG_WEB_PARSER_OUTPUT_DIR"] = str(get_parser_output_dir())
        merged["TIKTOKEN_CACHE_DIR"] = _client_tiktoken_cache_dir()
        merged.update(CLIENT_AUTO_KEYS)
    return merged


def get_form_values() -> list[dict[str, Any]]:
    data = load_env_dict()
    rows: list[dict[str, Any]] = []
    for item in SETUP_FIELDS:
        key = item["key"]
        row = {**item, "value": data.get(key, item.get("default", ""))}
        rows.append(row)
    return rows


def validate_form(payload: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for item in SETUP_FIELDS:
        key = item["key"]
        val = (payload.get(key) or "").strip()
        if item.get("required") and not val:
            errors.append(f"「{item['label']}」不能为空")
    return errors


def save_env(payload: dict[str, str]) -> Path:
    errors = validate_form(payload)
    if errors:
        raise ValueError("; ".join(errors))

    base = _parse_env_file(get_env_example_path())
    current = _parse_env_file(get_env_path())
    merged = {**base, **current}

    for item in SETUP_FIELDS:
        key = item["key"]
        val = (payload.get(key) or "").strip()
        if val:
            merged[key] = val
        elif item.get("default") is not None:
            merged[key] = str(item["default"])

    if is_client_mode():
        merged["HF_HOME"] = _client_hf_home()
        merged["RAG_WEB_WORKING_DIR"] = str(get_rag_storage_dir())
        merged["RAG_WEB_PARSER_OUTPUT_DIR"] = str(get_parser_output_dir())
        merged["TIKTOKEN_CACHE_DIR"] = _client_tiktoken_cache_dir()
        merged.update(CLIENT_AUTO_KEYS)

    # Preserve OPENAI_API_KEY alias if user only set LLM key
    if merged.get("LLM_BINDING_API_KEY") and not merged.get("OPENAI_API_KEY"):
        merged["OPENAI_API_KEY"] = merged["LLM_BINDING_API_KEY"]

    lines: list[str] = [
        "### Generated / updated by Nanxing RAG client setup wizard",
        "",
    ]
    priority = [
        *[f["key"] for f in SETUP_FIELDS],
        *CLIENT_AUTO_KEYS.keys(),
        "HF_HOME",
        "RAG_WEB_WORKING_DIR",
        "RAG_WEB_PARSER_OUTPUT_DIR",
        "TIKTOKEN_CACHE_DIR",
        "OPENAI_API_KEY",
    ]
    written: set[str] = set()
    for key in priority:
        if key in merged and key not in written:
            lines.append(f"{key}={_quote_env_value(merged[key])}")
            written.add(key)
    for key in sorted(merged.keys()):
        if key not in written:
            lines.append(f"{key}={_quote_env_value(merged[key])}")
            written.add(key)

    env_path = get_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def apply_env_to_process() -> None:
    """Reload saved .env into the current process."""
    from dotenv import load_dotenv

    load_dotenv(get_env_path(), override=True)
    if is_client_mode():
        try:
            from client_setup_service import resolve_hf_home_for_runtime  # noqa: WPS433

            os.environ["HF_HOME"] = str(resolve_hf_home_for_runtime())
        except Exception:
            os.environ["HF_HOME"] = str(get_models_dir())
        os.environ.setdefault("RAG_WEB_WORKING_DIR", str(get_rag_storage_dir()))
        os.environ.setdefault("RAG_WEB_PARSER_OUTPUT_DIR", str(get_parser_output_dir()))
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", _client_tiktoken_cache_dir())
        if (os.getenv("HF_EMBED_OFFLINE") or "").strip().lower() in ("1", "true", "yes"):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
