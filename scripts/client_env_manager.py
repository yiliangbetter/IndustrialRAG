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

VISION_MODEL_CUSTOM = "__custom__"

VISION_MODEL_OPTIONS: list[dict[str, str]] = [
    {"value": "qwen-vl-max", "label": "qwen-vl-max（DashScope 推荐）"},
    {"value": "qwen-vl-plus", "label": "qwen-vl-plus（DashScope，更省额度）"},
    {"value": "qwen2.5-vl-72b-instruct", "label": "qwen2.5-vl-72b-instruct（DashScope）"},
    {"value": "gpt-4o", "label": "gpt-4o（OpenAI 视觉模型）"},
    {"value": VISION_MODEL_CUSTOM, "label": "自定义模型名称…"},
]

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
        "hint": "与网关支持的模型名一致（文本灌库、图谱抽取、问答）",
        "type": "text",
        "required": True,
        "placeholder": "qwen-plus",
    },
    {
        "key": "VISION_MODEL",
        "label": "视觉模型（多模态灌库）",
        "hint": "步骤 3 启用多模态时，用于解析 PDF 内图片、表格、公式；须与 API 网关支持的模型一致",
        "type": "select",
        "required": True,
        "default": "qwen-vl-max",
        "options": VISION_MODEL_OPTIONS,
        "allow_custom": True,
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
    "RAG_PROMPT_LANGUAGE": "zh",
}

# Keys persisted when the setup wizard saves (client never needs to edit .env by hand).
CLIENT_WRITE_KEYS: frozenset[str] = frozenset(
    [
        *[f["key"] for f in SETUP_FIELDS],
        *CLIENT_AUTO_KEYS.keys(),
        "LLM_BINDING",
        "RAG_WEB_HOST",
        "RAG_WEB_PORT",
        "RAG_WEB_ENABLE_MULTIMODAL",
        "RAG_WEB_SKIP_MULTIMODAL",
        "HF_HOME",
        "RAG_WEB_WORKING_DIR",
        "RAG_WEB_PARSER_OUTPUT_DIR",
        "TIKTOKEN_CACHE_DIR",
        "OPENAI_API_KEY",
    ]
)


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


def _option_value(opt: Any) -> str:
    if isinstance(opt, dict):
        return str(opt.get("value", ""))
    return str(opt)


def _vision_preset_values() -> set[str]:
    return {
        _option_value(o)
        for o in VISION_MODEL_OPTIONS
        if _option_value(o) != VISION_MODEL_CUSTOM
    }


def get_form_values() -> list[dict[str, Any]]:
    data = load_env_dict()
    rows: list[dict[str, Any]] = []
    presets = _vision_preset_values()
    for item in SETUP_FIELDS:
        key = item["key"]
        raw = (data.get(key) or item.get("default", "")).strip()
        row = {**item, "value": raw, "custom_value": ""}
        if key == "VISION_MODEL":
            if raw and raw not in presets:
                row["value"] = VISION_MODEL_CUSTOM
                row["custom_value"] = raw
            elif not raw:
                row["value"] = str(item.get("default", "qwen-vl-max"))
        rows.append(row)
    return rows


_OPENAI_VISION_DEFAULTS = frozenset(
    {"", "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview", "gpt-4-turbo"}
)


def _sync_vision_model(merged: dict[str, str]) -> None:
    """Avoid stale OpenAI vision defaults when using DashScope or other gateways."""
    host = (merged.get("LLM_BINDING_HOST") or "").lower()
    llm = (merged.get("LLM_MODEL") or "").strip()
    vision = (merged.get("VISION_MODEL") or "").strip()
    if vision and vision not in _OPENAI_VISION_DEFAULTS:
        return
    if "dashscope.aliyuncs.com" in host:
        merged["VISION_MODEL"] = "qwen-vl-max"
    elif llm:
        merged["VISION_MODEL"] = llm


def _resolve_vision_from_payload(payload: dict[str, str]) -> str:
    vision = (payload.get("VISION_MODEL") or "").strip()
    if vision == VISION_MODEL_CUSTOM:
        vision = (payload.get("VISION_MODEL__custom") or "").strip()
    return vision


def validate_form(payload: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for item in SETUP_FIELDS:
        key = item["key"]
        val = (payload.get(key) or "").strip()
        if item.get("required") and not val:
            errors.append(f"「{item['label']}」不能为空")
        if key == "VISION_MODEL" and val == VISION_MODEL_CUSTOM:
            custom = (payload.get("VISION_MODEL__custom") or "").strip()
            if not custom:
                errors.append("「视觉模型」选择「自定义」时，请填写模型名称")
    return errors


def _write_env_dict(merged: dict[str, str]) -> Path:
    if is_client_mode():
        merged = {k: v for k, v in merged.items() if k in CLIENT_WRITE_KEYS}
    lines: list[str] = [
        "### Generated / updated by Nanxing RAG client setup wizard",
        "",
    ]
    priority = [
        *[f["key"] for f in SETUP_FIELDS],
        "VISION_MODEL",
        *CLIENT_AUTO_KEYS.keys(),
        "RAG_WEB_ENABLE_MULTIMODAL",
        "RAG_WEB_SKIP_MULTIMODAL",
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


def patch_env_keys(updates: dict[str, str]) -> Path:
    """Merge keys into saved .env without running the setup form."""
    merged = load_env_dict()
    merged.update({k: str(v) for k, v in updates.items()})
    _sync_vision_model(merged)
    if merged.get("LLM_BINDING_API_KEY") and not merged.get("OPENAI_API_KEY"):
        merged["OPENAI_API_KEY"] = merged["LLM_BINDING_API_KEY"]
    return _write_env_dict(merged)


def save_env(payload: dict[str, str]) -> Path:
    errors = validate_form(payload)
    if errors:
        raise ValueError("; ".join(errors))

    base = _parse_env_file(get_env_example_path())
    current = _parse_env_file(get_env_path())
    merged = {**base, **current}

    for item in SETUP_FIELDS:
        key = item["key"]
        if key == "VISION_MODEL":
            continue
        val = (payload.get(key) or "").strip()
        if val:
            merged[key] = val
        elif item.get("default") is not None:
            merged[key] = str(item["default"])

    vision = _resolve_vision_from_payload(payload)
    if vision:
        merged["VISION_MODEL"] = vision
    else:
        _sync_vision_model(merged)

    if is_client_mode():
        merged.setdefault("LLM_BINDING", "openai")
        merged.setdefault("RAG_WEB_HOST", "127.0.0.1")
        merged.setdefault("RAG_WEB_PORT", "8765")
        merged["HF_HOME"] = _client_hf_home()
        merged["RAG_WEB_WORKING_DIR"] = str(get_rag_storage_dir())
        merged["RAG_WEB_PARSER_OUTPUT_DIR"] = str(get_parser_output_dir())
        merged["TIKTOKEN_CACHE_DIR"] = _client_tiktoken_cache_dir()
        merged.update(CLIENT_AUTO_KEYS)

    # Preserve OPENAI_API_KEY alias if user only set LLM key
    if merged.get("LLM_BINDING_API_KEY") and not merged.get("OPENAI_API_KEY"):
        merged["OPENAI_API_KEY"] = merged["LLM_BINDING_API_KEY"]

    return _write_env_dict(merged)


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
