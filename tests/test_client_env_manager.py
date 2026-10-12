"""Setup configuration round-trips without leaking or dropping secrets/settings."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys

from dotenv import dotenv_values
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import client_env_manager as manager  # noqa: E402


def _payload(api_key: str = "new-secret") -> dict[str, str]:
    return {
        "LLM_BINDING_HOST": "https://example.test/v1",
        "LLM_BINDING_API_KEY": api_key,
        "LLM_MODEL": "model",
        "VISION_MODEL": "vision",
        "EMBEDDING_MODEL": "embedding",
        "EMBEDDING_DIM": "1024",
        "RERANK_MODEL": "reranker",
        "RAG_QUERY_MODE": "mix",
    }


@pytest.fixture()
def env_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    example = tmp_path / "env.example"
    saved = tmp_path / ".env"
    example.write_text(
        "\n".join(
            [
                "LLM_BINDING_HOST=https://example.test/v1",
                "LLM_BINDING_API_KEY=",
                "LLM_MODEL=model",
                "VISION_MODEL=vision",
                "EMBEDDING_MODEL=embedding",
                "EMBEDDING_DIM=1024",
                "RERANK_MODEL=reranker",
                "RAG_QUERY_MODE=mix",
                "RAG_WEB_ACCESS_KEY=access-secret",
                "MAX_CONCURRENT_QUERY_CONTENT=9",
                "UNKNOWN_FUTURE_FLAG=keep-me",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manager, "get_env_example_path", lambda: example)
    monkeypatch.setattr(manager, "get_env_path", lambda: saved)
    monkeypatch.setattr(manager, "is_client_mode", lambda: True)
    monkeypatch.setattr(manager, "_client_hf_home", lambda: str(tmp_path / "models"))
    monkeypatch.setattr(
        manager, "get_rag_storage_dir", lambda: tmp_path / "rag_storage"
    )
    monkeypatch.setattr(
        manager, "get_parser_output_dir", lambda: tmp_path / "pipeline_parse"
    )
    monkeypatch.setattr(
        manager, "_client_tiktoken_cache_dir", lambda: str(tmp_path / "tokens")
    )
    return example, saved


def test_client_save_preserves_all_template_and_future_keys(env_files) -> None:
    _, saved = env_files

    manager.save_env(_payload())

    values = dotenv_values(saved)
    assert values["RAG_WEB_ACCESS_KEY"] == "access-secret"
    assert values["MAX_CONCURRENT_QUERY_CONTENT"] == "9"
    assert values["UNKNOWN_FUTURE_FLAG"] == "keep-me"
    if os.name != "nt":
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600


def test_password_is_masked_and_blank_submission_preserves_it(env_files) -> None:
    _, saved = env_files
    saved.write_text("LLM_BINDING_API_KEY=stored-secret\n", encoding="utf-8")

    password = next(
        field
        for field in manager.get_form_values()
        if field["key"] == "LLM_BINDING_API_KEY"
    )
    assert password["value"] == ""
    assert password["configured"] is True
    assert password["required"] is False

    manager.save_env(_payload(api_key=""))
    assert dotenv_values(saved)["LLM_BINDING_API_KEY"] == "stored-secret"


def test_setup_rejects_env_line_injection(env_files) -> None:
    payload = _payload()
    payload["LLM_BINDING_HOST"] = "https://example.test\nRAG_WEB_WORKING_DIR=/"

    with pytest.raises(ValueError, match="控制字符"):
        manager.save_env(payload)
