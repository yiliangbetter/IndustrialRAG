#!/usr/bin/env python
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI


def _mask(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


def _print_result(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def _client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _with_retries(fn, attempts: int = 3, delay_s: float = 1.5):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            is_transient = "429" in msg or "ServerOverloaded" in msg
            if not is_transient or i == attempts - 1:
                raise
            time.sleep(delay_s * (i + 1))
    raise last_exc


def main() -> int:
    load_dotenv(".env")

    base_url = os.getenv("LLM_BINDING_HOST", "").strip()
    llm_model = os.getenv("LLM_MODEL", "").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "").strip()
    llm_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv(
        "LLM_BINDING_API_KEY", ""
    ).strip()
    embedding_key = os.getenv("EMBEDDING_API_KEY", "").strip() or llm_key

    embedding_dim_env = os.getenv("EMBEDDING_DIM", "").strip()
    embedding_dim = None
    if embedding_dim_env:
        try:
            embedding_dim = int(embedding_dim_env)
        except ValueError:
            embedding_dim = None

    print("Ark endpoint validation")
    print("=" * 60)
    print(f"base_url         : {base_url or '<missing>'}")
    print(f"llm_model        : {llm_model or '<missing>'}")
    print(f"embedding_model  : {embedding_model or '<missing>'}")
    print(f"llm_key          : {_mask(llm_key) if llm_key else '<missing>'}")
    print(f"embedding_key    : {_mask(embedding_key) if embedding_key else '<missing>'}")
    print(f"EMBEDDING_DIM    : {embedding_dim_env or '<missing>'}")
    print("=" * 60)

    required_ok = True
    for key, value in (
        ("LLM_BINDING_HOST", base_url),
        ("LLM_MODEL", llm_model),
        ("EMBEDDING_MODEL", embedding_model),
        ("OPENAI_API_KEY / LLM_BINDING_API_KEY", llm_key),
    ):
        ok = bool(value)
        _print_result(f"env:{key}", ok)
        required_ok = required_ok and ok

    if not required_ok:
        print("\nMissing required configuration in .env")
        return 2

    chat_ok = False
    embedding_ok = False
    dim_ok = False

    print("\nRunning API sanity checks...")

    try:
        response = _with_retries(
            lambda: _client(llm_key, base_url).chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
            )
        )
        msg = response.choices[0].message.content if response.choices else "empty"
        _print_result("chat:/chat/completions", True, f"reply={json.dumps(msg)[:80]}")
        chat_ok = True
    except Exception as exc:
        _print_result("chat:/chat/completions", False, str(exc)[:180])

    found_dim = None
    try:
        response = _with_retries(
            lambda: _client(embedding_key, base_url).embeddings.create(
                model=embedding_model, input=["ping"]
            )
        )
        embedding = response.data[0].embedding if response.data else []
        found_dim = len(embedding)
        _print_result("embed:/embeddings", True, f"vector_dim={found_dim}")
        embedding_ok = True
    except Exception as exc:
        _print_result("embed:/embeddings", False, str(exc)[:180])

    if embedding_ok:
        if embedding_dim is None:
            _print_result("embedding_dim_match", True, "EMBEDDING_DIM not set, skipped")
            dim_ok = True
        else:
            dim_ok = embedding_dim == found_dim
            _print_result(
                "embedding_dim_match",
                dim_ok,
                f"env={embedding_dim}, actual={found_dim}",
            )

    success = chat_ok and embedding_ok and dim_ok
    print("\nFinal:", "READY for ingestion" if success else "NOT ready for ingestion")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
