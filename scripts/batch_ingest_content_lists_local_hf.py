#!/usr/bin/env python
"""Embed MinerU *_content_list_v2.json trees using **local Hugging Face** embeddings only.

Sets ``ALLOW_EMBEDDING_ONLY_INGESTION`` and ``EMBEDDING_BACKEND=hf``. No OpenAI embedding
or LLM calls: install ``raganything[local-embed]`` (``sentence-transformers``).

Env: ``EMBEDDING_MODEL`` (default BAAI/bge-m3), ``EMBEDDING_DIM`` (default 1024), optional
``HF_HOME`` / repo ``.hf_cache`` via ``ensure_hf_home_from_repo_fallback``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(dotenv_path=_ROOT / ".env", override=False)


async def _noop_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    """LightRAG requires an llm_model_func; embedding-only ingestion never calls it."""
    return ""


async def _noop_vision(
    prompt,
    system_prompt=None,
    history_messages=None,
    image_data=None,
    messages=None,
    **kwargs,
):
    return ""


async def async_main() -> int:
    from lightrag import LightRAG
    from lightrag.utils import logger
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.local_hf_embedding import (
        ensure_hf_home_from_repo_fallback,
        make_local_hf_embedding_func,
    )

    ensure_hf_home_from_repo_fallback(_ROOT)

    p = argparse.ArgumentParser(
        description="Batch ingest MinerU JSON with local HF embeddings (embedding-only)."
    )
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "rag_storage_wt1536",
        help="RAG persistence directory (default: ./rag_storage_wt1536).",
    )
    p.add_argument(
        "--data-repo-root",
        type=Path,
        default=None,
        help="Root containing output/data_upload_test_v3/ … or set RAG_DATA_REPO.",
    )
    args = p.parse_args()

    repo_data = args.data_repo_root
    if repo_data is None:
        env_dr = os.getenv("RAG_DATA_REPO", "").strip()
        repo_data = Path(env_dr) if env_dr else _ROOT

    repo_root = repo_data.expanduser().resolve()
    glob_root = (repo_root / "output" / "data_upload_test_v3").resolve()

    requested_embedding_backend = os.getenv("EMBEDDING_BACKEND", "hf").strip().lower()
    if requested_embedding_backend != "hf":
        logger.error(
            "batch_ingest_content_lists_local_hf.py only supports local Hugging Face embeddings. "
            "Set EMBEDDING_BACKEND=hf (or unset; default is hf). "
            f"Got EMBEDDING_BACKEND={requested_embedding_backend!r}."
        )
        return 2

    os.environ["ALLOW_EMBEDDING_ONLY_INGESTION"] = "true"
    os.environ["EMBEDDING_BACKEND"] = "hf"

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_func = make_local_hf_embedding_func(
        embedding_dim,
        embedding_model=embedding_model,
    )

    config = RAGAnythingConfig(
        working_dir=str(args.working_dir),
        allow_embedding_only_ingestion=True,
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    llm_model_func = _noop_llm
    vision_model_func = _noop_vision

    lightrag = LightRAG(
        working_dir=str(args.working_dir),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        enable_llm_cache=True,
        embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "1")),
        embedding_batch_num=int(os.getenv("EMBEDDING_BATCH_NUM", "1")),
    )
    await lightrag.initialize_storages()

    rag = RAGAnything(
        config=config,
        lightrag=lightrag,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    paths = sorted(glob_root.rglob("*_content_list_v2.json"))
    if not paths:
        logger.error(f"No *_content_list_v2.json under {glob_root}")
        return 1

    ok = fail = 0
    for path in paths:
        path = path.resolve()
        rel = path.relative_to(repo_root)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                logger.warning(f"Skip (not a list): {rel}")
                fail += 1
                continue
            await rag.insert_content_list(raw, file_path=str(rel))
            ok += 1
        except Exception as e:
            logger.error(f"Ingest failed {rel}: {e}")
            fail += 1

    logger.info(f"INGEST_DONE::ok={ok}::fail={fail}")
    await rag.finalize_storages()
    return 0 if fail == 0 else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
