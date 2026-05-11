#!/usr/bin/env python3
"""Ingest MinerU *_content_list_v2.json through LightRAG with entity/relation extraction (KG).

Uses LLM from .env (LLM_MODEL, OPENAI_API_KEY / LLM_BINDING_*). Does **not** enable
embedding-only ingestion; multimodal batch is skipped by default after text insert
(--skip-multimodal, default True) to avoid duplicate vision/table costs when building
a text-first graph.

Mirrors embedding env handling in scripts/batch_ingest_content_lists.py.

Run from the repo root so `.env` resolves; use uv so `python-dotenv` (dev dependency) is available:

    uv run python scripts/batch_ingest_content_lists_with_graph.py -w ./rag_storage_with_kg
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(dotenv_path=_ROOT / ".env", override=False)


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc, logger
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.local_hf_embedding import (
        ensure_hf_home_from_repo_fallback,
        make_local_hf_embedding_func,
    )

    ensure_hf_home_from_repo_fallback(_ROOT)

    p = argparse.ArgumentParser()
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "rag_storage_with_kg",
        help="Fresh RAG dir for graph+chunks (default: ./rag_storage_with_kg).",
    )
    p.add_argument(
        "--data-repo-root",
        type=Path,
        default=None,
        help="Checkout containing output/<subdir>/... (or export RAG_DATA_REPO).",
    )
    p.add_argument(
        "--data-upload-subdir",
        type=str,
        default=os.getenv("RAG_DATA_UPLOAD_SUBDIR", "data_upload_test_v3").strip()
        or "data_upload_test_v3",
        help="Under output/ (default: env RAG_DATA_UPLOAD_SUBDIR or data_upload_test_v3). "
        "Use data_upload_test_v4 for OCR re-ingest trees.",
    )
    p.add_argument(
        "--content-list-glob",
        type=str,
        default="*_content_list_v2.json",
        help="Basename glob for MinerU content list JSON (default: *_content_list_v2.json).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ingest at most N JSON files (0 = all).",
    )
    p.add_argument(
        "--skip-multimodal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After text insert via LightRAG, skip multimodal processors (default: True).",
    )
    args = p.parse_args()

    repo_data = args.data_repo_root
    if repo_data is None:
        env_dr = os.getenv("RAG_DATA_REPO", "").strip()
        if env_dr:
            repo_data = Path(env_dr)
        else:
            repo_data = _ROOT

    repo_root = repo_data.expanduser().resolve()
    glob_root = (repo_root / "output" / args.data_upload_subdir).resolve()
    if not glob_root.is_dir():
        raise SystemExit(f"Data root does not exist: {glob_root}")

    llm_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("LLM_BINDING_API_KEY", "").strip()
    )
    if not llm_key:
        raise SystemExit("Set OPENAI_API_KEY or LLM_BINDING_API_KEY for graph extraction.")

    emb_key = os.getenv("EMBEDDING_API_KEY", "").strip() or llm_key
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "openai").strip().lower()

    # OpenAI-compatible clients often set OPENAI_BASE_URL; LightRAG uses LLM_BINDING_HOST.
    base_url = (
        os.getenv("LLM_BINDING_HOST", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or None
    )
    emb_host = os.getenv("EMBEDDING_BINDING_HOST", "").strip()
    embedding_base_url = emb_host if emb_host else base_url
    if embedding_backend != "hf":
        if emb_host and not os.getenv("EMBEDDING_API_KEY", "").strip():
            raise SystemExit(
                "EMBEDDING_BINDING_HOST is set; set EMBEDDING_API_KEY for that host."
            )

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vision_model = os.getenv("VISION_MODEL", llm_model)
    if embedding_backend == "hf":
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    else:
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    config = RAGAnythingConfig(
        working_dir=str(args.working_dir),
        allow_embedding_only_ingestion=False,
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        if history_messages is None:
            history_messages = []
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=llm_key,
            base_url=base_url,
            **kwargs,
        )

    async def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if history_messages is None:
            history_messages = []
        if messages:
            return await openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=llm_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            return await openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                },
                            },
                        ],
                    },
                ],
                api_key=llm_key,
                base_url=base_url,
                **kwargs,
            )
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    if embedding_backend == "hf":
        embedding_func = make_local_hf_embedding_func(
            embedding_dim,
            embedding_model=embedding_model,
        )
    else:
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=embedding_model,
                api_key=emb_key,
                base_url=embedding_base_url,
            ),
        )

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

    pattern = args.content_list_glob.lstrip("/")
    paths = sorted(
        p for p in glob_root.rglob(pattern) if p.is_file()
    )
    if not paths:
        raise SystemExit(f"No files matching {pattern!r} under {glob_root}")

    if args.limit > 0:
        paths = paths[: args.limit]

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
            await rag.insert_content_list(
                raw,
                file_path=str(rel),
                skip_multimodal_processing=args.skip_multimodal,
            )
            ok += 1
        except Exception as e:
            logger.error(f"Ingest failed {rel}: {e}")
            fail += 1

    logger.info(f"INGEST_DONE::ok={ok}::fail={fail}")
    await rag.finalize_storages()


if __name__ == "__main__":
    main()
