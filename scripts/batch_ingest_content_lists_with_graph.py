#!/usr/bin/env python3
"""Ingest MinerU *_content_list_v2.json through LightRAG with entity/relation extraction (KG).

Uses LLM from .env (LLM_MODEL, OPENAI_API_KEY / LLM_BINDING_*). Does **not** enable
embedding-only ingestion; multimodal batch is skipped by default after text insert
(--skip-multimodal, default True) to avoid duplicate vision/table costs when building
a text-first graph.

Mirrors embedding-related env vars where applicable; see ``scripts/batch_ingest_content_lists_local_hf.py`` for local-only batch ingest.

Run from the repo root so `.env` resolves; use uv so `python-dotenv` (dev dependency) is available:

    uv run python scripts/batch_ingest_content_lists_with_graph.py -w ./rag_storage_with_kg
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


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    from raganything import RAGAnythingConfig
    from raganything.runtime_factory import RuntimeOptions, create_rag_runtime

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

    config = RAGAnythingConfig(
        working_dir=str(args.working_dir),
        allow_embedding_only_ingestion=False,
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    runtime = await create_rag_runtime(
        config,
        RuntimeOptions(
            project_root=_ROOT,
            embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "1")),
            embedding_batch_num=int(os.getenv("EMBEDDING_BATCH_NUM", "1")),
        ),
    )
    rag, logger = runtime.rag, runtime.logger

    pattern = args.content_list_glob.lstrip("/")
    paths = sorted(p for p in glob_root.rglob(pattern) if p.is_file())
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
