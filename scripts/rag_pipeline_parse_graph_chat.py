#!/usr/bin/env python3
"""End-to-end pipeline: MinerU parse → LightRAG graph ingest → interactive Q&A.

Combines behaviour from:
  - ``reingest_uploaded_documents_ocr.py`` (MinerU env kwargs, optional model download),
  - ``batch_ingest_content_lists_with_graph.py`` (async LLM, ``insert_content_list`` +
    ``skip_multimodal_processing`` for text-first KG, same embedding defaults),
  - ``run_demo_question_bank.py``-style querying (``mix`` mode, ``vlm_enhanced=False``).

Parse results are inserted via ``insert_content_list`` so MinerU v2 paragraph/title/list
recovery matches the v4 graph batch path.

Examples (run from repo root with ``uv run python``):

- Parse + ingest + interactive chat:
  ``scripts/rag_pipeline_parse_graph_chat.py --input-folder ./my_docs -w ./rag_storage_my_run``
- Ingest only:
  ``scripts/rag_pipeline_parse_graph_chat.py --input-folder ./pdfs -w ./rag_storage_run --ingest-only``
- One-shot question after ingest:
  ``scripts/rag_pipeline_parse_graph_chat.py --input-folder ./pdfs -w ./rag_storage_run --query '...'``
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(dotenv_path=_ROOT / ".env", override=False)


def _ensure_venv_bin_on_path() -> None:
    venv_bin = _ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")


def _mineru_parse_kwargs(parser_name: str) -> dict:
    out: dict = {}
    lang = os.getenv("MINERU_LANG", "").strip() or os.getenv("OCR_LANG", "").strip()
    if lang:
        out["lang"] = lang
    for key, env in (
        ("backend", "MINERU_BACKEND"),
        ("source", "MINERU_SOURCE"),
        ("device", "MINERU_DEVICE"),
    ):
        v = os.getenv(env, "").strip()
        if v:
            out[key] = v
    if parser_name.lower() == "mineru" and "backend" not in out:
        out["backend"] = "pipeline"
    if parser_name.lower() == "mineru" and "device" not in out and sys.platform == "darwin":
        out["device"] = "cpu"
    return out


def _normalize_ext(e: str) -> str:
    e = e.strip().lower()
    return e if e.startswith(".") else f".{e}"


def _collect_files(folder: Path, extensions: list[str], recursive: bool) -> list[Path]:
    exts = {_normalize_ext(e) for e in extensions}
    files: list[Path] = []
    for ext in exts:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        files.extend(folder.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def _download_mineru_pipeline_models() -> None:
    src = os.getenv("MINERU_MODEL_SOURCE", "huggingface").strip().lower()
    if src not in ("huggingface", "modelscope"):
        src = "huggingface"
    cmd = ["mineru-models-download", "-s", src, "-m", "pipeline"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False, cwd=str(_ROOT))


async def _build_rag(
    working_dir: Path,
    parser_output_dir: Path,
):
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc, logger
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.local_hf_embedding import (
        ensure_hf_home_from_repo_fallback,
        make_local_hf_embedding_func,
    )

    ensure_hf_home_from_repo_fallback(_ROOT)

    llm_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("LLM_BINDING_API_KEY", "").strip()
    )
    if not llm_key:
        raise SystemExit("Set OPENAI_API_KEY or LLM_BINDING_API_KEY.")

    emb_key = os.getenv("EMBEDDING_API_KEY", "").strip() or llm_key
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "openai").strip().lower()

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
        working_dir=str(working_dir),
        allow_embedding_only_ingestion=False,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        parser_output_dir=str(parser_output_dir),
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
        max_concurrent_files=int(os.getenv("MAX_CONCURRENT_FILES", "1")),
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
        working_dir=str(working_dir),
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

    return rag, config, logger


async def _ingest_folder(
    rag,
    config,
    logger,
    *,
    input_folder: Path,
    parser_output_dir: Path,
    parse_method: str,
    parse_extra: dict,
    recursive: bool,
    limit: int,
    skip_multimodal: bool,
) -> tuple[int, int]:
    files = _collect_files(
        input_folder, config.supported_file_extensions, recursive
    )
    if not files:
        raise SystemExit(
            f"No supported files under {input_folder} "
            f"(extensions: {config.supported_file_extensions})"
        )
    if limit > 0:
        files = files[:limit]

    ok = fail = 0
    for fp in files:
        try:
            rel = str(fp.relative_to(input_folder))
            sub_out = parser_output_dir
            if fp.parent != input_folder:
                sub_out = parser_output_dir / fp.parent.relative_to(input_folder)
            sub_out.mkdir(parents=True, exist_ok=True)

            content_list, doc_id = await rag.parse_document(
                str(fp),
                output_dir=str(sub_out),
                parse_method=parse_method,
                display_stats=config.display_content_stats,
                **parse_extra,
            )
            await rag.insert_content_list(
                content_list,
                file_path=rel,
                doc_id=doc_id,
                skip_multimodal_processing=skip_multimodal,
            )
            ok += 1
            logger.info(f"INGEST_FILE_OK::{rel}")
        except Exception as e:
            logger.error(f"INGEST_FILE_FAIL::{fp}: {e}")
            fail += 1

    logger.info(f"INGEST_DONE::ok={ok}::fail={fail}")
    return ok, fail


async def _interactive_loop(rag, query_mode: str) -> None:
    from lightrag.utils import logger

    print(
        "Ready. Type a question and press Enter. Empty line or 'exit' / 'quit' ends.\n",
        flush=True,
    )
    while True:
        q = await asyncio.to_thread(input, "Q> ")
        q = (q or "").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        try:
            ans = await rag.aquery(q, mode=query_mode, vlm_enhanced=False)
            print(ans or "", flush=True)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            print(f"[error] {e}", flush=True)


async def async_main() -> None:
    _ensure_venv_bin_on_path()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Folder containing documents (PDF, Office, images, …).",
    )
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "rag_storage_pipeline",
        help="LightRAG persistence directory.",
    )
    p.add_argument(
        "--parser-output-dir",
        type=Path,
        default=_ROOT / "output" / "pipeline_parse",
        help="MinerU / parser output directory.",
    )
    p.add_argument(
        "--parse-method",
        type=str,
        default=os.getenv("PARSE_METHOD", "auto"),
        help="MinerU method: auto, ocr, txt, … (default: env PARSE_METHOD or auto).",
    )
    p.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subfolders for documents (default: True).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N files (0 = all).",
    )
    p.add_argument(
        "--skip-multimodal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After text LightRAG insert, skip multimodal processors (default: True).",
    )
    p.add_argument(
        "--query-mode",
        type=str,
        default=os.getenv("RAG_QUERY_MODE", "mix"),
        help="LightRAG query mode (default: env RAG_QUERY_MODE or mix).",
    )
    p.add_argument(
        "--ingest-only",
        action="store_true",
        help="Parse + graph ingest only; do not start the question loop.",
    )
    p.add_argument(
        "--query",
        type=str,
        default="",
        help="Single non-interactive question after ingest (then exit unless --ingest-only).",
    )
    p.add_argument(
        "--mineru-download-models",
        action="store_true",
        help="Run mineru-models-download pipeline weights before parsing.",
    )
    args = p.parse_args()

    input_folder = args.input_folder.expanduser().resolve()
    if not input_folder.is_dir():
        raise SystemExit(f"Not a directory: {input_folder}")

    args.working_dir = args.working_dir.expanduser().resolve()
    args.parser_output_dir = args.parser_output_dir.expanduser().resolve()
    args.parser_output_dir.mkdir(parents=True, exist_ok=True)

    if args.mineru_download_models:
        _download_mineru_pipeline_models()

    rag, config, logger = await _build_rag(
        args.working_dir,
        args.parser_output_dir,
    )
    parse_extra = _mineru_parse_kwargs(config.parser)

    await _ingest_folder(
        rag,
        config,
        logger,
        input_folder=input_folder,
        parser_output_dir=args.parser_output_dir,
        parse_method=args.parse_method,
        parse_extra=parse_extra,
        recursive=args.recursive,
        limit=args.limit,
        skip_multimodal=args.skip_multimodal,
    )
    await rag.finalize_storages()

    if args.ingest_only:
        return

    if args.query.strip():
        ans = await rag.aquery(
            args.query.strip(),
            mode=args.query_mode,
            vlm_enhanced=False,
        )
        print(ans or "", flush=True)
        return

    await _interactive_loop(rag, args.query_mode)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
