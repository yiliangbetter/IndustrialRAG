#!/usr/bin/env python3
"""Parse Documents/ with MinerU, build LightRAG vector + knowledge graph, then Q&A.

Uses Ollama for LLM and embeddings (see examples/ollama_integration_example.py).
Run from repo root:

  python3 scripts/ingest_documents_local_ollama.py

  # ingest only (no chat):
  python3 scripts/ingest_documents_local_ollama.py --ingest-only

  # one-shot question after ingest:
  python3 scripts/ingest_documents_local_ollama.py --query "封边机日常保养有哪些步骤？"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, TiktokenTokenizer, logger
from raganything import RAGAnything, RAGAnythingConfig

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", OLLAMA_LLM_MODEL)
# bge-m3 supports ~8192 tokens; bge-large-zh-v1.5 is limited to 512 (use CHUNK_TOKEN_SIZE<=400).
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
OLLAMA_EMBEDDING_DIM = int(os.getenv("OLLAMA_EMBEDDING_DIM", "1024"))
OLLAMA_EMBEDDING_MAX_TOKENS = int(os.getenv("OLLAMA_EMBEDDING_MAX_TOKENS", "8192"))
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "1200"))
CHUNK_OVERLAP_TOKEN_SIZE = int(os.getenv("CHUNK_OVERLAP_TOKEN_SIZE", "100"))

_embed_tokenizer = TiktokenTokenizer()
# Conservative cap for models with 512-token limits (tiktoken under-counts Chinese).
_BGE_SHORT_CONTEXT_MODELS = ("bge-large-zh", "bge-large-zh-v15")
OLLAMA_BASE_URL = f"{OLLAMA_HOST}/v1"
OLLAMA_API_KEY = "ollama"

DEFAULT_INPUT = _ROOT / "Documents"
DEFAULT_WORKING = _ROOT / "rag_storage_documents_v3"
DEFAULT_PARSER_OUT = _ROOT / "output" / "documents_mineru"


def _ensure_venv_bin_on_path() -> None:
    venv_bin = _ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")


def _mineru_parse_kwargs() -> dict:
    out: dict = {}
    lang = os.getenv("MINERU_LANG", "ch").strip() or os.getenv("OCR_LANG", "ch").strip()
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
    if "backend" not in out:
        out["backend"] = "pipeline"
    return out


def _collect_pdfs(
    folder: Path, recursive: bool, only_names: list[str] | None = None
) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    files = sorted({p.resolve() for p in folder.glob(pattern) if p.is_file()})
    if only_names:
        wanted = {n.strip() for n in only_names if n.strip()}
        files = [p for p in files if p.name in wanted]
    return files


async def ollama_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        model=OLLAMA_LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        **kwargs,
    )


async def ollama_vision_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    image_data: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    **kwargs,
) -> str:
    if messages:
        return await openai_complete_if_cache(
            model=OLLAMA_VISION_MODEL,
            prompt="",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            **kwargs,
        )
    if image_data:
        return await openai_complete_if_cache(
            model=OLLAMA_VISION_MODEL,
            prompt="",
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
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            **kwargs,
        )
    return await ollama_llm_model_func(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def _truncate_for_embed(text: str, max_tokens: int) -> str:
    model = OLLAMA_EMBEDDING_MODEL.lower()
    if any(m in model for m in _BGE_SHORT_CONTEXT_MODELS):
        max_tokens = min(max_tokens, 380)
    tokens = _embed_tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _embed_tokenizer.decode(tokens[:max_tokens])


async def ollama_embedding_async(
    texts: List[str],
    max_token_size: int | None = None,
    **kwargs,
) -> np.ndarray:
    import ollama

    limit = max_token_size or OLLAMA_EMBEDDING_MAX_TOKENS
    safe_texts = [_truncate_for_embed(t, limit) for t in texts]
    client = ollama.AsyncClient(host=OLLAMA_HOST)
    response = await client.embed(model=OLLAMA_EMBEDDING_MODEL, input=safe_texts)
    return np.array(response.embeddings, dtype=np.float32)


def _embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=OLLAMA_EMBEDDING_DIM,
        max_token_size=OLLAMA_EMBEDDING_MAX_TOKENS,
        func=ollama_embedding_async,
    )


async def _build_rag(working_dir: Path, parser_output_dir: Path) -> RAGAnything:
    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        allow_embedding_only_ingestion=False,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        parser_output_dir=str(parser_output_dir),
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=int(os.getenv("MAX_CONCURRENT_FILES", "1")),
    )

    embedding_func = _embedding_func()
    lightrag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_llm_model_func,
        embedding_func=embedding_func,
        enable_llm_cache=True,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "2")),
        embedding_batch_num=int(os.getenv("EMBEDDING_BATCH_NUM", "8")),
    )
    await lightrag.initialize_storages()

    return RAGAnything(
        config=config,
        lightrag=lightrag,
        llm_model_func=ollama_llm_model_func,
        vision_model_func=ollama_vision_model_func,
        embedding_func=embedding_func,
    )


async def _ingest_folder(
    rag: RAGAnything,
    *,
    input_folder: Path,
    parser_output_dir: Path,
    parse_method: str,
    parse_extra: dict,
    recursive: bool,
    limit: int,
    skip_multimodal: bool,
    only_names: list[str] | None = None,
) -> tuple[int, int]:
    files = _collect_pdfs(input_folder, recursive, only_names)
    if not files:
        raise SystemExit(f"No PDF files under {input_folder}")
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

            logger.info(f"Parsing with MinerU: {rel}")
            content_list, doc_id = await rag.parse_document(
                str(fp),
                output_dir=str(sub_out),
                parse_method=parse_method,
                display_stats=True,
                **parse_extra,
            )
            logger.info(f"Ingesting into vector + graph store: {rel}")
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


async def _interactive_loop(rag: RAGAnything, query_mode: str) -> None:
    print(
        "Ready. Type a question (Chinese or English). Empty line or exit/quit ends.\n",
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
        default=DEFAULT_INPUT,
        help=f"Folder with PDFs (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING,
        help=f"LightRAG storage (default: {DEFAULT_WORKING})",
    )
    p.add_argument(
        "--parser-output-dir",
        type=Path,
        default=DEFAULT_PARSER_OUT,
        help=f"MinerU output (default: {DEFAULT_PARSER_OUT})",
    )
    p.add_argument(
        "--parse-method",
        default=os.getenv("PARSE_METHOD", "auto"),
        help="MinerU method: auto, ocr, txt",
    )
    p.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--limit", type=int, default=0, help="Max files (0=all)")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated PDF basenames to process (default: all).",
    )
    p.add_argument(
        "--skip-multimodal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--query-mode",
        default=os.getenv("RAG_QUERY_MODE", "mix"),
        help="LightRAG mode: local, global, hybrid, mix, …",
    )
    p.add_argument("--ingest-only", action="store_true")
    p.add_argument("--query", type=str, default="")
    args = p.parse_args()

    input_folder = args.input_folder.expanduser().resolve()
    if not input_folder.is_dir():
        raise SystemExit(f"Not a directory: {input_folder}")

    args.working_dir.mkdir(parents=True, exist_ok=True)
    args.parser_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:   {input_folder}")
    print(f"Storage: {args.working_dir}")
    print(f"MinerU:  {args.parser_output_dir}")
    print(
        f"Ollama:  LLM={OLLAMA_LLM_MODEL} vision={OLLAMA_VISION_MODEL} "
        f"embed={OLLAMA_EMBEDDING_MODEL}"
    )
    print(flush=True)

    rag = await _build_rag(args.working_dir, args.parser_output_dir)
    parse_extra = _mineru_parse_kwargs()

    only_names = [n.strip() for n in args.only.split(",") if n.strip()] or None

    ok, fail = await _ingest_folder(
        rag,
        input_folder=input_folder,
        parser_output_dir=args.parser_output_dir,
        parse_method=args.parse_method,
        parse_extra=parse_extra,
        recursive=args.recursive,
        limit=args.limit,
        skip_multimodal=args.skip_multimodal,
        only_names=only_names,
    )
    await rag.finalize_storages()

    print(f"\nIngestion finished: {ok} ok, {fail} failed.", flush=True)
    if fail and ok == 0:
        raise SystemExit(1)

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
