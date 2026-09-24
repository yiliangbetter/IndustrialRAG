#!/usr/bin/env python
"""Batch RAG answers for question/Demo问题库.xlsx → Demo问题库_回答结果.xlsx + .jsonl."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

from raganything import RAGAnythingConfig
from raganything.runtime_factory import RuntimeOptions, create_rag_runtime


def _json_safe(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


async def run_batch(
    input_xlsx: Path,
    out_xlsx: Path,
    out_jsonl: Path,
    working_dir: Path,
    mode: str,
    delay_seconds: float,
    embedding_func_max_async: int,
    embedding_batch_num: int,
    limit_questions: int,
    timing_log: Path | None,
) -> None:
    import openpyxl

    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        parser=os.getenv("PARSER", "mineru"),
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    runtime = await create_rag_runtime(
        config,
        RuntimeOptions(
            project_root=_ROOT,
            embedding_func_max_async=embedding_func_max_async,
            embedding_batch_num=embedding_batch_num,
            enable_rerank=True,
            await_model_calls=False,
        ),
    )
    rag, logger = runtime.rag, runtime.logger

    wb_in = openpyxl.load_workbook(input_xlsx)
    ws_in = wb_in.active
    headers = [c.value for c in next(ws_in.iter_rows(min_row=1, max_row=1))]
    try:
        q_col = headers.index("问题") + 1
    except ValueError as e:
        raise SystemExit(f'Column "问题" not found; got {headers!r}') from e

    if "RAG回答" not in headers:
        r_col = len(headers) + 1
        ws_in.cell(row=1, column=r_col, value="RAG回答")
    else:
        r_col = headers.index("RAG回答") + 1

    records: list[dict] = []
    rows = list(ws_in.iter_rows(min_row=2, values_only=False))
    answered = 0
    for row_cells in rows:
        if limit_questions > 0 and answered >= limit_questions:
            break
        q_cell = row_cells[q_col - 1]
        question = q_cell.value
        if question is None or (isinstance(question, str) and not question.strip()):
            continue
        question = str(question).strip()
        row_num = q_cell.row
        logger.info(f"[{row_num}] Q: {question[:80]}...")
        t0 = time.perf_counter()
        answer = await rag.aquery(
            question,
            mode=mode,
            vlm_enhanced=False,
        )
        q_secs = time.perf_counter() - t0
        if timing_log is not None:
            timing_log.parent.mkdir(parents=True, exist_ok=True)
            with open(timing_log, "a", encoding="utf-8") as tf:
                tf.write(
                    json.dumps(
                        {
                            "row": row_num,
                            "question_chars": len(question),
                            "query_seconds": round(q_secs, 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if answer is None:
            answer = ""
        ws_in.cell(row=row_num, column=r_col, value=answer)
        row_vals = [c.value for c in row_cells]
        rec = {
            headers[i]: _json_safe(row_vals[i]) if i < len(row_vals) else None
            for i in range(len(headers))
        }
        rec["RAG回答"] = answer
        records.append(rec)
        answered += 1
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    wb_in.save(out_xlsx)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    logger.info(f"Wrote {len(records)} rows → {out_xlsx} and {out_jsonl}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "question" / "Demo问题库.xlsx",
    )
    p.add_argument(
        "--out-xlsx",
        type=Path,
        default=_ROOT / "question" / "Demo问题库_回答结果.xlsx",
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=_ROOT / "question" / "Demo问题库_回答结果.jsonl",
    )
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "rag_storage",
    )
    p.add_argument(
        "--mode",
        default=os.getenv("RAG_QUERY_MODE", "mix"),
        help="LightRAG query mode (default: mix)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("DEMO_QA_QUERY_DELAY_SECONDS", "4")),
        help="Seconds to sleep after each answer (helps avoid embedding rate limits).",
    )
    p.add_argument(
        "--embedding-max-async",
        type=int,
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "1")),
        help="Concurrency for embedding RPCs (lower helps with Ark rate limits).",
    )
    p.add_argument(
        "--embedding-batch-num",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_NUM", "1")),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Answer at most N non-empty questions from the sheet (0 = all).",
    )
    p.add_argument(
        "--timing-log",
        type=Path,
        default=None,
        help="Append one JSON line per answered question with row + query_seconds (wall time for aquery).",
    )
    args = p.parse_args()
    asyncio.run(
        run_batch(
            args.input,
            args.out_xlsx,
            args.out_jsonl,
            args.working_dir,
            args.mode,
            args.delay,
            args.embedding_max_async,
            args.embedding_batch_num,
            args.limit,
            args.timing_log,
        )
    )


if __name__ == "__main__":
    main()
