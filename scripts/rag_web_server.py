#!/usr/bin/env python3
"""Browser UI server for RAG-Anything (FastAPI + static web).

Reuses ``scripts/rag_pipeline_parse_graph_chat`` for ``_build_rag``, ingest, and query.

Examples::

  uv sync --extra web
  uv run python scripts/rag_web_server.py

  # Open http://127.0.0.1:8765

Env (optional):

  RAG_WEB_HOST=127.0.0.1
  RAG_WEB_PORT=8765
  RAG_WEB_WORKING_DIR=rag_storage_run
  RAG_WEB_PARSER_OUTPUT_DIR=output/pipeline_parse
  RAG_WEB_SKIP_MULTIMODAL=true
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

from client_paths import (  # noqa: E402
    apply_client_env_defaults,
    is_client_mode,
    is_setup_complete,
    write_setup_complete,
)

apply_client_env_defaults()
load_dotenv(_ROOT / ".env", override=False)

from client_env_manager import apply_env_to_process, get_form_values, save_env  # noqa: E402
from client_paths import get_env_path  # noqa: E402
from client_setup_service import get_setup_status  # noqa: E402

if get_env_path().is_file():
    apply_env_to_process()

if (os.getenv("HF_EMBED_OFFLINE") or "").strip().lower() in ("1", "true", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_WEB_DIR = _ROOT / "web"
_RPC: Any = None


def _load_rpc():
    global _RPC
    if _RPC is not None:
        return _RPC
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_graph_chat",
        _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _RPC = mod
    return mod


def _resolve_path(env_key: str, default: str) -> Path:
    raw = (os.getenv(env_key) or default).strip()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_ROOT / p).resolve()
    return p


class QueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    mode: str | None = Field(None, description="LightRAG mode; default from RAG_QUERY_MODE")
    stream: bool = Field(
        True,
        description="When true, prefer /api/query/stream; ignored on non-stream endpoint.",
    )


class SetupEnvBody(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _iter_llm_chunks(result: str | AsyncIterator[str]) -> AsyncIterator[str]:
    if isinstance(result, str):
        if result:
            yield result
        return
    if hasattr(result, "__aiter__"):
        async for chunk in result:
            if chunk:
                yield chunk if isinstance(chunk, str) else str(chunk)
        return
    for chunk in result:
        if chunk:
            yield chunk if isinstance(chunk, str) else str(chunk)


async def _run_aquery(q: str, mode: str, *, stream: bool) -> str | AsyncIterator[str]:
    rpc = _load_rpc()
    async with state.lock:
        return await state.rag.aquery(
            q,
            mode=mode,
            stream=stream,
            vlm_enhanced=False,
            **rpc._query_extras_from_env(q),
        )


class AppState:
    rag: Any = None
    config: Any = None
    working_dir: str = ""
    parser_output_dir: str = ""
    query_mode: str = "mix"
    skip_multimodal: bool = True
    lock: asyncio.Lock = asyncio.Lock()
    ready: bool = False
    init_error: str | None = None
    ingest_active: bool = False
    ingest_cancel_requested: bool = False


state = AppState()


def _wipe_directory(path: Path) -> None:
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


async def _shutdown_priority_workers(func: Any) -> None:
    """Gracefully stop LightRAG priority queue workers attached to a callable."""
    seen: set[int] = set()

    async def _walk(obj: Any) -> None:
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        shutdown = getattr(obj, "shutdown", None)
        if callable(shutdown):
            try:
                result = shutdown()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

        for attr in ("func", "__wrapped__"):
            inner = getattr(obj, attr, None)
            if inner is not None:
                await _walk(inner)

    await _walk(func)


async def _shutdown_rag_instance(rag: Any) -> None:
    if rag is None:
        return
    lightrag = getattr(rag, "lightrag", None)
    if lightrag is not None:
        await _shutdown_priority_workers(getattr(lightrag, "llm_model_func", None))
        embedding = getattr(lightrag, "embedding_func", None)
        if embedding is not None:
            await _shutdown_priority_workers(
                getattr(embedding, "func", embedding)
            )
    try:
        await rag.finalize_storages()
    except Exception:
        pass
    await asyncio.sleep(0)


async def _shutdown_rag() -> None:
    rag = state.rag
    state.rag = None
    state.config = None
    state.ready = False
    state.init_error = None
    if rag is not None:
        await _shutdown_rag_instance(rag)


async def _clear_knowledge_base() -> None:
    wd = Path(state.working_dir) if state.working_dir else _resolve_path(
        "RAG_WEB_WORKING_DIR", "rag_storage_run"
    )
    pod = Path(state.parser_output_dir) if state.parser_output_dir else _resolve_path(
        "RAG_WEB_PARSER_OUTPUT_DIR", "output/pipeline_parse"
    )
    async with state.lock:
        await _shutdown_rag()
        _wipe_directory(wd)
        _wipe_directory(pod)
        pod.mkdir(parents=True, exist_ok=True)
    await _init_rag_engine()


async def _init_rag_engine() -> None:
    rpc = _load_rpc()
    wd = _resolve_path("RAG_WEB_WORKING_DIR", "rag_storage_run")
    pod = _resolve_path("RAG_WEB_PARSER_OUTPUT_DIR", "output/pipeline_parse")
    pod.mkdir(parents=True, exist_ok=True)
    skip_mm = (os.getenv("RAG_WEB_SKIP_MULTIMODAL", "true") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    state.working_dir = str(wd)
    state.parser_output_dir = str(pod)
    state.query_mode = (os.getenv("RAG_QUERY_MODE") or "mix").strip()
    state.skip_multimodal = skip_mm
    try:
        rag, config, _logger = await rpc._build_rag(wd, pod, skip_multimodal=skip_mm)
        state.rag = rag
        state.config = config
        state.ready = True
        state.init_error = None
    except Exception as exc:
        state.init_error = str(exc)
        state.ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if is_client_mode() and not is_setup_complete():
        yield
        return
    await _init_rag_engine()
    try:
        yield
    finally:
        await _shutdown_rag()


app = FastAPI(title="Nanxing RAG Client", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (_WEB_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")


@app.get("/")
async def index_page():
    if is_client_mode() and not is_setup_complete():
        return RedirectResponse("/setup", status_code=302)
    index = _WEB_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(500, "web/index.html missing")
    return FileResponse(index)


@app.get("/setup")
async def setup_page():
    setup = _WEB_DIR / "setup.html"
    if not setup.is_file():
        raise HTTPException(500, "web/setup.html missing")
    return FileResponse(setup)


@app.get("/api/setup/status")
async def api_setup_status():
    status = get_setup_status()
    if (
        not state.ready
        and state.rag is None
        and not state.init_error
        and get_env_path().is_file()
        and status.get("env", {}).get("ok")
    ):
        await _init_rag_engine()
    status["rag_ready"] = state.ready
    status["rag_error"] = state.init_error
    return status


@app.get("/api/setup/env")
async def api_setup_env_get():
    return {"fields": get_form_values(), "env_path": str(get_env_path())}


@app.post("/api/setup/env")
async def api_setup_env_save(body: SetupEnvBody):
    try:
        path = save_env(body.values)
        apply_env_to_process()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await _shutdown_rag()
    await _init_rag_engine()
    return {
        "ok": True,
        "env_path": str(path),
        "rag_ready": state.ready,
        "rag_error": state.init_error,
    }


@app.post("/api/setup/reload-rag")
async def api_setup_reload_rag():
    apply_env_to_process()
    await _shutdown_rag()
    await _init_rag_engine()
    return {
        "rag_ready": state.ready,
        "rag_error": state.init_error,
        "working_dir": state.working_dir,
    }


@app.post("/api/setup/complete")
async def api_setup_complete():
    status = get_setup_status()
    if not status.get("can_finish_setup"):
        raise HTTPException(
            400,
            "尚未满足完成条件：请保存配置、确认内置模型就绪并完成至少一次灌库。",
        )
    marker = write_setup_complete()
    if not state.ready:
        apply_env_to_process()
        await _init_rag_engine()
    return {"ok": True, "marker": marker, "redirect": "/"}


@app.get("/api/health")
async def health():
    return {
        "ready": state.ready,
        "init_error": state.init_error,
        "working_dir": state.working_dir,
        "parser_output_dir": state.parser_output_dir,
        "query_mode": state.query_mode,
        "skip_multimodal": state.skip_multimodal,
        "client_mode": is_client_mode(),
        "setup_complete": is_setup_complete(),
    }


@app.post("/api/query")
async def api_query(body: QueryBody):
    if not state.ready or state.rag is None:
        raise HTTPException(
            503,
            state.init_error or "RAG engine not initialized; check server logs and .env",
        )
    mode = (body.mode or state.query_mode or "mix").strip()
    q = body.query.strip()
    try:
        answer = await _run_aquery(q, mode, stream=False)
    except Exception as exc:
        raise HTTPException(500, f"Query failed: {exc}") from exc
    if not isinstance(answer, str):
        parts: list[str] = []
        async for chunk in _iter_llm_chunks(answer):
            parts.append(chunk)
        answer = "".join(parts)
    _scripts_dir = _ROOT / "scripts"
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    from stream_cot_parser import parse_complete_cot  # noqa: WPS433

    thinking, final_answer = parse_complete_cot(answer or "")
    return {
        "thinking": thinking,
        "answer": final_answer,
        "mode": mode,
        "query": q,
    }


async def _query_stream_events(q: str, mode: str) -> AsyncIterator[str]:
    """SSE tied to real LightRAG stages (retrieve → rerank → generate), then token deltas."""
    _scripts_dir = _ROOT / "scripts"
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    from query_progress_hooks import query_progress_hooks  # noqa: WPS433
    from stream_cot_parser import StreamCotParser  # noqa: WPS433

    result_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=1)

    async with query_progress_hooks() as progress_queue:
        async def _worker() -> None:
            try:
                result = await _run_aquery(q, mode, stream=True)
                await result_queue.put(("ok", result))
            except Exception as exc:
                await result_queue.put(("error", exc))

        worker = asyncio.create_task(_worker())
        cot_parser = StreamCotParser()

        try:
            while True:
                prog_task = asyncio.create_task(progress_queue.get())
                res_task = asyncio.create_task(result_queue.get())
                done, pending = await asyncio.wait(
                    {prog_task, res_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()

                if prog_task in done and not prog_task.cancelled():
                    try:
                        ev = prog_task.result()
                        if ev.get("type") in ("status", "retrieval_scope"):
                            yield _sse(ev)
                    except Exception:
                        pass

                if res_task in done and not res_task.cancelled():
                    kind, payload = res_task.result()
                    if kind == "error":
                        yield _sse({"type": "error", "message": str(payload)})
                        return

                    async for chunk in _iter_llm_chunks(payload):
                        for kind, piece in cot_parser.feed(chunk):
                            if not piece:
                                continue
                            if kind == "thinking":
                                yield _sse({"type": "thinking_delta", "text": piece})
                            else:
                                yield _sse({"type": "answer_delta", "text": piece})
                    for kind, piece in cot_parser.flush():
                        if not piece:
                            continue
                        if kind == "thinking":
                            yield _sse({"type": "thinking_delta", "text": piece})
                        else:
                            yield _sse({"type": "answer_delta", "text": piece})
                    yield _sse({"type": "done", "mode": mode})
                    return
        finally:
            if not worker.done():
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass


@app.post("/api/query/stream")
async def api_query_stream(body: QueryBody):
    if not state.ready or state.rag is None:
        raise HTTPException(
            503,
            state.init_error or "RAG engine not initialized; check server logs and .env",
        )
    mode = (body.mode or state.query_mode or "mix").strip()
    q = body.query.strip()
    return StreamingResponse(
        _query_stream_events(q, mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ingest")
async def api_ingest(files: list[UploadFile] = File(...)):
    if not state.ready or state.rag is None:
        raise HTTPException(503, state.init_error or "RAG engine not initialized")
    if not files:
        raise HTTPException(400, "No files uploaded")

    tmp_root = Path(tempfile.mkdtemp(prefix="rag_web_ingest_"))
    try:
        saved = await _save_uploaded_files(files, tmp_root)
        ok, fail, errors, cancelled = await _run_ingest_on_folder(tmp_root)
        return {
            "ok": ok,
            "fail": fail,
            "files": [p.name for p in saved],
            "errors": errors,
            "cancelled": cancelled,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Ingest failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


async def _save_uploaded_files(files: list[UploadFile], tmp_root: Path) -> list[Path]:
    saved: list[Path] = []
    for uf in files:
        name = Path(uf.filename or "upload").name
        if name in (".", "..") or "/" in name or "\\" in name:
            raise HTTPException(400, f"Invalid filename: {uf.filename}")
        dest = tmp_root / name
        with dest.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        saved.append(dest)
    return saved


async def _run_ingest_on_folder(
    input_folder: Path,
    on_event=None,
) -> tuple[int, int, list[dict[str, str]], bool]:
    rpc = _load_rpc()
    pod = Path(state.parser_output_dir)
    parse_method = (os.getenv("PARSE_METHOD") or "auto").strip()
    parse_extra = rpc._mineru_parse_kwargs(state.config.parser)
    logger = __import__("lightrag.utils", fromlist=["logger"]).logger

    async with state.lock:
        ok, fail, errors, cancelled = await rpc._ingest_folder(
            state.rag,
            state.config,
            logger,
            input_folder=input_folder,
            parser_output_dir=pod,
            parse_method=parse_method,
            parse_extra=parse_extra,
            recursive=False,
            limit=0,
            skip_multimodal=state.skip_multimodal,
            on_event=on_event,
            should_cancel=lambda: state.ingest_cancel_requested,
        )
        if not cancelled and state.rag is not None:
            await state.rag.finalize_storages()

    if cancelled:
        await _clear_knowledge_base()

    return ok, fail, errors, cancelled


async def _ingest_stream_events(files: list[UploadFile]) -> AsyncIterator[str]:
    tmp_root = Path(tempfile.mkdtemp(prefix="rag_web_ingest_"))
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    state.ingest_active = True
    state.ingest_cancel_requested = False

    async def on_event(ev: dict[str, Any]) -> None:
        await queue.put(ev)

    async def worker() -> None:
        try:
            saved = await _save_uploaded_files(files, tmp_root)
            await queue.put(
                {
                    "type": "ingest_saved",
                    "files": [p.name for p in saved],
                    "message": f"已接收 {len(saved)} 个文件，开始灌库…",
                }
            )
            ok, fail, errors, cancelled = await _run_ingest_on_folder(
                tmp_root, on_event=on_event
            )
            if cancelled:
                await queue.put(
                    {
                        "type": "cancelled",
                        "ok": ok,
                        "fail": fail,
                        "errors": errors,
                        "message": "已停止灌库并清空知识库",
                        "storage_cleared": True,
                    }
                )
            else:
                await queue.put(
                    {
                        "type": "done",
                        "ok": ok,
                        "fail": fail,
                        "errors": errors,
                    }
                )
        except Exception as exc:
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
            state.ingest_active = False
            state.ingest_cancel_requested = False

    task = asyncio.create_task(worker())
    try:
        while True:
            ev = await queue.get()
            yield _sse(ev)
            if ev.get("type") in ("done", "error", "cancelled"):
                break
    finally:
        if not task.done():
            if state.ingest_cancel_requested:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@app.post("/api/ingest/cancel")
async def api_ingest_cancel():
    if not state.ingest_active:
        return {"ok": True, "active": False, "message": "当前没有进行中的灌库任务"}
    state.ingest_cancel_requested = True
    return {
        "ok": True,
        "active": True,
        "message": "已请求停止；当前文件处理完成后将终止并清空知识库",
    }


@app.post("/api/ingest/stream")
async def api_ingest_stream(files: list[UploadFile] = File(...)):
    if not state.ready or state.rag is None:
        raise HTTPException(503, state.init_error or "RAG engine not initialized")
    if not files:
        raise HTTPException(400, "No files uploaded")
    return StreamingResponse(
        _ingest_stream_events(files),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    host = (os.getenv("RAG_WEB_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("RAG_WEB_PORT") or "8765").strip())
    print(f"Open http://{host}:{port}/ in your browser", flush=True)
    uvicorn.run(app, host=host, port=port, reload=False)
