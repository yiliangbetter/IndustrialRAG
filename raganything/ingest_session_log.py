"""Persistent session logs for Web/CLI ingest (``logs/ingest/``)."""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOG_DIR = _REPO_ROOT / "logs" / "ingest"
_LOGGER_NAMES = (
    "lightrag",
    "raganything",
    "raganything.parser",
    "raganything.processor",
)


def ingest_log_enabled() -> bool:
    raw = (os.getenv("RAG_INGEST_LOG") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def ingest_log_dir() -> Path:
    raw = (os.getenv("RAG_INGEST_LOG_DIR") or "").strip()
    return Path(raw) if raw else _DEFAULT_LOG_DIR


class _IngestFileHandler(logging.Handler):
    def __init__(self, session: IngestSessionLog) -> None:
        super().__init__(level=logging.INFO)
        self._session = session

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{''.join(traceback.format_exception(*record.exc_info))}"
        self._session.write(
            "BACKEND",
            level=record.levelname,
            logger=record.name,
            message=msg,
        )


class IngestSessionLog:
    """Append-only ingest trace: UI events + backend logger lines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8", buffering=1)
        self._handler: logging.Handler | None = None
        self._attached: list[logging.Logger] = []

    @classmethod
    def start(
        cls,
        *,
        files: list[str] | None = None,
        working_dir: str = "",
        parser_dir: str = "",
    ) -> IngestSessionLog | None:
        if not ingest_log_enabled():
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ingest_log_dir() / f"{ts}.log"
        sess = cls(path)
        sess.write(
            "SESSION_START",
            files=files or [],
            working_dir=working_dir,
            parser_dir=parser_dir,
        )
        sess.attach_backend_loggers()
        try:
            rel = path.relative_to(_REPO_ROOT)
            latest_ref = rel.as_posix()
        except ValueError:
            latest_ref = str(path)
        (ingest_log_dir() / "latest.path").write_text(
            latest_ref + "\n", encoding="utf-8"
        )
        return sess

    def attach_backend_loggers(self) -> None:
        handler = _IngestFileHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._handler = handler
        for name in _LOGGER_NAMES:
            log = logging.getLogger(name)
            log.addHandler(handler)
            self._attached.append(log)

    def write(self, kind: str, **fields: Any) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {"ts": ts, "kind": kind, **fields}
        line = json.dumps(payload, ensure_ascii=False, default=str)
        self._fp.write(line + "\n")

    def event(self, ev: dict[str, Any]) -> None:
        kind = str(ev.get("type") or "event")
        row: dict[str, Any] = {"event": kind}
        for key in (
            "message",
            "file",
            "error",
            "current",
            "total",
            "ok",
            "fail",
            "files",
            "rollback_removed",
        ):
            if key in ev and ev[key] is not None:
                row[key] = ev[key]
        if kind == "file_fail" and ev.get("error"):
            row["level"] = "ERROR"
        elif kind == "file_ok":
            row["level"] = "OK"
        else:
            row["level"] = "INFO"
        self.write("UI", **row)

    def exception(self, label: str, exc: BaseException) -> None:
        self.write(
            "EXCEPTION",
            level="ERROR",
            label=label,
            message=str(exc),
            traceback=traceback.format_exc(),
        )

    def finalize(
        self,
        *,
        ok: int,
        fail: int,
        errors: list[dict[str, str]],
        cancelled: bool,
        rollback_removed: int = 0,
    ) -> None:
        self.write(
            "SESSION_END",
            level="ERROR" if fail else "OK",
            ok=ok,
            fail=fail,
            cancelled=cancelled,
            rollback_removed=rollback_removed,
            errors=errors,
            log_path=str(self.path),
        )

    def close(self) -> None:
        if self._handler is not None:
            for log in self._attached:
                log.removeHandler(self._handler)
            self._handler = None
            self._attached.clear()
        self._fp.close()
