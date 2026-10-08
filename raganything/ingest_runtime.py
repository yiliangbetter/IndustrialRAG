"""Cooperative ingest cancel: flag check + optional MinerU subprocess kill."""

from __future__ import annotations

import subprocess
import threading
from typing import Callable


class IngestCancelledError(Exception):
    """Raised when ingest is aborted while parsing (e.g. MinerU subprocess killed)."""


_lock = threading.Lock()
_cancel_check: Callable[[], bool] | None = None
_active_process: subprocess.Popen | None = None


def set_ingest_cancel_check(check: Callable[[], bool] | None) -> None:
    global _cancel_check
    with _lock:
        _cancel_check = check


def ingest_cancel_requested() -> bool:
    with _lock:
        check = _cancel_check
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:
        return False


def register_ingest_subprocess(process: subprocess.Popen | None) -> None:
    global _active_process
    with _lock:
        _active_process = process


def clear_ingest_subprocess() -> None:
    register_ingest_subprocess(None)


def abort_active_ingest_subprocess() -> bool:
    """Kill the running parser subprocess when cancel was requested."""
    with _lock:
        proc = _active_process
        check = _cancel_check
    if proc is None or proc.poll() is not None:
        return False
    if check is None or not check():
        return False
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    return True
