#!/usr/bin/env python3
"""Nanxing RAG client launcher (portable green package entry point).

Sets client layout paths, starts the local web server, and opens the setup
wizard or chat UI in the default browser.

Examples::

  # Development (from repo root, uses data/ + config/ under repo)
  uv run python scripts/client_launcher.py

  # Packaged layout: set RAG_CLIENT_APP_ROOT to the extracted folder
  set RAG_CLIENT_APP_ROOT=D:\\NanxingRAG
  NanxingRAG.exe
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

os.environ.setdefault("RAG_CLIENT_MODE", "1")
os.environ.setdefault("RAG_CLIENT_APP_ROOT", str(_ROOT))

from client_paths import apply_client_env_defaults, get_app_root, is_setup_complete  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

apply_client_env_defaults()


def _load_env() -> None:
    from client_env_manager import apply_env_to_process
    from client_paths import get_env_path

    if get_env_path().is_file():
        apply_env_to_process()
    else:
        load_dotenv(_ROOT / ".env", override=False)


def main() -> None:
    _load_env()
    host = (os.getenv("RAG_WEB_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("RAG_WEB_PORT") or "8765").strip())
    path = "/setup" if not is_setup_complete() else "/"
    url = f"http://{host}:{port}{path}"

    def _open_browser() -> None:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except OSError:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"Nanxing RAG client — app root: {get_app_root()}", flush=True)
    print(f"Open {url} in your browser", flush=True)

    import uvicorn

    uvicorn.run(
        "rag_web_server:app",
        host=host,
        port=port,
        reload=False,
        app_dir=str(_SCRIPTS),
    )


if __name__ == "__main__":
    main()
