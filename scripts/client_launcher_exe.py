#!/usr/bin/env python3
"""Thin NanxingRAG.exe entry point (PyInstaller onefile).

Does not bundle torch/MinerU — starts the pre-built ``runtime/`` Python and
``scripts/client_launcher.py`` beside this executable.  End users double-click
only; no uv, pip, or command line required.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _show_error(title: str, message: str) -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
            return
        except OSError:
            pass
    print(f"{title}\n{message}", file=sys.stderr)
    try:
        input("按 Enter 键退出...")
    except EOFError:
        pass


def main() -> int:
    root = _app_root()
    os.chdir(root)

    python_exe = root / "runtime" / "Scripts" / "python.exe"
    launcher = root / "scripts" / "client_launcher.py"

    if not python_exe.is_file():
        _show_error(
            "Nanxing RAG 启动失败",
            "未找到运行环境（runtime\\Scripts\\python.exe）。\n\n"
            "请确认安装包已完整解压，勿只复制 exe 文件。\n"
            "若仍无法启动，请联系技术支持。",
        )
        return 1
    if not launcher.is_file():
        _show_error(
            "Nanxing RAG 启动失败",
            "未找到启动脚本（scripts\\client_launcher.py）。\n\n"
            "请确认安装包已完整解压，或联系技术支持。",
        )
        return 1

    env = os.environ.copy()
    env["RAG_CLIENT_MODE"] = "1"
    env["RAG_CLIENT_APP_ROOT"] = str(root)
    env["HF_EMBED_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TIKTOKEN_CACHE_DIR"] = str(root / "config" / "tiktoken_cache")

    try:
        result = subprocess.run(
            [str(python_exe), str(launcher)],
            cwd=str(root),
            env=env,
        )
        return int(result.returncode or 0)
    except OSError as exc:
        _show_error("Nanxing RAG 启动失败", f"无法启动服务：\n{exc}\n\n请联系技术支持。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
