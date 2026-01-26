#!/usr/bin/env python3
"""Environment sanity checks for local/dev deployments.

Goal: quickly answer “why doesn't Streamlit/FastAPI start?” without guessing.
This script does NOT make external network calls.
"""

from __future__ import annotations

import importlib
import os
import py_compile
import sys
from pathlib import Path


def _check_import(module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        return False, f"{type(exc).__name__}: {exc}"


def _check_socket_permission() -> tuple[bool, str]:
    try:
        import socket

        _ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return True, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"


def _check_compile(path: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(path), doraise=True)
        return True, ""
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    print("== Doedutch / Anki CSV Builder — doctor ==")
    print(f"python: {sys.version.split()[0]}")
    print(f"cwd: {Path.cwd()}")
    print(f"repo_root: {root}")
    print()

    env_keys = ["API_SHARED_SECRET", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "DATABASE_URL", "PORT"]
    print("env:")
    for key in env_keys:
        val = os.getenv(key)
        if not val:
            print(f"  {key}: (not set)")
            continue
        if key.endswith("_KEY"):
            print(f"  {key}: set (len={len(val)})")
        else:
            print(f"  {key}: {val}")
    print()

    ok_socket, socket_err = _check_socket_permission()
    print("network sockets:")
    if ok_socket:
        print("  ok")
    else:
        print(f"  FAILED ({socket_err})")
    print()

    modules = [
        "fastapi",
        "uvicorn",
        "openai",
        "pydantic",
        "psycopg",
        "streamlit",
        "pandas",
    ]
    print("imports:")
    imports_ok = True
    for mod in modules:
        ok, err = _check_import(mod)
        imports_ok = imports_ok and ok
        print(f"  {mod}: {'ok' if ok else 'FAILED'}{'' if ok else f' ({err})'}")
    print()

    print("project imports:")
    ok_api, api_err = _check_import("api.main")
    print(f"  api.main: {'ok' if ok_api else f'FAILED ({api_err})'}")
    ok_core, core_err = _check_import("core.generation")
    print(f"  core.generation: {'ok' if ok_core else f'FAILED ({core_err})'}")
    print()

    # Avoid importing Streamlit UI entrypoints for side effects beyond app.app check.
    streamlit_entry = root / "app" / "app.py"
    api_entry = root / "api" / "main.py"
    ok_streamlit_compile, streamlit_compile_err = _check_compile(streamlit_entry) if streamlit_entry.exists() else (False, "missing")
    print("compile:")
    print(
        f"  app/app.py: {'ok' if ok_streamlit_compile else f'FAILED ({streamlit_compile_err})'}"
    )
    print()
    print("entrypoints:")
    print(f"  streamlit: {'ok' if streamlit_entry.exists() else 'MISSING'} ({streamlit_entry})")
    print(f"  fastapi:   {'ok' if api_entry.exists() else 'MISSING'} ({api_entry})")
    print()

    ok = imports_ok and ok_api and ok_core and ok_streamlit_compile and ok_socket
    if not ok:
        print("Result: FAILED (see checks above)")
        return 1
    print("Result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
