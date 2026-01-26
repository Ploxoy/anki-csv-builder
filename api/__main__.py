"""Module entrypoint for running the FastAPI service.

Example:
  python -m api
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")
    reload_enabled = os.getenv("RELOAD", "").strip().lower() in {"1", "true", "yes"}

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()

