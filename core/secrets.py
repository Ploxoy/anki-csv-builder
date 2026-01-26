"""Secret-loading helpers.

Supports three sources (in priority order):
1) Direct env var (e.g. OPENAI_API_KEY)
2) File pointer env var (e.g. OPENAI_API_KEY_FILE=/run/secrets/openai_api_key)
3) Docker secrets default paths (/run/secrets/<name> and common variants)

This enables storing keys at the Docker level (secrets) while keeping local
developer workflows compatible with plain env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def _normalize_secret(value: str) -> str:
    """Best-effort normalization for secrets from env/files.

    - trims whitespace
    - strips a UTF-8 BOM if present
    - strips a single pair of matching surrounding quotes (", ')
    """
    if value is None:
        return ""
    v = value.strip()
    if not v:
        return ""
    # Sometimes secrets end up with a BOM or are copy-pasted with invisible prefix chars.
    if v and v[0] == "\ufeff":
        v = v[1:]
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
        v = v[1:-1]
    return v.strip()


def _read_file(path: Path) -> str:
    try:
        return _normalize_secret(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""


def _candidate_secret_paths(name: str) -> Iterable[Path]:
    # Docker secrets are typically mounted at /run/secrets/<secret_name>.
    run = Path("/run/secrets")
    yield run / name
    yield run / name.lower()
    yield run / name.upper()

    # Common secret naming convention: lower snake case.
    snake = name.lower()
    yield run / snake


def read_secret(name: str, *, required: bool = False) -> str:
    """Return secret value for `name` or empty string if missing (unless required)."""

    raw = os.getenv(name)
    raw_norm = _normalize_secret(raw or "")
    if raw is not None and raw_norm:
        return raw_norm

    file_env = os.getenv(f"{name}_FILE")
    file_env_norm = _normalize_secret(file_env or "")
    if file_env_norm:
        value = _read_file(Path(file_env_norm))
        if value:
            return value

    for path in _candidate_secret_paths(name):
        value = _read_file(path)
        if value:
            return value

    if required:
        raise RuntimeError(f"{name} is not configured")
    return ""
