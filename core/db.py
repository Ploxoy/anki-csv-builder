"""Lightweight Postgres logging helpers (optional).

If env DATABASE_URL is not set, functions are no-ops.
Designed for simple synchronous inserts of usage_events and minimal per-user
settings persistence for Phase 0.5.
"""
from __future__ import annotations

import json
import os
import logging
import hashlib
import secrets
import time
import threading
import uuid
from datetime import datetime
from typing import Iterable, Mapping, Any

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore

logger = logging.getLogger(__name__)

_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


def _db_url() -> str:
    """Resolve DB URL from primary and common integration env names."""
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRES_PRISMA_URL")
        or ""
    ).strip()


def _db_connect_retries() -> int:
    try:
        return max(1, int(os.getenv("DB_CONNECT_RETRIES", "3")))
    except Exception:
        return 3


def _db_connect_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "8")))
    except Exception:
        return 8.0

USAGE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS usage_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT,
    run_id TEXT,
    kind TEXT,
    provider TEXT,
    model TEXT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    cached_tokens BIGINT,
    audio_chars BIGINT,
    audio_tokens BIGINT,
    seconds DOUBLE PRECISION,
    raw_cost_usd DOUBLE PRECISION,
    raw_cost_eur DOUBLE PRECISION,
    charged_cost_eur DOUBLE PRECISION,
    markup_tier TEXT,
    markup_multiplier DOUBLE PRECISION,
    request_id TEXT
);
"""

USER_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT PRIMARY KEY,
    settings_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    label TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    token_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ
);
"""

GENERATION_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    payload_json JSONB NOT NULL,
    state_json JSONB,
    result_json JSONB,
    error_text TEXT,
    total_items INT NOT NULL DEFAULT 0,
    processed_items INT NOT NULL DEFAULT 0,
    attempt_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ
);
"""

# Helpful index for lookups in /api/usage.
USAGE_EVENTS_USER_CREATED_IDX = """
CREATE INDEX IF NOT EXISTS usage_events_user_created_at_idx
ON usage_events (user_id, created_at DESC);
"""

GENERATION_JOBS_USER_CREATED_IDX = """
CREATE INDEX IF NOT EXISTS generation_jobs_user_created_at_idx
ON generation_jobs (user_id, created_at DESC);
"""

GENERATION_JOBS_STATUS_CREATED_IDX = """
CREATE INDEX IF NOT EXISTS generation_jobs_status_created_at_idx
ON generation_jobs (status, created_at ASC);
"""


def _get_conn():
    global _SCHEMA_READY
    db_url = _db_url()
    if not db_url:
        return None
    if psycopg is None:
        logger.warning("psycopg is not installed; skipping DB logging")
        return None
    retries = _db_connect_retries()
    timeout_seconds = _db_connect_timeout()
    last_error = None

    for attempt in range(1, retries + 1):
        conn = None
        try:
            conn = psycopg.connect(
                db_url,
                autocommit=True,
                connect_timeout=timeout_seconds,
            )
            if not _SCHEMA_READY:
                with _SCHEMA_LOCK:
                    if not _SCHEMA_READY:
                        with conn.cursor() as cur:
                            cur.execute(USAGE_EVENTS_TABLE)
                            cur.execute(USERS_TABLE)
                            cur.execute(USER_SETTINGS_TABLE)
                            cur.execute(GENERATION_JOBS_TABLE)
                            cur.execute(USAGE_EVENTS_USER_CREATED_IDX)
                            cur.execute(GENERATION_JOBS_USER_CREATED_IDX)
                            cur.execute(GENERATION_JOBS_STATUS_CREATED_IDX)
                        _SCHEMA_READY = True
            return conn
        except Exception as exc:  # pragma: no cover - runtime env
            last_error = exc
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if attempt < retries:
                backoff = min(2.0, 0.35 * attempt)
                time.sleep(backoff)

    logger.error("Failed to connect or init schema: %s", last_error)
    return None


def db_status() -> tuple[bool, str]:
    """Return (ok, reason) for the configured DB connection.

    Used by Phase 0.5 endpoints where persistence is required.
    """
    if not _db_url():
        return False, "DATABASE_URL is not set"
    if psycopg is None:
        return False, "psycopg is not installed (install psycopg[binary])"
    conn = _get_conn()
    if conn is None:
        return False, "Failed to connect to DB or init schema (check server logs)"
    try:
        conn.close()
    except Exception:
        pass
    return True, ""


def log_usage_events(user_id: str, run_id: str, events: Iterable[Mapping[str, Any]]) -> None:
    """Insert usage events into Postgres if DATABASE_URL is set.

    Silently no-ops on missing DB or failures.
    """
    conn = _get_conn()
    if conn is None:
        return
    rows = []
    for ev in events or []:
        try:
            rows.append(
                (
                    user_id or None,
                    run_id or None,
                    ev.get("kind"),
                    ev.get("provider"),
                    ev.get("model"),
                    ev.get("input_tokens"),
                    ev.get("output_tokens"),
                    ev.get("cached_tokens"),
                    ev.get("audio_chars"),
                    ev.get("audio_tokens"),
                    ev.get("seconds"),
                    ev.get("raw_cost_usd"),
                    ev.get("raw_cost_eur"),
                    ev.get("charged_cost_eur"),
                    ev.get("markup_tier"),
                    ev.get("markup_multiplier"),
                    ev.get("request_id"),
                )
            )
        except Exception:
            continue
    if not rows:
        return
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO usage_events (
                    user_id, run_id, kind, provider, model,
                    input_tokens, output_tokens, cached_tokens,
                    audio_chars, audio_tokens, seconds,
                    raw_cost_usd, raw_cost_eur, charged_cost_eur,
                    markup_tier, markup_multiplier, request_id
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to insert usage events: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_user_settings(user_id: str) -> tuple[dict[str, Any], str] | None:
    """Return (settings_json, updated_at_iso) for user, or None if missing/unavailable."""
    if not user_id:
        return None
    conn = _get_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT settings_json, updated_at FROM user_settings WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            settings_raw, updated_at = row[0], row[1]
            if isinstance(settings_raw, str):
                try:
                    settings: dict[str, Any] = json.loads(settings_raw) if settings_raw else {}
                except Exception:
                    settings = {}
            elif isinstance(settings_raw, dict):
                settings = settings_raw
            else:
                settings = {}
            updated_at_iso = updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at)
            return settings, updated_at_iso
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to read user settings: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def upsert_user_settings(user_id: str, settings: Mapping[str, Any]) -> None:
    """Insert or update settings for user (no-op if DB is unavailable)."""
    if not user_id:
        return
    conn = _get_conn()
    if conn is None:
        return
    try:
        payload = json.dumps(dict(settings or {}), ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_settings (user_id, settings_json, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (user_id) DO UPDATE
                SET settings_json=excluded.settings_json, updated_at=now()
                """,
                (user_id, payload),
            )
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to upsert user settings: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_usage_events(
    *,
    user_id: str,
    limit: int = 200,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """List usage_events rows for user (newest first)."""
    if not user_id:
        return []
    conn = _get_conn()
    if conn is None:
        return []
    limit = max(1, min(int(limit or 200), 1000))
    try:
        sql = """
            SELECT
                created_at, run_id, kind, provider, model,
                input_tokens, output_tokens, cached_tokens,
                audio_chars, audio_tokens, seconds,
                raw_cost_usd, raw_cost_eur, charged_cost_eur,
                markup_tier, markup_multiplier, request_id
            FROM usage_events
            WHERE user_id=%s
        """
        params: list[Any] = [user_id]
        if run_id:
            sql += " AND run_id=%s"
            params.append(run_id)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            created_at = row[0]
            out.append(
                {
                    "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at),
                    "run_id": row[1],
                    "kind": row[2],
                    "provider": row[3],
                    "model": row[4],
                    "input_tokens": row[5],
                    "output_tokens": row[6],
                    "cached_tokens": row[7],
                    "audio_chars": row[8],
                    "audio_tokens": row[9],
                    "seconds": row[10],
                    "raw_cost_usd": row[11],
                    "raw_cost_eur": row[12],
                    "charged_cost_eur": row[13],
                    "markup_tier": row[14],
                    "markup_multiplier": row[15],
                    "request_id": row[16],
                }
            )
        return out
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to list usage events: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_user_invite(*, label: str | None = None) -> tuple[str, str] | None:
    """Create a new user + return (user_id, token) or None if DB unavailable."""
    conn = _get_conn()
    if conn is None:
        return None
    try:
        for _ in range(5):
            user_id = str(uuid.uuid4())
            token = secrets.token_urlsafe(32)
            token_hash = _hash_token(token)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (id, label, status, token_hash) VALUES (%s, %s, 'active', %s)",
                        (user_id, label, token_hash),
                    )
                return user_id, token
            except Exception:
                continue
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def resolve_user_id_from_token(token: str) -> str | None:
    """Return user_id for a user token, or None if invalid."""
    if not token:
        return None
    conn = _get_conn()
    if conn is None:
        return None
    try:
        token_hash = _hash_token(token)
        with conn.cursor() as cur:
            cur.execute("SELECT id, status FROM users WHERE token_hash=%s", (token_hash,))
            row = cur.fetchone()
            if not row:
                return None
            user_id, status = row[0], row[1]
            if status != "active":
                return None
            try:
                cur.execute("UPDATE users SET last_used_at=now() WHERE id=%s", (user_id,))
            except Exception:
                pass
            return str(user_id)
    except Exception:  # pragma: no cover - runtime env
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_users(limit: int = 200) -> list[dict[str, Any]]:
    """Admin: list users with metadata (no tokens)."""
    conn = _get_conn()
    if conn is None:
        return []
    limit = max(1, min(int(limit or 200), 1000))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, label, status, created_at, last_used_at
                FROM users
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row[0],
                    "label": row[1],
                    "status": row[2],
                    "created_at": row[3].isoformat() if isinstance(row[3], datetime) else str(row[3]),
                    "last_used_at": row[4].isoformat() if isinstance(row[4], datetime) else (row[4] or None),
                }
            )
        return out
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to list users: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def set_user_status(user_id: str, status: str) -> bool:
    """Admin: set user status ('active' or 'blocked')."""
    if status not in {"active", "blocked"}:
        return False
    conn = _get_conn()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET status=%s WHERE id=%s", (status, user_id))
            return cur.rowcount > 0
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to update user status: %s", exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def rotate_user_token(user_id: str) -> str | None:
    """Admin: rotate token for a user, returning new token."""
    conn = _get_conn()
    if conn is None:
        return None
    try:
        for _ in range(3):
            token = secrets.token_urlsafe(32)
            token_hash = _hash_token(token)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET token_hash=%s, last_used_at=NULL WHERE id=%s",
                    (token_hash, user_id),
                )
                if cur.rowcount > 0:
                    return token
        return None
    except Exception as exc:  # pragma: no cover - runtime env
        logger.error("Failed to rotate token: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def create_generation_job(*, user_id: str, run_id: str, payload: Mapping[str, Any], total_items: int) -> str | None:
    conn = _get_conn()
    if conn is None:
        return None
    try:
        job_id = str(uuid.uuid4())
        payload_json = json.dumps(dict(payload or {}), ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO generation_jobs (
                    id, user_id, run_id, status, payload_json, total_items, processed_items, updated_at, heartbeat_at
                )
                VALUES (%s, %s, %s, 'queued', %s::jsonb, %s, 0, now(), now())
                """,
                (job_id, user_id, run_id, payload_json, int(total_items or 0)),
            )
        return job_id
    except Exception as exc:
        logger.error("Failed to create generation job: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_generation_job(*, job_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    if not job_id:
        return None
    conn = _get_conn()
    if conn is None:
        return None
    try:
        sql = """
            SELECT id, user_id, run_id, status, payload_json, state_json, result_json, error_text,
                   total_items, processed_items, attempt_count, created_at, updated_at, started_at, finished_at, heartbeat_at
            FROM generation_jobs
            WHERE id=%s
        """
        params: list[Any] = [job_id]
        if user_id:
            sql += " AND user_id=%s"
            params.append(user_id)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        if not row:
            return None
        def _json(val: Any) -> dict[str, Any]:
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    loaded = json.loads(val)
                    return loaded if isinstance(loaded, dict) else {}
                except Exception:
                    return {}
            return {}
        def _ts(val: Any) -> str | None:
            if isinstance(val, datetime):
                return val.isoformat()
            if val is None:
                return None
            return str(val)
        return {
            "id": row[0],
            "user_id": row[1],
            "run_id": row[2],
            "status": row[3],
            "payload_json": _json(row[4]),
            "state_json": _json(row[5]),
            "result_json": _json(row[6]),
            "error_text": row[7],
            "total_items": int(row[8] or 0),
            "processed_items": int(row[9] or 0),
            "attempt_count": int(row[10] or 0),
            "created_at": _ts(row[11]),
            "updated_at": _ts(row[12]),
            "started_at": _ts(row[13]),
            "finished_at": _ts(row[14]),
            "heartbeat_at": _ts(row[15]),
        }
    except Exception as exc:
        logger.error("Failed to get generation job: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def claim_generation_job(*, user_id: str | None = None, job_id: str | None = None, stale_seconds: int = 300) -> dict[str, Any] | None:
    conn = _get_conn()
    if conn is None:
        return None
    stale_seconds = max(30, int(stale_seconds or 300))
    try:
        filters = ["j.status='queued' OR (j.status='running' AND j.heartbeat_at < now() - (%s || ' seconds')::interval)"]
        params: list[Any] = [stale_seconds]
        if user_id:
            filters.append("j.user_id=%s")
            params.append(user_id)
        if job_id:
            filters.append("j.id=%s")
            params.append(job_id)
        where_sql = " AND ".join(f"({f})" for f in filters)

        sql = f"""
            WITH candidate AS (
                SELECT j.id
                FROM generation_jobs j
                WHERE {where_sql}
                ORDER BY j.created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE generation_jobs j
            SET status='running',
                started_at=COALESCE(j.started_at, now()),
                updated_at=now(),
                heartbeat_at=now(),
                attempt_count=j.attempt_count + 1
            FROM candidate
            WHERE j.id=candidate.id
            RETURNING j.id
        """
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        if not row:
            return None
        return get_generation_job(job_id=str(row[0]))
    except Exception as exc:
        logger.error("Failed to claim generation job: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def update_generation_job_progress(
    *,
    job_id: str,
    processed_items: int,
    state: Mapping[str, Any],
) -> bool:
    conn = _get_conn()
    if conn is None:
        return False
    try:
        state_json = json.dumps(dict(state or {}), ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE generation_jobs
                SET processed_items=%s,
                    state_json=%s::jsonb,
                    status='running',
                    updated_at=now(),
                    heartbeat_at=now()
                WHERE id=%s
                """,
                (int(processed_items or 0), state_json, job_id),
            )
            return cur.rowcount > 0
    except Exception as exc:
        logger.error("Failed to update generation job progress: %s", exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def complete_generation_job(*, job_id: str, result: Mapping[str, Any], processed_items: int) -> bool:
    conn = _get_conn()
    if conn is None:
        return False
    try:
        result_json = json.dumps(dict(result or {}), ensure_ascii=False)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE generation_jobs
                SET status='done',
                    result_json=%s::jsonb,
                    processed_items=%s,
                    updated_at=now(),
                    finished_at=now(),
                    heartbeat_at=now(),
                    error_text=NULL
                WHERE id=%s
                """,
                (result_json, int(processed_items or 0), job_id),
            )
            return cur.rowcount > 0
    except Exception as exc:
        logger.error("Failed to complete generation job: %s", exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fail_generation_job(*, job_id: str, error_text: str, state: Mapping[str, Any] | None = None, processed_items: int | None = None) -> bool:
    conn = _get_conn()
    if conn is None:
        return False
    try:
        state_payload = json.dumps(dict(state or {}), ensure_ascii=False) if state is not None else None
        with conn.cursor() as cur:
            if state_payload is not None and processed_items is not None:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET status='failed',
                        error_text=%s,
                        state_json=%s::jsonb,
                        processed_items=%s,
                        updated_at=now(),
                        finished_at=now(),
                        heartbeat_at=now()
                    WHERE id=%s
                    """,
                    (error_text, state_payload, int(processed_items), job_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE generation_jobs
                    SET status='failed',
                        error_text=%s,
                        updated_at=now(),
                        finished_at=now(),
                        heartbeat_at=now()
                    WHERE id=%s
                    """,
                    (error_text, job_id),
                )
            return cur.rowcount > 0
    except Exception as exc:
        logger.error("Failed to fail generation job: %s", exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
