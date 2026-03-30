from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response


LOG_LEVEL = os.getenv("WAKER_LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [waker] %(message)s",
)
logger = logging.getLogger("waker")

DOCKER_API_BASE = os.getenv("WAKER_DOCKER_API_BASE", "http://socket-proxy:2375").rstrip("/")
TARGET_BASE = os.getenv("WAKER_TARGET_BASE", "http://anki_web:80").rstrip("/")
SERVICE_ORDER = [name.strip() for name in os.getenv("WAKER_SERVICE_ORDER", "anki_db,anki_api,anki_web").split(",") if name.strip()]
IDLE_MINUTES = max(1, int(os.getenv("WAKER_IDLE_MINUTES", "60")))
IDLE_CHECK_SECONDS = max(10, int(os.getenv("WAKER_IDLE_CHECK_SECONDS", "30")))
START_TIMEOUT_SECONDS = max(10, int(os.getenv("WAKER_START_TIMEOUT_SECONDS", "120")))
TOUCH_FLUSH_SECONDS = max(5, int(os.getenv("WAKER_TOUCH_FLUSH_SECONDS", "15")))
PROXY_TIMEOUT_SECONDS = max(5, int(os.getenv("WAKER_PROXY_TIMEOUT_SECONDS", "600")))
IDLE_STOP_ENABLED = os.getenv("WAKER_IDLE_STOP", "1").strip().lower() in {"1", "true", "yes", "on"}
STATE_FILE = Path(os.getenv("WAKER_STATE_FILE", "/state/last_access.txt"))

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

app = FastAPI(title="synology-waker", version="0.1.0")
_state_lock = asyncio.Lock()
_service_lock = asyncio.Lock()
_last_access_ts = time.time()
_last_flush_ts = 0.0


def _sanitize_headers(headers: httpx.Headers) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}


def _read_last_access() -> float:
    if not STATE_FILE.exists():
        return time.time()
    try:
        text = STATE_FILE.read_text(encoding="utf-8").strip()
        parsed = float(text)
        if parsed <= 0:
            return time.time()
        return parsed
    except Exception:
        return time.time()


def _write_last_access(value: float) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(f"{value:.3f}", encoding="utf-8")


async def _docker_request(method: str, path: str, timeout: float = 10.0) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.request(method, f"{DOCKER_API_BASE}{path}")


async def _container_state(name: str) -> str:
    response = await _docker_request("GET", f"/containers/{name}/json")
    if response.status_code == 404:
        return "missing"
    if response.status_code >= 400:
        raise RuntimeError(f"Failed reading container {name}: HTTP {response.status_code}")
    payload = response.json()
    state = (payload.get("State") or {}).get("Status")
    return state or "unknown"


async def _start_container(name: str) -> None:
    response = await _docker_request("POST", f"/containers/{name}/start")
    if response.status_code not in {204, 304}:
        raise RuntimeError(f"Failed starting {name}: HTTP {response.status_code}")


async def _stop_container(name: str) -> None:
    response = await _docker_request("POST", f"/containers/{name}/stop?t=10")
    if response.status_code not in {204, 304}:
        raise RuntimeError(f"Failed stopping {name}: HTTP {response.status_code}")


async def _target_is_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            response = await client.get(f"{TARGET_BASE}/")
        return response.status_code < 500
    except Exception:
        return False


async def _ensure_awake() -> None:
    async with _service_lock:
        for name in SERVICE_ORDER:
            state = await _container_state(name)
            if state == "missing":
                raise RuntimeError(f"Container not found: {name}")
            if state != "running":
                logger.info("Starting %s (state=%s)", name, state)
                await _start_container(name)

        deadline = time.time() + START_TIMEOUT_SECONDS
        while time.time() < deadline:
            if await _target_is_ready():
                return
            await asyncio.sleep(1.0)

        raise RuntimeError("Timed out waiting for web target readiness")


async def _stop_for_idle(deep: bool = True) -> List[str]:
    async with _service_lock:
        targets = list(reversed(SERVICE_ORDER)) if deep else [name for name in reversed(SERVICE_ORDER) if name != SERVICE_ORDER[0]]
        stopped: List[str] = []
        for name in targets:
            state = await _container_state(name)
            if state == "running":
                logger.info("Stopping %s (idle sleep)", name)
                await _stop_container(name)
                stopped.append(name)
        return stopped


async def _touch_last_access(force: bool = False) -> None:
    global _last_access_ts, _last_flush_ts
    now = time.time()
    async with _state_lock:
        _last_access_ts = now
        if force or (now - _last_flush_ts) >= TOUCH_FLUSH_SECONDS:
            _write_last_access(now)
            _last_flush_ts = now


async def _idle_loop() -> None:
    global _last_flush_ts
    while True:
        await asyncio.sleep(IDLE_CHECK_SECONDS)
        if not IDLE_STOP_ENABLED:
            continue

        async with _state_lock:
            idle_for = time.time() - _last_access_ts

        if idle_for < IDLE_MINUTES * 60:
            continue

        try:
            stopped = await _stop_for_idle(deep=True)
            if stopped:
                logger.warning("Idle sleep activated after %.0fs, stopped: %s", idle_for, ", ".join(stopped))
                async with _state_lock:
                    now = time.time()
                    _last_access_ts = now
                    _write_last_access(now)
                    _last_flush_ts = now
        except Exception as exc:
            logger.error("Idle loop failed: %s", exc)


@app.on_event("startup")
async def _startup() -> None:
    global _last_access_ts, _last_flush_ts
    _last_access_ts = _read_last_access()
    _last_flush_ts = _last_access_ts
    _write_last_access(_last_access_ts)
    asyncio.create_task(_idle_loop())
    logger.info(
        "Waker started: target=%s services=%s idle=%s idle_minutes=%s",
        TARGET_BASE,
        ",".join(SERVICE_ORDER),
        IDLE_STOP_ENABLED,
        IDLE_MINUTES,
    )


@app.middleware("http")
async def _wake_and_proxy(request: Request, call_next):
    path = request.url.path
    if path.startswith("/_waker"):
        return await call_next(request)

    await _touch_last_access()
    try:
        await _ensure_awake()
    except Exception as exc:
        return PlainTextResponse(
            f"Service is waking up. Retry in 10-30 seconds.\nReason: {exc}",
            status_code=503,
        )

    body = await request.body()
    target_url = f"{TARGET_BASE}{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=PROXY_TIMEOUT_SECONDS, follow_redirects=False) as client:
        upstream = await client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=headers,
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_sanitize_headers(upstream.headers),
    )


@app.get("/_waker/health")
async def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/_waker/status")
async def status() -> JSONResponse:
    states = {}
    for name in SERVICE_ORDER:
        try:
            states[name] = await _container_state(name)
        except Exception as exc:
            states[name] = f"error: {exc}"
    return JSONResponse(
        {
            "target": TARGET_BASE,
            "service_order": SERVICE_ORDER,
            "idle_stop_enabled": IDLE_STOP_ENABLED,
            "idle_minutes": IDLE_MINUTES,
            "states": states,
            "last_access_epoch": _last_access_ts,
            "now_epoch": time.time(),
        }
    )


@app.post("/_waker/wake")
async def wake() -> JSONResponse:
    await _touch_last_access(force=True)
    await _ensure_awake()
    return JSONResponse({"ok": True, "message": "Services are running"})


@app.post("/_waker/sleep")
async def sleep(deep: bool = True) -> JSONResponse:
    stopped = await _stop_for_idle(deep=deep)
    return JSONResponse({"ok": True, "deep": deep, "stopped": stopped})
