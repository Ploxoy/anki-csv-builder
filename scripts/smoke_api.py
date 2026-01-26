#!/usr/bin/env python3
"""API smoke checks without binding TCP sockets.

This environment disallows creating sockets, so we can't run uvicorn or use
Starlette/FastAPI TestClient (it uses socketpair internally via anyio).

Instead we call the endpoint functions directly with Pydantic payloads and a
minimal Starlette Request object.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starlette.requests import Request  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import api.main as api  # noqa: E402
from core.api_schemas import GenerateRequest, TTSRequest, UserSettingsUpsertRequest, UserSettings  # noqa: E402


@contextmanager
def temp_env(overrides: Dict[str, Optional[str]]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_request(headers: Dict[str, str] | None = None) -> Request:
    hdrs = []
    for k, v in (headers or {}).items():
        hdrs.append((k.lower().encode("ascii"), v.encode("utf-8")))
    scope = {"type": "http", "method": "POST", "path": "/", "headers": hdrs}
    return Request(scope)  # type: ignore[arg-type]


def call_generate(x_api_key: Optional[str]) -> Tuple[int, str]:
    payload = GenerateRequest(
        run_id="smoke",
        prompt_version="p0",
        provider="openai",
        model="gpt-4.1-mini",
        cefr="B1",
        profile="default",
        l1="en",
        items=[],
    )
    req = make_request({"X-User-Id": "smoke"})
    try:
        out = api.api_generate(payload, req, x_api_key=x_api_key)
        return 200, f"items={len(out.items)}"
    except HTTPException as exc:
        return exc.status_code, str(exc.detail)


def call_tts(x_api_key: Optional[str], provider: str) -> Tuple[int, str]:
    payload = TTSRequest(
        run_id="smoke",
        provider=provider,
        model="eleven_multilingual_v2" if provider == "elevenlabs" else "gpt-4o-mini-tts-2025-12-15",
        items=[],
    )
    req = make_request({"X-User-Id": "smoke"})
    try:
        out = api.api_tts(payload, req, x_api_key=x_api_key)
        return 200, f"audios={len(out.audios)} ok={out.summary.ok}"
    except HTTPException as exc:
        return exc.status_code, str(exc.detail)

def call_get_settings(x_api_key: Optional[str]) -> Tuple[int, str]:
    req = make_request({"X-User-Id": "smoke"})
    try:
        out = api.api_get_settings(req, x_api_key=x_api_key)
        return 200, f"model={out.settings.model} cefr={out.settings.cefr}"
    except HTTPException as exc:
        return exc.status_code, str(exc.detail)


def call_put_settings(x_api_key: Optional[str]) -> Tuple[int, str]:
    payload = UserSettingsUpsertRequest(settings=UserSettings(model="gpt-4.1-mini", cefr="B1", profile="balanced", l1="EN"))
    req = make_request({"X-User-Id": "smoke"})
    try:
        out = api.api_put_settings(payload, req, x_api_key=x_api_key)
        return 200, f"model={out.settings.model} cefr={out.settings.cefr}"
    except HTTPException as exc:
        return exc.status_code, str(exc.detail)


def call_usage(x_api_key: Optional[str]) -> Tuple[int, str]:
    req = make_request({"X-User-Id": "smoke"})
    try:
        out = api.api_usage(req, x_api_key=x_api_key)
        return 200, f"events={out.summary.events}"
    except HTTPException as exc:
        return exc.status_code, str(exc.detail)


def main() -> None:
    print("/health:", api.health())

    # Missing API_SHARED_SECRET -> 500
    with temp_env(
        {
            "API_REQUIRE_SHARED_SECRET": "1",
            "API_ALLOW_LEGACY_USER_ID": "1",
            "API_SHARED_SECRET": None,
            "API_SHARED_SECRET_FILE": None,
            "DATABASE_URL": None,
        }
    ):
        code, detail = call_generate(x_api_key="anything")
        print("generate without API_SHARED_SECRET:", code, detail)

    # Missing X-API-Key -> 401
    with temp_env(
        {
            "API_REQUIRE_SHARED_SECRET": "1",
            "API_ALLOW_LEGACY_USER_ID": "1",
            "API_SHARED_SECRET": "dev-shared",
            "OPENAI_API_KEY": "sk-dummy",
            "DATABASE_URL": None,
        }
    ):
        code, detail = call_generate(x_api_key=None)
        print("generate without X-API-Key:", code, detail)

    # Missing OPENAI_API_KEY -> 500
    with temp_env(
        {
            "API_REQUIRE_SHARED_SECRET": "1",
            "API_ALLOW_LEGACY_USER_ID": "1",
            "API_SHARED_SECRET": "dev-shared",
            "OPENAI_API_KEY": None,
            "OPENAI_API_KEY_FILE": None,
            "DATABASE_URL": None,
        }
    ):
        code, detail = call_generate(x_api_key="dev-shared")
        print("generate without OPENAI_API_KEY:", code, detail)

    # Happy path (empty items => no external calls)
    with temp_env(
        {
            "API_REQUIRE_SHARED_SECRET": "1",
            "API_ALLOW_LEGACY_USER_ID": "1",
            "API_SHARED_SECRET": "dev-shared",
            "OPENAI_API_KEY": "sk-dummy",
            "DATABASE_URL": None,
        }
    ):
        code, detail = call_generate(x_api_key="dev-shared")
        print("generate ok (empty items):", code, detail)

        code, detail = call_tts(x_api_key="dev-shared", provider="openai")
        print("tts openai ok (empty items):", code, detail)

        code, detail = call_get_settings(x_api_key="dev-shared")
        print("settings get ok:", code, detail)

        code, detail = call_put_settings(x_api_key="dev-shared")
        print("settings put ok:", code, detail)

        code, detail = call_usage(x_api_key="dev-shared")
        print("usage ok:", code, detail)

    with temp_env(
        {
            "API_REQUIRE_SHARED_SECRET": "1",
            "API_ALLOW_LEGACY_USER_ID": "1",
            "API_SHARED_SECRET": "dev-shared",
            "ELEVENLABS_API_KEY": "dummy-eleven",
            "DATABASE_URL": None,
        }
    ):
        code, detail = call_tts(x_api_key="dev-shared", provider="elevenlabs")
        print("tts elevenlabs ok (empty items):", code, detail)

    # Dev mode: disable shared secret
    with temp_env(
        {
            "API_REQUIRE_SHARED_SECRET": "0",
            "API_ALLOW_LEGACY_USER_ID": "1",
            "API_SHARED_SECRET": None,
            "OPENAI_API_KEY": "sk-dummy",
            "DATABASE_URL": None,
        }
    ):
        code, detail = call_generate(x_api_key=None)
        print("generate ok (no shared secret required):", code, detail)

    # Secret-file support (simulates Docker secrets)
    with tempfile.TemporaryDirectory() as td:
        api_secret_path = os.path.join(td, "api_shared_secret")
        openai_path = os.path.join(td, "openai_api_key")
        eleven_path = os.path.join(td, "elevenlabs_api_key")
        Path(api_secret_path).write_text("file-shared", encoding="utf-8")
        Path(openai_path).write_text("sk-from-file", encoding="utf-8")
        Path(eleven_path).write_text("eleven-from-file", encoding="utf-8")

        with temp_env(
            {
                "API_REQUIRE_SHARED_SECRET": "1",
                "API_ALLOW_LEGACY_USER_ID": "1",
                "API_SHARED_SECRET": None,
                "OPENAI_API_KEY": None,
                "ELEVENLABS_API_KEY": None,
                "DATABASE_URL": None,
                "API_SHARED_SECRET_FILE": api_secret_path,
                "OPENAI_API_KEY_FILE": openai_path,
                "ELEVENLABS_API_KEY_FILE": eleven_path,
            }
        ):
            code, detail = call_generate(x_api_key="file-shared")
            print("generate ok (secret files, empty items):", code, detail)

            code, detail = call_tts(x_api_key="file-shared", provider="openai")
            print("tts openai ok (secret files, empty items):", code, detail)

            code, detail = call_tts(x_api_key="file-shared", provider="elevenlabs")
            print("tts elevenlabs ok (secret files, empty items):", code, detail)


if __name__ == "__main__":
    main()
