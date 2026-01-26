"""FastAPI entrypoint exposing generation and TTS endpoints.

This wiring reuses existing core logic (generation, TTS) and the run_report
builder so that responses and billing usage match the Streamlit UI behavior.
"""
from __future__ import annotations

import base64
import hmac
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Header, Request

from core.secrets import read_secret
from core.api_schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerateItemResult,
    UsageEvent,
    UserSettings,
    UserSettingsUpsertRequest,
    UserSettingsResponse,
    UsageEventRecord,
    UsageListResponse,
    UsageSummary,
    InviteCreateRequest,
    InviteCreateResponse,
    WhoAmIResponse,
    UserListResponse,
    UserRecord,
    UserStatusRequest,
    UserRotateResponse,
    TTSRequest,
    TTSResponse,
    TTSAudio,
    TTSSummary,
)
from core.generation import GenerationSettings, generate_card
from core.llm_clients import create_client
from config.signalword_groups import SIGNALWORD_GROUPS
from config.settings import (
    SIGNALWORDS_B1,
    SIGNALWORDS_B2_PLUS,
    AUDIO_TTS_MODEL,
    AUDIO_TTS_FALLBACK,
    AUDIO_TTS_INSTRUCTIONS,
    AUDIO_SENTENCE_INSTRUCTION_DEFAULT,
    AUDIO_WORD_INSTRUCTION_DEFAULT,
    AUDIO_VOICES,
    AUDIO_ELEVEN_STYLES,
    AUDIO_ELEVEN_VOICES,
    L1_LANGS,
)
from core.audio import ensure_audio_for_cards
from core.run_report import build_run_report, resolve_audio_pricing
from openai import AuthenticationError
from core.db import (
    create_user_invite,
    db_status,
    get_user_settings,
    list_usage_events,
    log_usage_events,
    list_users,
    rotate_user_token,
    resolve_user_id_from_token,
    set_user_status,
    upsert_user_settings,
)


app = FastAPI(title="Doedutch API", version="0.1.0")
logger = logging.getLogger(__name__)

def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default


@app.on_event("startup")
def _warn_if_auth_disabled() -> None:
    if not _env_flag("API_REQUIRE_SHARED_SECRET", default=True):
        logger.warning(
            "API_REQUIRE_SHARED_SECRET=0: shared-secret auth is disabled. "
            "This is intended for local development only; do NOT use in production."
        )


def _require_api_shared_secret(x_api_key: str | None) -> None:
    if not _env_flag("API_REQUIRE_SHARED_SECRET", default=True):
        return
    try:
        shared_secret = read_secret("API_SHARED_SECRET", required=True)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="API_SHARED_SECRET is not configured on the server")
    if not x_api_key or not hmac.compare_digest(x_api_key, shared_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")

def _extract_bearer_token(authorization_header: str | None) -> str:
    if not authorization_header:
        return ""
    parts = authorization_header.strip().split(None, 1)
    if len(parts) != 2:
        return ""
    scheme, token = parts[0].lower(), parts[1].strip()
    if scheme != "bearer":
        return ""
    return token


def _require_user(request: Request, x_api_key: str | None) -> str:
    """Resolve authenticated user_id for beta.

    Preferred: Authorization: Bearer <token> (or X-User-Token).
    Legacy fallback (disabled by default): X-API-Key + X-User-Id.
    """
    token = (request.headers.get("X-User-Token") or "").strip()
    if not token:
        token = _extract_bearer_token(request.headers.get("Authorization"))
    if token:
        ok_db, reason = db_status()
        if not ok_db:
            raise HTTPException(status_code=503, detail=reason)
        user_id = resolve_user_id_from_token(token)
        if not user_id:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user_id

    if _env_flag("API_ALLOW_LEGACY_USER_ID", default=False):
        _require_api_shared_secret(x_api_key)
        legacy_user_id = (request.headers.get("X-User-Id") or "").strip()
        if not legacy_user_id:
            raise HTTPException(status_code=400, detail="X-User-Id header is required")
        return legacy_user_id

    raise HTTPException(status_code=401, detail="Unauthorized")


def _make_usage_response(user_id: str, rows: List[Dict[str, Any]]) -> UsageListResponse:
    items = [UsageEventRecord(**row) for row in rows]

    summary = UsageSummary(events=len(items))
    cost_total = 0.0
    cost_any = False
    for it in items:
        summary.input_tokens += int(it.input_tokens or 0)
        summary.output_tokens += int(it.output_tokens or 0)
        summary.cached_tokens += int(it.cached_tokens or 0)
        summary.audio_chars += int(it.audio_chars or 0)
        if it.raw_cost_usd is not None:
            cost_total += float(it.raw_cost_usd or 0.0)
            cost_any = True
    if cost_any:
        summary.raw_cost_usd = round(cost_total, 6)

    return UsageListResponse(user_id=user_id, items=items, summary=summary)


def _openai_client_or_500() -> Any:
    try:
        api_key = read_secret("OPENAI_API_KEY", required=True)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server")
    client = create_client(api_key)
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK is not available")
    return client


def _elevenlabs_key_or_500() -> str:
    try:
        api_key = read_secret("ELEVENLABS_API_KEY", required=True)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not configured on the server")
    return api_key


def _status_from_card(card: Dict[str, Any]) -> str:
    err = (card.get("error") or "").strip()
    meta = card.get("meta") or {}
    if err == "flagged_precheck":
        return "flagged"
    if err:
        return "failed"
    if meta.get("repair_attempted"):
        return "repaired"
    return "ok"


def _usage_from_meta(meta: Dict[str, Any]) -> UsageEvent:
    req = meta.get("request") or {}
    return UsageEvent(
        provider=meta.get("provider", "unknown") or "unknown",
        model=meta.get("model", "unknown") or "unknown",
        input_tokens=int(req.get("prompt_tokens", 0) or 0),
        output_tokens=int(req.get("completion_tokens", 0) or 0),
        cached_tokens=int(req.get("cached_tokens", 0) or 0),
        audio_chars=None,
        audio_tokens=None,
        seconds=None,
        raw_cost_usd=None,
        raw_cost_eur=None,
        charged_cost_eur=None,
        markup_tier=None,
        markup_multiplier=None,
        request_id=None,
        elapsed_ms=None,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/admin/invite", response_model=InviteCreateResponse)
def api_admin_invite(
    payload: InviteCreateRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> InviteCreateResponse:
    _require_api_shared_secret(x_api_key)
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    created = create_user_invite(label=(payload.label or None))
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create invite")
    user_id, token = created
    return InviteCreateResponse(user_id=user_id, token=token)


@app.get("/api/admin/users", response_model=UserListResponse)
def api_admin_users(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    limit: int = 200,
) -> UserListResponse:
    _require_api_shared_secret(x_api_key)
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    rows = list_users(limit=limit)
    return UserListResponse(items=[UserRecord(**row) for row in rows])


@app.post("/api/admin/users/{user_id}/status", response_model=UserRecord)
def api_admin_set_user_status(
    user_id: str,
    payload: UserStatusRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> UserRecord:
    _require_api_shared_secret(x_api_key)
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    if not set_user_status(user_id, payload.status):
        raise HTTPException(status_code=404, detail="User not found")
    rows = list_users(limit=500)
    row = next((r for r in rows if r["id"] == user_id), None)
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return UserRecord(**row)


@app.post("/api/admin/users/{user_id}/rotate", response_model=UserRotateResponse)
def api_admin_rotate_user(
    user_id: str,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> UserRotateResponse:
    _require_api_shared_secret(x_api_key)
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    token = rotate_user_token(user_id)
    if not token:
        raise HTTPException(status_code=404, detail="User not found or rotation failed")
    return UserRotateResponse(user_id=user_id, token=token)


@app.get("/api/admin/usage", response_model=UsageListResponse)
def api_admin_usage(
    user_id: str,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    limit: int = 200,
    run_id: str | None = None,
) -> UsageListResponse:
    _require_api_shared_secret(x_api_key)
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    rows = list_usage_events(user_id=user_id, limit=limit, run_id=run_id)
    return _make_usage_response(user_id=user_id, rows=rows)


@app.get("/api/whoami", response_model=WhoAmIResponse)
def api_whoami(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> WhoAmIResponse:
    user_id = _require_user(request, x_api_key)
    return WhoAmIResponse(user_id=user_id)


@app.get("/api/settings", response_model=UserSettingsResponse)
def api_get_settings(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> UserSettingsResponse:
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    user_id = _require_user(request, x_api_key)

    row = get_user_settings(user_id)
    if not row:
        return UserSettingsResponse(user_id=user_id, settings=UserSettings(), updated_at=None)

    settings_json, updated_at = row
    try:
        settings = UserSettings(**(settings_json or {}))
    except Exception:
        settings = UserSettings()
    return UserSettingsResponse(user_id=user_id, settings=settings, updated_at=updated_at)


@app.put("/api/settings", response_model=UserSettingsResponse)
def api_put_settings(
    payload: UserSettingsUpsertRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> UserSettingsResponse:
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    user_id = _require_user(request, x_api_key)

    settings_data = payload.settings.model_dump() if hasattr(payload.settings, "model_dump") else payload.settings.dict()
    upsert_user_settings(user_id, settings_data)
    row = get_user_settings(user_id)
    if not row:
        return UserSettingsResponse(user_id=user_id, settings=payload.settings, updated_at=None)
    settings_json, updated_at = row
    try:
        settings = UserSettings(**(settings_json or {}))
    except Exception:
        settings = payload.settings
    return UserSettingsResponse(user_id=user_id, settings=settings, updated_at=updated_at)


@app.get("/api/usage", response_model=UsageListResponse)
def api_usage(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    limit: int = 200,
    run_id: str | None = None,
) -> UsageListResponse:
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    user_id = _require_user(request, x_api_key)

    rows = list_usage_events(user_id=user_id, limit=limit, run_id=run_id)
    return _make_usage_response(user_id=user_id, rows=rows)


@app.post("/api/generate", response_model=GenerateResponse)
def api_generate(
    payload: GenerateRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> GenerateResponse:
    user_id = _require_user(request, x_api_key)
    client = _openai_client_or_500()

    l1_code = (payload.l1 or "").strip().upper()
    l1_meta = L1_LANGS.get(l1_code)
    if not l1_meta:
        supported = ", ".join(sorted(L1_LANGS.keys()))
        raise HTTPException(status_code=400, detail=f"Unsupported l1={payload.l1!r}. Supported: {supported}")

    signal_usage: Dict[str, int] = {}
    signal_last = None

    results_cards: List[Dict[str, Any]] = []
    items_out: List[GenerateItemResult] = []

    started = time.time()
    for idx, item in enumerate(payload.items):
        row = {
            "woord": item.woord,
            "def_nl": item.def_nl or "",
            "translation": item.translation or "",
        }
        settings = GenerationSettings(
            model=payload.model,
            provider=payload.provider or "openai",
            L1_code=l1_code,
            L1_name=l1_meta.get("name", l1_code),
            level=payload.cefr,
            profile=payload.profile,
            temperature=payload.temperature,
            max_output_tokens=payload.max_output_tokens,
            signalword_seed=idx,
        )

        try:
            result = generate_card(
                client=client,
                row=row,
                settings=settings,
                signalword_groups=SIGNALWORD_GROUPS,
                signalwords_b1=SIGNALWORDS_B1,
                signalwords_b2_plus=SIGNALWORDS_B2_PLUS,
                signal_usage=signal_usage,
                signal_last=signal_last,
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        signal_usage = result.signal_usage or signal_usage
        signal_last = result.signal_last or signal_last
        card = result.card
        results_cards.append(card)

        meta = card.get("meta") or {}
        status = _status_from_card(card)
        items_out.append(
            GenerateItemResult(
                id=item.id,
                status=status,  # type: ignore[arg-type]
                card=card if status in {"ok", "repaired"} else None,
                error=card.get("error") or None,
                usage=_usage_from_meta(meta),
            )
        )

    elapsed = time.time() - started
    run_stats = {
        "elapsed": elapsed,
        "batches": 1,
        "items": len(results_cards),
        "start_ts": started,
        "transient": 0,
    }

    # Build a fake state compatible with run_report
    state: Dict[str, Any] = {
        "results": results_cards,
        "sig_usage": signal_usage,
        "sig_last": signal_last,
        "audio_summary": {},
        "run_stats": run_stats,
    }
    run_report = build_run_report(SimpleNamespace(**state))

    # Persist usage events if DB configured
    try:
        log_usage_events(user_id=user_id, run_id=payload.run_id or "", events=run_report.get("usage_events", []))
    except Exception:
        pass

    return GenerateResponse(
        run_id=payload.run_id or "",
        prompt_version=payload.prompt_version,
        provider=payload.provider,
        model=payload.model,
        items=items_out,
        run_report=run_report,
        timing={"elapsed_ms": int(elapsed * 1000)},
    )


def _extract_sound_filename(sound_field: str) -> str:
    if not sound_field:
        return ""
    if sound_field.startswith("[sound:") and sound_field.endswith("]"):
        return sound_field[len("[sound:") : -1]
    return sound_field


@app.post("/api/tts", response_model=TTSResponse)
def api_tts(
    payload: TTSRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> TTSResponse:
    user_id = _require_user(request, x_api_key)
    provider = (payload.provider or "openai").strip().lower()

    openai_client = None
    eleven_api_key = None
    if provider == "openai":
        openai_client = _openai_client_or_500()
    elif provider == "elevenlabs":
        eleven_api_key = _elevenlabs_key_or_500()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported TTS provider: {payload.provider}")

    include_word = any(it.type == "word" for it in payload.items)
    include_sentence = any(it.type == "sentence" for it in payload.items)

    cards: List[Dict[str, Any]] = []
    for item in payload.items:
        card = {
            "id": item.card_id,
            "L2_word": item.text if item.type == "word" else "",
            "L2_cloze": item.text if item.type == "sentence" else "",
            "AudioWord": "",
            "AudioSentence": "",
        }
        cards.append(card)

    if provider == "elevenlabs":
        word_key = "Eleven_word_dictionary"
        sentence_key = "Eleven_sentence_tutor"
        word_payload = (AUDIO_ELEVEN_STYLES.get("word", {}).get(word_key) or {}).get("payload")
        sentence_payload = (AUDIO_ELEVEN_STYLES.get("sentence", {}).get(sentence_key) or {}).get("payload")
        instruction_payloads = {"word": word_payload, "sentence": sentence_payload}
        instruction_keys = {"word": word_key, "sentence": sentence_key}
        default_voice = AUDIO_ELEVEN_VOICES[0]["id"] if AUDIO_ELEVEN_VOICES else ""
        voice = payload.voice or default_voice
        if not voice:
            raise HTTPException(status_code=400, detail="voice is required for ElevenLabs TTS")
    else:
        word_key = AUDIO_WORD_INSTRUCTION_DEFAULT
        sentence_key = AUDIO_SENTENCE_INSTRUCTION_DEFAULT
        instruction_payloads = {
            "word": AUDIO_TTS_INSTRUCTIONS.get(word_key, ""),
            "sentence": AUDIO_TTS_INSTRUCTIONS.get(sentence_key, ""),
        }
        instruction_keys = {"word": word_key, "sentence": sentence_key}
        voice = payload.voice or (AUDIO_VOICES[0]["id"] if AUDIO_VOICES else "alloy")

    media_map, summary = ensure_audio_for_cards(
        cards,
        provider=provider,
        voice=voice,
        include_word=include_word,
        include_sentence=include_sentence,
        instruction_payloads=instruction_payloads,
        instruction_keys=instruction_keys,
        openai_client=openai_client,
        openai_model=(payload.model or AUDIO_TTS_MODEL) if provider == "openai" else None,
        openai_fallback_model=AUDIO_TTS_FALLBACK if provider == "openai" else None,
        eleven_api_key=eleven_api_key,
        eleven_model=(payload.model or None) if provider == "elevenlabs" else None,
        max_workers=4,
    )

    # Build response audios
    audios: List[TTSAudio] = []
    for item, card in zip(payload.items, cards):
        if item.type == "word":
            sound_field = card.get("AudioWord", "")
        else:
            sound_field = card.get("AudioSentence", "")
        filename = _extract_sound_filename(sound_field)
        data = media_map.get(filename, b"")
        usage_entry = None
        # We approximate per-item audio chars by len(text)
        if data:
            usage_entry = UsageEvent(
                provider=summary.provider or (payload.provider or "openai"),
                model=payload.model or AUDIO_TTS_MODEL,
                audio_chars=len(item.text or ""),
                audio_tokens=None,
                seconds=None,
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                raw_cost_usd=None,
                raw_cost_eur=None,
                charged_cost_eur=None,
                markup_tier=None,
                markup_multiplier=None,
                request_id=None,
                elapsed_ms=None,
            )
        audios.append(
            TTSAudio(
                card_id=item.card_id,
                type=item.type,
                filename=filename,
                audio_b64=base64.b64encode(data).decode("ascii") if data else None,
                usage=usage_entry,
            )
        )

    # Compute cost estimate per model using the same pricing helper as run_report
    cost_by_model: Dict[str, Dict[str, Any]] = {}
    audio_cost_total = 0.0
    if summary.model_usage:
        for model_name, data in summary.model_usage.items():
            chars = int(data.get("chars", 0) or 0)
            pricing = resolve_audio_pricing(model_name)
            est = None
            if pricing and chars:
                est = (chars / 1_000_000.0) * pricing
                audio_cost_total += est
            cost_by_model[model_name] = {"estimated_usd": est, "characters": chars}

    summary_block = TTSSummary(
        ok=summary.word_success + summary.sentence_success,
        failed=len(summary.errors or []),
        cached=summary.cache_hits,
        usage={"audio_chars": summary.total_characters},
        cost={"estimated_usd": round(audio_cost_total, 6) if audio_cost_total else None, "by_model": cost_by_model},
    )

    # Persist usage events (audio) if DB configured
    try:
        audio_events = [
            {
                "kind": "audio",
                "provider": summary.provider or (payload.provider or "openai"),
                "model": payload.model or AUDIO_TTS_MODEL,
                "audio_chars": summary.total_characters,
                "raw_cost_usd": round(audio_cost_total, 6) if audio_cost_total else None,
                "input_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "audio_tokens": None,
                "seconds": None,
                "raw_cost_eur": None,
                "charged_cost_eur": None,
                "markup_tier": None,
                "markup_multiplier": None,
                "request_id": None,
                "elapsed_ms": None,
            }
        ]
        log_usage_events(user_id=user_id, run_id=payload.run_id or "", events=audio_events)
    except Exception:
        pass

    return TTSResponse(
        run_id=payload.run_id or "",
        provider=payload.provider,
        model=payload.model or AUDIO_TTS_MODEL,
        audios=audios,
        summary=summary_block,
    )
