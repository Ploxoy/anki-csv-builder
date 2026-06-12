"""FastAPI entrypoint exposing generation and TTS endpoints.

This wiring reuses existing core logic (generation, TTS) and the run_report
builder so that responses and billing usage match the Streamlit UI behavior.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse

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
    ExportDeckRequest,
    ExportFileResponse,
    AudioAssetCheckRequest,
    AudioAssetCheckResponse,
    TTSRequest,
    TTSOptionsResponse,
    TTSProviderOptions,
    TTSOption,
    TTSResponse,
    TTSAudio,
    TTSSummary,
    TTSStorageInfo,
    GenerateJobCreateResponse,
    GenerateJobStatusResponse,
    GenerateJobWorkerRequest,
    GenerateJobWorkerResponse,
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
    DEFAULT_MODELS,
    CSV_DELIMITER,
    CSV_LINETERMINATOR,
    ANKI_MODEL_ID,
    ANKI_DECK_ID,
    ANKI_MODEL_NAME,
    ANKI_DECK_NAME,
    CLOZE_FRONT_TEMPLATE_PATH,
    CLOZE_BACK_TEMPLATE_PATH,
    CLOZE_CSS_PATH,
    BASIC_CARD1_FRONT_TEMPLATE_PATH,
    BASIC_CARD1_BACK_TEMPLATE_PATH,
    BASIC_CARD2_FRONT_TEMPLATE_PATH,
    BASIC_CARD2_BACK_TEMPLATE_PATH,
    TYPEIN_FRONT_TEMPLATE_PATH,
    TYPEIN_BACK_TEMPLATE_PATH,
    get_preferred_order,
    get_allowed_prefixes,
    get_block_substrings,
)
from core.audio import AudioClipResult, AudioSynthesisSummary, ensure_audio_for_cards, sentence_for_tts, tts_asset_identity
from core.export_csv import generate_csv
from core.export_anki import HAS_GENANKI, build_anki_package
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
    create_generation_job,
    get_generation_job,
    claim_generation_job,
    update_generation_job_progress,
    complete_generation_job,
    fail_generation_job,
    store_run_media_assets,
    load_run_media_assets,
    store_audio_assets,
    load_audio_assets,
    load_audio_assets_by_filenames,
    list_existing_audio_asset_filenames,
    touch_audio_assets,
    store_generated_card_asset,
    load_generated_card_asset,
    touch_generated_card_asset,
    load_generated_card_assets,
    touch_generated_card_assets,
)


app = FastAPI(title="Doedutch API", version="0.1.0")
logger = logging.getLogger(__name__)
VERCEL_FUNCTION_BODY_LIMIT_BYTES = 4_500_000
EXPORT_REQUEST_SOFT_LIMIT_BYTES = 4_200_000

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


def _env_text(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw).strip()) if raw is not None else default
    except Exception:
        value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _estimate_export_request_size_bytes(payload: ExportDeckRequest) -> int:
    payload_data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    try:
        return len(json.dumps(payload_data, ensure_ascii=False).encode("utf-8"))
    except Exception:
        media_total = 0
        for content_b64 in (payload.media_map or {}).values():
            if content_b64:
                media_total += len(content_b64.encode("utf-8"))
        cards_total = sum(
            len((card.L2_word or "").encode("utf-8"))
            + len((card.L2_cloze or "").encode("utf-8"))
            + len((card.L1_sentence or "").encode("utf-8"))
            + len((card.L2_collocations or "").encode("utf-8"))
            + len((card.L2_definition or "").encode("utf-8"))
            + len((card.L1_gloss or "").encode("utf-8"))
            for card in payload.cards
        )
        return media_total + cards_total + 2048


def _iter_bytesio(buffer: io.BytesIO, chunk_size: int = 64 * 1024):
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        yield chunk


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
        user_id = resolve_user_id_from_token(token)
        if user_id:
            return user_id
        # One fallback DB health probe helps distinguish invalid token vs DB outage
        # without doing two DB roundtrips on every authenticated request.
        ok_db, reason = db_status()
        if not ok_db:
            raise HTTPException(status_code=503, detail=reason)
        raise HTTPException(status_code=401, detail="Unauthorized")

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
    elapsed_raw = req.get("total_elapsed_ms", req.get("elapsed_ms"))
    try:
        elapsed_ms = int(elapsed_raw) if elapsed_raw is not None else None
    except Exception:
        elapsed_ms = None
    return UsageEvent(
        provider=meta.get("provider", "unknown") or "unknown",
        model=meta.get("model", "unknown") or "unknown",
        input_tokens=int(req.get("prompt_tokens", 0) or 0),
        output_tokens=int(req.get("completion_tokens", 0) or 0),
        cached_tokens=int(req.get("cached_tokens", 0) or 0),
        audio_chars=None,
        audio_tokens=None,
        seconds=(elapsed_ms / 1000.0) if elapsed_ms else None,
        raw_cost_usd=None,
        raw_cost_eur=None,
        charged_cost_eur=None,
        markup_tier=None,
        markup_multiplier=None,
        request_id=None,
        elapsed_ms=elapsed_ms,
    )


def _normalize_generation_input_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _generation_card_asset_identity(*, payload: GenerateRequest, item: Any, l1_code: str) -> Dict[str, Any]:
    input_json = {
        "woord": _normalize_generation_input_text(getattr(item, "woord", "")),
        "def_nl": _normalize_generation_input_text(getattr(item, "def_nl", "") or ""),
        "translation": _normalize_generation_input_text(getattr(item, "translation", "") or ""),
    }
    settings_json = {
        "provider": (payload.provider or "openai").strip().lower(),
        "model": (payload.model or "").strip(),
        "prompt_version": (payload.prompt_version or "").strip(),
        "cefr": (payload.cefr or "").strip(),
        "profile": (payload.profile or "").strip(),
        "l1": (l1_code or payload.l1 or "").strip().upper(),
        "temperature": payload.temperature,
        "max_output_tokens": payload.max_output_tokens,
        "force_schema": bool(getattr(payload.flags, "force_schema", True)),
        "allow_repair": bool(getattr(payload.flags, "allow_repair", True)),
    }
    input_hash = hashlib.sha256(json.dumps(input_json, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    key_material = {
        "kind": "generated-card-v1",
        "settings": settings_json,
        "input": input_json,
    }
    asset_key = hashlib.sha256(json.dumps(key_material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "asset_key": asset_key,
        "input_hash": input_hash,
        "input_json": input_json,
        **settings_json,
    }


def _preload_generated_card_assets(
    *,
    payload: GenerateRequest,
    items: List[Any],
    l1_code: str,
) -> tuple[Dict[str, Dict[str, Any]], str | None]:
    if not bool(getattr(payload.flags, "reuse_text_cache", False)) or not items:
        return {}, None
    keys = [
        _generation_card_asset_identity(payload=payload, item=item, l1_code=l1_code)["asset_key"]
        for item in items
    ]
    return load_generated_card_assets(asset_keys=keys)


def _needs_text_provider(
    *,
    payload: GenerateRequest,
    items: List[Any],
    l1_code: str,
    text_cache_assets: Dict[str, Dict[str, Any]],
) -> bool:
    if not items:
        return False
    if not bool(getattr(payload.flags, "reuse_text_cache", False)):
        return True
    for item in items:
        identity = _generation_card_asset_identity(payload=payload, item=item, l1_code=l1_code)
        if identity["asset_key"] not in text_cache_assets:
            return True
    return False


def _card_copy_without_cache_meta(card: Mapping[str, Any]) -> Dict[str, Any]:
    clean = dict(card or {})
    meta = dict(clean.get("meta") or {})
    meta.pop("text_cache", None)
    clean["meta"] = meta
    return clean


def _safe_export_basename(raw: str | None, fallback: str) -> str:
    value = (raw or "").strip()
    if not value:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _safe_media_filename(raw: str) -> str:
    name = (raw or "").strip().replace("\\", "/")
    name = name.split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not cleaned:
        raise ValueError("Invalid media filename")
    return cleaned


def _export_cards_to_dict(cards: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for card in cards:
        if hasattr(card, "model_dump"):
            data = card.model_dump()
        else:
            data = dict(card)
        out.append(
            {
                "L2_word": str(data.get("L2_word", "") or ""),
                "L2_cloze": str(data.get("L2_cloze", "") or ""),
                "L1_sentence": str(data.get("L1_sentence", "") or ""),
                "L2_collocations": str(data.get("L2_collocations", "") or ""),
                "L2_definition": str(data.get("L2_definition", "") or ""),
                "L1_gloss": str(data.get("L1_gloss", "") or ""),
                "L1_hint": str(data.get("L1_hint", "") or ""),
                "AudioSentence": str(data.get("AudioSentence", "") or ""),
                "AudioWord": str(data.get("AudioWord", "") or ""),
            }
        )
    return out


def _sort_model_id(model_id: str, preferred_order: Dict[str, int]) -> tuple[int, str]:
    for prefix, rank in preferred_order.items():
        if model_id.startswith(prefix):
            return (rank, model_id)
    return (999, model_id)


def _list_openai_model_ids() -> List[str]:
    api_key = read_secret("OPENAI_API_KEY", required=False)
    if not api_key:
        return []
    client = create_client(api_key)
    if client is None:
        return []
    try:
        models = client.models.list()
    except Exception:
        return []

    ids: set[str] = set()

    iterator = getattr(models, "auto_paging_iter", None)
    if callable(iterator):
        try:
            for model in iterator():
                mid = str(getattr(model, "id", "") or "").strip()
                if mid:
                    ids.add(mid)
            return sorted(ids)
        except Exception:
            pass

    page = models
    hops = 0
    while page is not None and hops < 30:
        for model in getattr(page, "data", []) or []:
            mid = str(getattr(model, "id", "") or "").strip()
            if mid:
                ids.add(mid)
        has_next = getattr(page, "has_next_page", None)
        get_next = getattr(page, "get_next_page", None)
        if not callable(has_next) or not callable(get_next):
            break
        try:
            if not has_next():
                break
            page = get_next()
        except Exception:
            break
        hops += 1

    return sorted(ids)


def _filter_text_models(model_ids: List[str]) -> List[str]:
    preferred_order = get_preferred_order()
    allowed_prefixes = get_allowed_prefixes()
    block_substrings = get_block_substrings()
    ids = [
        mid
        for mid in model_ids
        if any(mid.startswith(p) for p in allowed_prefixes) and not any(b in mid for b in block_substrings)
    ]
    if not ids:
        return list(DEFAULT_MODELS)
    return sorted(set(ids), key=lambda mid: _sort_model_id(mid, preferred_order))


def _filter_openai_tts_models(model_ids: List[str]) -> List[str]:
    discovered: List[str] = []
    for mid in model_ids:
        low = mid.lower()
        if "tts" not in low:
            continue
        if any(blocked in low for blocked in ("transcribe", "whisper", "realtime", "asr")):
            continue
        discovered.append(mid)

    discovered_unique = sorted(set(discovered))
    if discovered_unique:
        return discovered_unique

    # Fallback only when dynamic model listing is unavailable.
    fallback: List[str] = []
    for value in [AUDIO_TTS_MODEL, AUDIO_TTS_FALLBACK, "gpt-4o-mini-tts", "gpt-4o-tts", "tts-1", "tts-1-hd"]:
        clean = (value or "").strip()
        if clean and clean not in fallback:
            fallback.append(clean)
    return fallback


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


def _generate_one_item(
    *,
    client: Any | None,
    payload: GenerateRequest,
    item: Any,
    idx: int,
    l1_code: str,
    l1_meta: Dict[str, Any],
    signal_usage: Dict[str, int],
    signal_last: Any,
    text_cache_assets: Dict[str, Dict[str, Any]] | None = None,
    text_cache_touched_keys: List[str] | None = None,
) -> tuple[GenerateItemResult, Dict[str, Any], Dict[str, int], Any]:
    reuse_text_cache = bool(getattr(payload.flags, "reuse_text_cache", False))
    asset_identity = _generation_card_asset_identity(payload=payload, item=item, l1_code=l1_code)
    if reuse_text_cache:
        cache_error = None
        if text_cache_assets is not None:
            cached_asset = text_cache_assets.get(asset_identity["asset_key"])
        else:
            cached_asset, cache_error = load_generated_card_asset(asset_key=asset_identity["asset_key"])
        if cached_asset and isinstance(cached_asset.get("card_json"), dict):
            card = dict(cached_asset["card_json"])
            meta = dict(card.get("meta") or {})
            meta["provider"] = payload.provider or meta.get("provider") or "openai"
            meta["model"] = payload.model or meta.get("model") or "unknown"
            meta["text_cache"] = {
                "status": "hit",
                "asset_key": asset_identity["asset_key"],
            }
            meta["request"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_elapsed_ms": 0,
            }
            card["meta"] = meta
            status = _status_from_card(card)
            out_item = GenerateItemResult(
                id=item.id,
                status=status,  # type: ignore[arg-type]
                card=card if status in {"ok", "repaired"} else None,
                error=card.get("error") or None,
                usage=_usage_from_meta(meta),
            )
            if text_cache_touched_keys is not None:
                text_cache_touched_keys.append(asset_identity["asset_key"])
            else:
                try:
                    touch_generated_card_asset(asset_key=asset_identity["asset_key"])
                except Exception:
                    pass
            return out_item, card, signal_usage, signal_last
        if cache_error:
            logger.info("Generated-card cache lookup skipped: %s", cache_error)

    if client is None:
        client = _openai_client_or_500()

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
        prompt_version=payload.prompt_version,
        prompt_cache_retention=_env_text("OPENAI_PROMPT_CACHE_RETENTION"),
    )
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
    next_usage = result.signal_usage or signal_usage
    next_last = result.signal_last or signal_last
    card = result.card
    meta = card.get("meta") or {}
    status = _status_from_card(card)
    if reuse_text_cache and status in {"ok", "repaired"}:
        try:
            cache_card = _card_copy_without_cache_meta(card)
            stored, store_error = store_generated_card_asset(
                asset={
                    **asset_identity,
                    "card_json": cache_card,
                    "status": status,
                }
            )
            meta = dict(card.get("meta") or {})
            meta["text_cache"] = {
                "status": "stored" if stored else "store_failed",
                "asset_key": asset_identity["asset_key"],
                "error": store_error,
            }
            card["meta"] = meta
        except Exception as exc:
            meta = dict(card.get("meta") or {})
            meta["text_cache"] = {
                "status": "store_failed",
                "asset_key": asset_identity["asset_key"],
                "error": str(exc),
            }
            card["meta"] = meta
    out_item = GenerateItemResult(
        id=item.id,
        status=status,  # type: ignore[arg-type]
        card=card if status in {"ok", "repaired"} else None,
        error=card.get("error") or None,
        usage=_usage_from_meta(meta),
    )
    return out_item, card, next_usage, next_last


def _finalize_generate_response(
    *,
    payload: GenerateRequest,
    run_id: str,
    results_cards: List[Dict[str, Any]],
    items_out: List[GenerateItemResult],
    signal_usage: Dict[str, int],
    signal_last: Any,
    started: float,
    user_id: str,
) -> GenerateResponse:
    elapsed = time.time() - started
    run_stats = {
        "elapsed": elapsed,
        "batches": 1,
        "items": len(results_cards),
        "start_ts": started,
        "transient": 0,
    }
    state: Dict[str, Any] = {
        "results": results_cards,
        "sig_usage": signal_usage,
        "sig_last": signal_last,
        "audio_summary": {},
        "run_stats": run_stats,
    }
    run_report = build_run_report(SimpleNamespace(**state))
    text_cache_hits = 0
    text_assets_stored = 0
    text_cache_errors = 0
    for card in results_cards:
        meta = card.get("meta") if isinstance(card, dict) else {}
        cache_meta = (meta or {}).get("text_cache") if isinstance(meta, dict) else {}
        status = (cache_meta or {}).get("status") if isinstance(cache_meta, dict) else None
        if status == "hit":
            text_cache_hits += 1
        elif status == "stored":
            text_assets_stored += 1
        elif status == "store_failed":
            text_cache_errors += 1
    try:
        log_usage_events(user_id=user_id, run_id=run_id or "", events=run_report.get("usage_events", []))
    except Exception:
        pass
    return GenerateResponse(
        run_id=run_id,
        prompt_version=payload.prompt_version,
        provider=payload.provider,
        model=payload.model,
        items=items_out,
        run_report=run_report,
        timing={
            "elapsed_ms": int(elapsed * 1000),
            "text_cache_hits": text_cache_hits,
            "text_assets_stored": text_assets_stored,
            "text_cache_errors": text_cache_errors,
        },
    )


@app.post("/api/generate", response_model=GenerateResponse)
def api_generate(
    payload: GenerateRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> GenerateResponse:
    user_id = _require_user(request, x_api_key)
    client = None

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
    text_cache_assets, text_cache_error = _preload_generated_card_assets(
        payload=payload,
        items=list(payload.items or []),
        l1_code=l1_code,
    )
    if text_cache_error:
        logger.info("Generated-card cache preload skipped: %s", text_cache_error)
    if _needs_text_provider(
        payload=payload,
        items=list(payload.items or []),
        l1_code=l1_code,
        text_cache_assets=text_cache_assets,
    ):
        client = _openai_client_or_500()
    text_cache_touched_keys: List[str] = []
    for idx, item in enumerate(payload.items):
        try:
            out_item, card, signal_usage, signal_last = _generate_one_item(
                client=client,
                payload=payload,
                item=item,
                idx=idx,
                l1_code=l1_code,
                l1_meta=l1_meta,
                signal_usage=signal_usage,
                signal_last=signal_last,
                text_cache_assets=text_cache_assets,
                text_cache_touched_keys=text_cache_touched_keys,
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        items_out.append(out_item)
        results_cards.append(card)
    if text_cache_touched_keys:
        try:
            touch_generated_card_assets(asset_keys=text_cache_touched_keys)
        except Exception:
            pass

    return _finalize_generate_response(
        payload=payload,
        run_id=payload.run_id or "",
        results_cards=results_cards,
        items_out=items_out,
        signal_usage=signal_usage,
        signal_last=signal_last,
        started=started,
        user_id=user_id,
    )


@app.post("/api/jobs/generate", response_model=GenerateJobCreateResponse)
def api_create_generate_job(
    payload: GenerateRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> GenerateJobCreateResponse:
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    user_id = _require_user(request, x_api_key)
    run_id = (payload.run_id or "").strip() or str(uuid.uuid4())
    payload_data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    payload_data["run_id"] = run_id
    total_items = len(payload.items or [])
    job_id = create_generation_job(
        user_id=user_id,
        run_id=run_id,
        payload=payload_data,
        total_items=total_items,
    )
    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to create generation job")
    return GenerateJobCreateResponse(job_id=job_id, run_id=run_id, status="queued")


@app.get("/api/jobs/generate/{job_id}", response_model=GenerateJobStatusResponse)
def api_get_generate_job(
    job_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> GenerateJobStatusResponse:
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    user_id = _require_user(request, x_api_key)
    job = get_generation_job(job_id=job_id, user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result_payload = None
    if job.get("status") == "done":
        result_raw = job.get("result_json") or {}
        if isinstance(result_raw, dict) and result_raw:
            try:
                result_payload = GenerateResponse(**result_raw)
            except Exception:
                result_payload = None
    return GenerateJobStatusResponse(
        job_id=job["id"],
        run_id=job["run_id"],
        status=job["status"],
        processed_items=int(job.get("processed_items") or 0),
        total_items=int(job.get("total_items") or 0),
        error=(job.get("error_text") or None),
        result=result_payload,
        updated_at=job.get("updated_at"),
    )


@app.post("/api/jobs/generate/worker", response_model=GenerateJobWorkerResponse)
def api_generate_worker(
    payload: GenerateJobWorkerRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> GenerateJobWorkerResponse:
    ok_db, reason = db_status()
    if not ok_db:
        raise HTTPException(status_code=503, detail=reason)
    auth_header = (request.headers.get("Authorization") or "").strip()
    cron_secret = (os.getenv("CRON_SECRET") or "").strip()
    cron_authorized = bool(
        cron_secret
        and auth_header.startswith("Bearer ")
        and hmac.compare_digest(auth_header[len("Bearer ") :].strip(), cron_secret)
    )
    user_id: str | None = None
    if not cron_authorized:
        user_id = _require_user(request, x_api_key)

    stale_seconds = int(os.getenv("GENERATE_JOB_STALE_SECONDS", "90") or "90")
    default_max_items = int(os.getenv("GENERATE_JOB_MAX_ITEMS_PER_WORKER", "6") or "6")
    max_items = max(1, min(int(payload.max_items or default_max_items), 20))

    claimed = claim_generation_job(
        user_id=user_id if not cron_authorized else None,
        job_id=(payload.job_id or None),
        stale_seconds=stale_seconds,
    )
    if not claimed:
        return GenerateJobWorkerResponse(
            processed=False,
            message="No queued/runnable jobs",
        )

    job_id = claimed["id"]
    run_id = claimed["run_id"]
    job_owner_id = str(claimed.get("user_id") or user_id or "")
    try:
        request_payload = GenerateRequest(**(claimed.get("payload_json") or {}))
    except Exception as exc:
        fail_generation_job(job_id=job_id, error_text=f"Invalid payload: {exc}")
        return GenerateJobWorkerResponse(
            processed=True,
            job_id=job_id,
            status="failed",
            processed_items=int(claimed.get("processed_items") or 0),
            total_items=int(claimed.get("total_items") or 0),
            message="Job payload is invalid",
        )

    l1_code = (request_payload.l1 or "").strip().upper()
    l1_meta = L1_LANGS.get(l1_code)
    if not l1_meta:
        fail_generation_job(job_id=job_id, error_text=f"Unsupported l1={request_payload.l1!r}")
        return GenerateJobWorkerResponse(
            processed=True,
            job_id=job_id,
            status="failed",
            processed_items=int(claimed.get("processed_items") or 0),
            total_items=int(claimed.get("total_items") or 0),
            message="Unsupported L1 language",
        )

    state = claimed.get("state_json") or {}
    processed_items = int(claimed.get("processed_items") or 0)
    total_items = int(claimed.get("total_items") or len(request_payload.items or []))
    total_items = max(total_items, len(request_payload.items or []))

    items_out_raw = state.get("items_out") if isinstance(state.get("items_out"), list) else []
    results_cards = state.get("results_cards") if isinstance(state.get("results_cards"), list) else []
    signal_usage = state.get("signal_usage") if isinstance(state.get("signal_usage"), dict) else {}
    signal_last = state.get("signal_last")
    started_epoch = float(state.get("started_epoch") or time.time())

    client = None

    if processed_items >= total_items:
        existing = get_generation_job(job_id=job_id, user_id=(None if cron_authorized else user_id))
        status = (existing or {}).get("status", "running")
        return GenerateJobWorkerResponse(
            processed=True,
            job_id=job_id,
            status=status,
            processed_items=processed_items,
            total_items=total_items,
            message="Job already processed",
        )

    end_index = min(total_items, processed_items + max_items)
    chunk_items = list(request_payload.items[processed_items:end_index])
    text_cache_assets, text_cache_error = _preload_generated_card_assets(
        payload=request_payload,
        items=chunk_items,
        l1_code=l1_code,
    )
    if text_cache_error:
        logger.info("Generated-card cache preload skipped: %s", text_cache_error)
    try:
        if _needs_text_provider(
            payload=request_payload,
            items=chunk_items,
            l1_code=l1_code,
            text_cache_assets=text_cache_assets,
        ):
            client = _openai_client_or_500()
    except HTTPException as exc:
        fail_generation_job(job_id=job_id, error_text=str(exc.detail), state=state, processed_items=processed_items)
        return GenerateJobWorkerResponse(
            processed=True,
            job_id=job_id,
            status="failed",
            processed_items=processed_items,
            total_items=total_items,
            message=str(exc.detail),
        )
    text_cache_touched_keys: List[str] = []
    for idx in range(processed_items, end_index):
        item = request_payload.items[idx]
        try:
            out_item, card, signal_usage, signal_last = _generate_one_item(
                client=client,
                payload=request_payload,
                item=item,
                idx=idx,
                l1_code=l1_code,
                l1_meta=l1_meta,
                signal_usage=signal_usage,
                signal_last=signal_last,
                text_cache_assets=text_cache_assets,
                text_cache_touched_keys=text_cache_touched_keys,
            )
        except AuthenticationError as exc:
            fail_generation_job(
                job_id=job_id,
                error_text=str(exc),
                state={
                    "items_out": items_out_raw,
                    "results_cards": results_cards,
                    "signal_usage": signal_usage,
                    "signal_last": signal_last,
                    "started_epoch": started_epoch,
                },
                processed_items=idx,
            )
            return GenerateJobWorkerResponse(
                processed=True,
                job_id=job_id,
                status="failed",
                processed_items=idx,
                total_items=total_items,
                message=str(exc),
            )
        except HTTPException as exc:
            fail_generation_job(
                job_id=job_id,
                error_text=str(exc.detail),
                state={
                    "items_out": items_out_raw,
                    "results_cards": results_cards,
                    "signal_usage": signal_usage,
                    "signal_last": signal_last,
                    "started_epoch": started_epoch,
                },
                processed_items=idx,
            )
            return GenerateJobWorkerResponse(
                processed=True,
                job_id=job_id,
                status="failed",
                processed_items=idx,
                total_items=total_items,
                message=str(exc.detail),
            )
        except Exception as exc:
            fail_generation_job(
                job_id=job_id,
                error_text=f"Generation failed at item {idx + 1}: {exc}",
                state={
                    "items_out": items_out_raw,
                    "results_cards": results_cards,
                    "signal_usage": signal_usage,
                    "signal_last": signal_last,
                    "started_epoch": started_epoch,
                },
                processed_items=idx,
            )
            return GenerateJobWorkerResponse(
                processed=True,
                job_id=job_id,
                status="failed",
                processed_items=idx,
                total_items=total_items,
                message=str(exc),
            )

        items_out_raw.append(out_item.model_dump() if hasattr(out_item, "model_dump") else out_item.dict())
        results_cards.append(card)
        processed_items = idx + 1

        # Persist heartbeat/progress after each item so long-running serverless
        # invocations can be safely resumed without waiting for stale timeout.
        update_generation_job_progress(
            job_id=job_id,
            processed_items=processed_items,
            state={
                "items_out": items_out_raw,
                "results_cards": results_cards,
                "signal_usage": signal_usage,
                "signal_last": signal_last,
                "started_epoch": started_epoch,
            },
        )
    if text_cache_touched_keys:
        try:
            touch_generated_card_assets(asset_keys=text_cache_touched_keys)
        except Exception:
            pass

    new_state = {
        "items_out": items_out_raw,
        "results_cards": results_cards,
        "signal_usage": signal_usage,
        "signal_last": signal_last,
        "started_epoch": started_epoch,
    }

    if processed_items < total_items:
        update_generation_job_progress(
            job_id=job_id,
            processed_items=processed_items,
            state=new_state,
        )
        return GenerateJobWorkerResponse(
            processed=True,
            job_id=job_id,
            status="running",
            processed_items=processed_items,
            total_items=total_items,
            message="Progress saved",
        )

    items_out: List[GenerateItemResult] = []
    for raw in items_out_raw:
        try:
            items_out.append(GenerateItemResult(**raw))
        except Exception:
            pass
    response_payload = _finalize_generate_response(
        payload=request_payload,
        run_id=run_id,
        results_cards=results_cards,
        items_out=items_out,
        signal_usage=signal_usage,
        signal_last=signal_last,
        started=started_epoch,
        user_id=job_owner_id,
    )
    result_data = response_payload.model_dump() if hasattr(response_payload, "model_dump") else response_payload.dict()
    complete_generation_job(
        job_id=job_id,
        result=result_data,
        processed_items=processed_items,
    )
    return GenerateJobWorkerResponse(
        processed=True,
        job_id=job_id,
        status="done",
        processed_items=processed_items,
        total_items=total_items,
        message="Job completed",
    )


@app.get("/api/jobs/generate/worker", response_model=GenerateJobWorkerResponse)
def api_generate_worker_cron(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    max_items: int | None = None,
) -> GenerateJobWorkerResponse:
    payload = GenerateJobWorkerRequest(job_id=None, max_items=max_items)
    return api_generate_worker(payload=payload, request=request, x_api_key=x_api_key)


@app.post("/api/export/csv", response_model=ExportFileResponse)
def api_export_csv(
    payload: ExportDeckRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> ExportFileResponse:
    user_id = _require_user(request, x_api_key)

    if not payload.cards:
        raise HTTPException(status_code=400, detail="No cards to export")

    l1_code = (payload.l1 or "").strip().upper()
    l1_meta = L1_LANGS.get(l1_code)
    if not l1_meta:
        supported = ", ".join(sorted(L1_LANGS.keys()))
        raise HTTPException(status_code=400, detail=f"Unsupported l1={payload.l1!r}. Supported: {supported}")

    cards = _export_cards_to_dict(payload.cards)
    csv_data = generate_csv(
        cards,
        l1_meta,
        delimiter=CSV_DELIMITER,
        line_terminator=CSV_LINETERMINATOR,
        include_header=True,
        include_extras=True,
        anki_field_header=True,
        extras_meta={
            "level": payload.cefr,
            "profile": payload.profile,
            "model": payload.model,
            "L1": l1_code,
        },
    )
    base = _safe_export_basename(payload.deck_name, ANKI_DECK_NAME)
    return ExportFileResponse(
        file_name=f"{base}.csv",
        mime_type="text/csv",
        content_b64=base64.b64encode(csv_data.encode("utf-8")).decode("ascii"),
        card_count=len(cards),
    )


@app.post("/api/export/apkg")
def api_export_apkg(
    payload: ExportDeckRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    user_id = _require_user(request, x_api_key)

    if not payload.cards:
        raise HTTPException(status_code=400, detail="No cards to export")
    if not HAS_GENANKI:
        raise HTTPException(status_code=501, detail="APKG export is disabled: genanki is not installed on the API server.")

    estimated_request_size = _estimate_export_request_size_bytes(payload)
    if estimated_request_size >= EXPORT_REQUEST_SOFT_LIMIT_BYTES:
        size_mb = estimated_request_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"APKG export request is too large for Vercel ({size_mb:.2f} MB estimated; limit is about "
                f"{VERCEL_FUNCTION_BODY_LIMIT_BYTES / (1024 * 1024):.1f} MB). "
                "Try fewer cards, disable audio, or split the deck into smaller batches."
            ),
        )

    l1_code = (payload.l1 or "").strip().upper()
    l1_meta = L1_LANGS.get(l1_code)
    if not l1_meta:
        supported = ", ".join(sorted(L1_LANGS.keys()))
        raise HTTPException(status_code=400, detail=f"Unsupported l1={payload.l1!r}. Supported: {supported}")

    cards = _export_cards_to_dict(payload.cards)
    run_id = (payload.run_id or "").strip() or str(int(time.time()))
    deck_name = (payload.deck_name or "").strip() or ANKI_DECK_NAME
    media_files = _resolve_export_media_files(payload=payload, user_id=user_id, cards=cards)

    try:
        front_template_raw = CLOZE_FRONT_TEMPLATE_PATH.read_text(encoding="utf-8")
        front_html = front_template_raw.replace("{L1_LABEL}", l1_meta.get("label", l1_code))
        back_html = CLOZE_BACK_TEMPLATE_PATH.read_text(encoding="utf-8")
        css_content = CLOZE_CSS_PATH.read_text(encoding="utf-8")
        basic_templates = None
        if payload.include_basic_reversed:
            basic_templates = {
                "card1_front": BASIC_CARD1_FRONT_TEMPLATE_PATH.read_text(encoding="utf-8"),
                "card1_back": BASIC_CARD1_BACK_TEMPLATE_PATH.read_text(encoding="utf-8"),
                "card2_front": BASIC_CARD2_FRONT_TEMPLATE_PATH.read_text(encoding="utf-8"),
                "card2_back": BASIC_CARD2_BACK_TEMPLATE_PATH.read_text(encoding="utf-8"),
            }
        typein_templates = None
        if payload.include_basic_typein:
            typein_templates = {
                "front": TYPEIN_FRONT_TEMPLATE_PATH.read_text(encoding="utf-8"),
                "back": TYPEIN_BACK_TEMPLATE_PATH.read_text(encoding="utf-8"),
            }
        anki_bytes = build_anki_package(
            cards,
            l1_label=l1_meta.get("label", l1_code),
            guid_policy=payload.guid_policy,
            run_id=run_id,
            model_id=ANKI_MODEL_ID,
            model_name=ANKI_MODEL_NAME,
            deck_id=ANKI_DECK_ID,
            deck_name=deck_name,
            front_template=front_html,
            back_template=back_html,
            css=css_content,
            tags_meta={
                "level": payload.cefr,
                "profile": payload.profile,
                "model": payload.model,
                "L1": l1_code,
            },
            media_files=media_files,
            include_basic_reversed=payload.include_basic_reversed,
            include_basic_typein=payload.include_basic_typein,
            basic_templates=basic_templates,
            typein_templates=typein_templates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build .apkg: {exc}") from exc

    base = _safe_export_basename(deck_name, ANKI_DECK_NAME)
    file_name = f"{base}.apkg"
    headers = {
        "Content-Disposition": f'attachment; filename="{file_name}"',
        "X-Card-Count": str(len(cards)),
        "Access-Control-Expose-Headers": "Content-Disposition, X-Card-Count",
    }
    return StreamingResponse(
        _iter_bytesio(io.BytesIO(anki_bytes)),
        media_type="application/octet-stream",
        headers=headers,
    )


@app.get("/api/tts/options", response_model=TTSOptionsResponse)
def api_tts_options(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> TTSOptionsResponse:
    _require_user(request, x_api_key)

    all_openai_models = _list_openai_model_ids()
    text_models = _filter_text_models(all_openai_models)
    openai_models = _filter_openai_tts_models(all_openai_models)

    openai_voices = [
        TTSOption(id=(voice.get("id") or "").strip(), label=(voice.get("label") or voice.get("id") or "").strip())
        for voice in AUDIO_VOICES
        if (voice.get("id") or "").strip()
    ]
    eleven_voices = [
        TTSOption(id=(voice.get("id") or "").strip(), label=(voice.get("label") or voice.get("id") or "").strip())
        for voice in AUDIO_ELEVEN_VOICES
        if (voice.get("id") or "").strip()
    ]
    eleven_models = ["eleven_multilingual_v2"]

    return TTSOptionsResponse(
        text_models=text_models,
        providers=["openai", "elevenlabs"],
        by_provider={
            "openai": TTSProviderOptions(
                models=openai_models,
                voices=openai_voices,
                default_model=openai_models[0] if openai_models else None,
                default_voice=openai_voices[0].id if openai_voices else None,
            ),
            "elevenlabs": TTSProviderOptions(
                models=eleven_models,
                voices=eleven_voices,
                default_model=eleven_models[0],
                default_voice=eleven_voices[0].id if eleven_voices else None,
            ),
        },
    )


@app.post("/api/audio/assets/check", response_model=AudioAssetCheckResponse)
def api_audio_assets_check(
    payload: AudioAssetCheckRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> AudioAssetCheckResponse:
    _require_user(request, x_api_key)

    requested: list[str] = []
    for raw_name in payload.filenames or []:
        try:
            filename = _safe_media_filename(raw_name)
        except ValueError:
            continue
        if filename:
            requested.append(filename)
    requested_unique = sorted(set(requested))
    found, error = list_existing_audio_asset_filenames(filenames=requested_unique)
    found_sorted = sorted(name for name in requested_unique if name in found)
    missing = sorted(name for name in requested_unique if name not in found)
    return AudioAssetCheckResponse(found=found_sorted, missing=missing, error=error)


def _extract_sound_filename(sound_field: str) -> str:
    if not sound_field:
        return ""
    if sound_field.startswith("[sound:") and sound_field.endswith("]"):
        return sound_field[len("[sound:") : -1]
    return sound_field


def _referenced_audio_filenames(cards: List[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for card in cards or []:
        for field_name in ("AudioWord", "AudioSentence"):
            raw_value = str(card.get(field_name, "") or "").strip()
            if not raw_value:
                continue
            extracted = _extract_sound_filename(raw_value)
            if not extracted:
                continue
            filename = _safe_media_filename(extracted)
            if filename:
                out.add(filename)
    return out


def _resolve_export_media_files(
    *,
    payload: ExportDeckRequest,
    user_id: str,
    cards: List[Dict[str, Any]],
) -> Dict[str, bytes] | None:
    media_files: Dict[str, bytes] = {}
    for raw_name, content_b64 in (payload.media_map or {}).items():
        if not content_b64:
            continue
        safe_name = _safe_media_filename(raw_name)
        try:
            media_files[safe_name] = base64.b64decode(content_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid media payload for {safe_name}") from exc

    referenced = _referenced_audio_filenames(cards)
    if referenced:
        missing = sorted(name for name in referenced if name not in media_files)
        storage_error: str | None = None
        if missing and payload.run_id:
            persisted_media, storage_error = load_run_media_assets(
                user_id=user_id,
                run_id=payload.run_id,
                filenames=missing,
            )
            media_files.update(persisted_media)
            missing = sorted(name for name in referenced if name not in media_files)
        if missing:
            reusable_media, reusable_error = load_audio_assets_by_filenames(filenames=missing)
            media_files.update(reusable_media)
            if reusable_error and not storage_error:
                storage_error = reusable_error
            missing = sorted(name for name in referenced if name not in media_files)

        if missing:
            detail = (
                f"APKG export is missing {len(missing)} audio file(s). "
                f"First missing: {missing[0]}."
            )
            if payload.use_persisted_media and not payload.run_id:
                detail += " run_id is required for persisted-media export."
            elif storage_error:
                detail += f" Storage lookup error: {storage_error}"
            elif payload.use_persisted_media:
                detail += " The audio batch may not have been persisted yet."
            raise HTTPException(status_code=409, detail=detail)

    return media_files or None


@app.post("/api/tts", response_model=TTSResponse)
def api_tts(
    payload: TTSRequest,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> TTSResponse:
    endpoint_started = time.time()
    user_id = _require_user(request, x_api_key)
    provider = (payload.provider or "openai").strip().lower()

    if provider not in {"openai", "elevenlabs"}:
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

    resolved_model = (payload.model or AUDIO_TTS_MODEL) if provider == "openai" else (payload.model or "eleven_multilingual_v2")

    asset_entries: List[Dict[str, Any]] = []
    requested_asset_keys: List[str] = []
    for idx, item in enumerate(payload.items):
        tts_text = (item.text or "").strip() if item.type == "word" else sentence_for_tts(item.text or "")
        if not tts_text:
            continue
        identity = tts_asset_identity(
            provider=provider,
            model=resolved_model,
            voice=voice,
            kind=item.type,
            text=tts_text,
            instruction_payload=instruction_payloads.get(item.type),
        )
        entry = {
            "index": idx,
            "item": item,
            "text": tts_text,
            "identity": identity,
        }
        asset_entries.append(entry)
        requested_asset_keys.append(identity["asset_key"])

    durable_assets: Dict[str, Dict[str, Any]] = {}
    durable_cache_error: str | None = None
    if requested_asset_keys:
        durable_assets, durable_cache_error = load_audio_assets(asset_keys=sorted(set(requested_asset_keys)))

    cards_to_generate: List[Dict[str, Any]] = []
    generation_entries: List[Dict[str, Any]] = []
    media_map: Dict[str, bytes] = {}
    summary = AudioSynthesisSummary(provider=provider, voice=voice)
    if instruction_keys:
        summary.sentence_instruction_key = instruction_keys.get("sentence", "")
        summary.word_instruction_key = instruction_keys.get("word", "")

    durable_hit_keys: List[str] = []
    for entry in asset_entries:
        idx = int(entry["index"])
        item = entry["item"]
        identity = entry["identity"]
        asset = durable_assets.get(identity["asset_key"])
        if asset:
            filename = str(asset.get("filename") or identity["filename"]).strip()
            content = asset.get("content") or b""
            if filename and content:
                media_map[filename] = bytes(content)
                if item.type == "word":
                    cards[idx]["AudioWord"] = f"[sound:{filename}]"
                    summary.word_success += 1
                else:
                    cards[idx]["AudioSentence"] = f"[sound:{filename}]"
                    summary.sentence_success += 1
                summary.cache_hits += 1
                durable_hit_keys.append(identity["asset_key"])
                summary.clip_results.append(
                    AudioClipResult(
                        card_index=idx,
                        kind=item.type,
                        text=str(entry["text"]),
                        status="cached",
                        filename=filename,
                        error="",
                        model=str(asset.get("model") or resolved_model),
                    )
                )
                continue
        generation_entries.append(entry)
        cards_to_generate.append(
            {
                "id": item.card_id,
                "L2_word": item.text if item.type == "word" else "",
                "L2_cloze": item.text if item.type == "sentence" else "",
                "AudioWord": "",
                "AudioSentence": "",
            }
        )

    if durable_hit_keys:
        try:
            touch_audio_assets(asset_keys=sorted(set(durable_hit_keys)))
        except Exception:
            pass

    synth_started = time.time()
    generated_media_map: Dict[str, bytes] = {}
    generated_summary = AudioSynthesisSummary(provider=provider, voice=voice)
    audio_assets_stored = 0
    audio_assets_storage_error: str | None = None
    if generation_entries:
        openai_client = None
        eleven_api_key = None
        if provider == "openai":
            openai_client = _openai_client_or_500()
            tts_max_workers = _env_int("OPENAI_TTS_MAX_WORKERS", 4, min_value=1, max_value=4)
        elif provider == "elevenlabs":
            eleven_api_key = _elevenlabs_key_or_500()
            tts_max_workers = _env_int("ELEVENLABS_TTS_MAX_WORKERS", 2, min_value=1, max_value=2)
        else:
            tts_max_workers = 1
        try:
            generated_media_map, generated_summary = ensure_audio_for_cards(
                cards_to_generate,
                provider=provider,
                voice=voice,
                include_word=include_word,
                include_sentence=include_sentence,
                instruction_payloads=instruction_payloads,
                instruction_keys=instruction_keys,
                openai_client=openai_client,
                openai_model=resolved_model if provider == "openai" else None,
                openai_fallback_model=AUDIO_TTS_FALLBACK if provider == "openai" else None,
                eleven_api_key=eleven_api_key,
                eleven_model=(payload.model or None) if provider == "elevenlabs" else None,
                max_workers=tts_max_workers,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"TTS pipeline failed: {exc}") from exc

        media_map.update(generated_media_map)
        summary.word_success += generated_summary.word_success
        summary.sentence_success += generated_summary.sentence_success
        summary.word_skipped += generated_summary.word_skipped
        summary.sentence_skipped += generated_summary.sentence_skipped
        summary.errors.extend(generated_summary.errors or [])
        summary.fallback_switches += generated_summary.fallback_switches
        summary.total_characters += generated_summary.total_characters
        summary.total_requests_billed += generated_summary.total_requests_billed
        summary.total_requests += generated_summary.total_requests
        summary.cache_hits += generated_summary.cache_hits
        for model_name, data in (generated_summary.model_usage or {}).items():
            target = summary.model_usage.setdefault(model_name, {key: 0 for key in data.keys()})
            for key, value in data.items():
                target[key] = int(target.get(key, 0) or 0) + int(value or 0)

        generated_assets_to_store: List[Dict[str, Any]] = []
        for clip in generated_summary.clip_results:
            subset_idx = int(clip.card_index)
            if subset_idx < 0 or subset_idx >= len(generation_entries):
                continue
            entry = generation_entries[subset_idx]
            original_idx = int(entry["index"])
            clip.card_index = original_idx
            summary.clip_results.append(clip)
            generated_card = cards_to_generate[subset_idx]
            cards[original_idx]["AudioWord"] = generated_card.get("AudioWord", "")
            cards[original_idx]["AudioSentence"] = generated_card.get("AudioSentence", "")

            if clip.status not in {"ok", "cached"} or not clip.filename:
                continue
            content = generated_media_map.get(clip.filename)
            if not content:
                continue
            item = entry["item"]
            model_for_asset = str(getattr(clip, "model", "") or resolved_model)
            identity = tts_asset_identity(
                provider=provider,
                model=model_for_asset,
                voice=voice,
                kind=item.type,
                text=str(entry["text"]),
                instruction_payload=instruction_payloads.get(item.type),
            )
            generated_assets_to_store.append({**identity, "content": content, "filename": clip.filename})

        if generated_assets_to_store:
            try:
                audio_assets_stored, audio_assets_storage_error = store_audio_assets(assets=generated_assets_to_store)
            except Exception as exc:
                audio_assets_storage_error = str(exc)
    synth_elapsed_ms = int((time.time() - synth_started) * 1000)

    # Build response audios
    clip_map: Dict[tuple[int, str], Any] = {}
    for clip in summary.clip_results:
        key = (int(clip.card_index), str(clip.kind))
        clip_map[key] = clip

    audios: List[TTSAudio] = []
    for idx, (item, card) in enumerate(zip(payload.items, cards)):
        clip = clip_map.get((idx, item.type))
        status = str(getattr(clip, "status", "") or "").strip().lower()
        if status not in {"ok", "failed", "cached"}:
            status = "failed"

        clip_error = str(getattr(clip, "error", "") or "").strip()
        if status == "failed" and not clip_error:
            if not (item.text or "").strip():
                clip_error = "Text is empty."
            else:
                clip_error = "Audio synthesis did not return a clip."

        filename = str(getattr(clip, "filename", "") or "").strip()
        if not filename:
            if item.type == "word":
                sound_field = card.get("AudioWord", "")
            else:
                sound_field = card.get("AudioSentence", "")
            filename = _extract_sound_filename(sound_field)
        if not filename:
            status = "failed"

        data = media_map.get(filename, b"") if filename else b""
        if status in {"ok", "cached"} and not data:
            status = "failed"
            if not clip_error:
                clip_error = "Audio bytes are missing in media payload."

        model_used = str(getattr(clip, "model", "") or "").strip() or resolved_model
        usage_entry = None
        # We approximate per-item audio chars by len(text)
        if status in {"ok", "cached"} and data:
            usage_entry = UsageEvent(
                provider=summary.provider or (payload.provider or "openai"),
                model=model_used,
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
                status=status,
                filename=filename or None,
                audio_b64=base64.b64encode(data).decode("ascii") if data else None,
                error=clip_error or None,
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

    ok_count = sum(1 for audio in audios if audio.status in {"ok", "cached"})
    failed_count = sum(1 for audio in audios if audio.status == "failed")
    cached_count = sum(1 for audio in audios if audio.status == "cached")
    summary_errors: List[str] = []
    for err in list(summary.errors or []):
        text = str(err or "").strip()
        if text and text not in summary_errors:
            summary_errors.append(text)
    for audio in audios:
        if audio.status == "failed" and audio.error:
            text = str(audio.error or "").strip()
            if text and text not in summary_errors:
                summary_errors.append(text)

    summary_block = TTSSummary(
        ok=ok_count,
        failed=failed_count,
        cached=cached_count,
        errors=summary_errors,
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

    storage_block: TTSStorageInfo | None = None
    storage_elapsed_ms = 0
    if media_map:
        storage_started = time.time()
        stored_count, storage_error = store_run_media_assets(
            user_id=user_id,
            run_id=payload.run_id or "",
            media_files=media_map,
        )
        storage_elapsed_ms = int((time.time() - storage_started) * 1000)
        storage_block = TTSStorageInfo(
            persisted=bool(payload.run_id) and stored_count >= len(media_map),
            stored_clips=stored_count,
            error=storage_error,
        )
    elif payload.run_id:
        storage_block = TTSStorageInfo(persisted=True, stored_clips=0, error=None)

    endpoint_elapsed_ms = int((time.time() - endpoint_started) * 1000)
    timing_block = {
        "elapsed_ms": endpoint_elapsed_ms,
        "synthesis_ms": synth_elapsed_ms,
        "storage_ms": storage_elapsed_ms,
        "items": len(payload.items or []),
        "unique_media_files": len(media_map or {}),
        "cache_hits": int(summary.cache_hits or 0),
        "durable_cache_hits": len(durable_hit_keys),
        "durable_cache_error": durable_cache_error,
        "audio_assets_stored": audio_assets_stored,
        "audio_assets_storage_error": audio_assets_storage_error,
        "total_requests": int(summary.total_requests or 0),
        "provider": provider,
    }

    return TTSResponse(
        run_id=payload.run_id or "",
        provider=payload.provider,
        model=resolved_model,
        audios=audios,
        summary=summary_block,
        storage=storage_block,
        timing=timing_block,
    )
