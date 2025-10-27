"""Voice catalogue helpers for the audio panel."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Optional, Sequence

from core.audio import fetch_elevenlabs_voices


_CACHE_KEY = "elevenlabs_voice_catalog_cache"
_ERROR_KEY = "elevenlabs_voice_catalog_errors"
_LOADING_KEY = "elevenlabs_voice_catalog_loading"
_ATTEMPT_KEY = "elevenlabs_voice_catalog_attempts"


def _ensure_dict(store: MutableMapping[str, Any], key: str) -> Dict[str, Any]:
    value = store.get(key)
    if not isinstance(value, dict):
        value = {}
        store[key] = value
    return value


@dataclass
class ElevenLabsCatalog:
    """Metadata about the cached ElevenLabs voice catalogue."""

    voices: List[Dict[str, Any]]
    error: Optional[str]
    updated_at: Optional[float]


def elevenlabs_cache_key(api_key: str, language_codes: Optional[Sequence[str]]) -> str:
    if not api_key:
        return ""
    normalized = ",".join(sorted(code.strip().lower() for code in (language_codes or []) if code))
    digest = hashlib.sha1(api_key.encode("utf-8")).hexdigest()
    return f"{digest}:{normalized or 'all'}"


def _attempts(store: MutableMapping[str, Any]) -> Dict[str, float]:
    attempts = store.get(_ATTEMPT_KEY)
    if not isinstance(attempts, dict):
        attempts = {}
        store[_ATTEMPT_KEY] = attempts
    return attempts


def record_attempt(store: MutableMapping[str, Any], cache_key: str, *, timestamp: Optional[float] = None) -> None:
    if not cache_key:
        return
    attempts = _attempts(store)
    attempts[cache_key] = float(timestamp if timestamp is not None else time.time())


def should_throttle(store: MutableMapping[str, Any], cache_key: str, *, cooldown_seconds: float = 3.0) -> bool:
    if not cache_key or cooldown_seconds <= 0:
        return False
    attempts = _attempts(store)
    last_attempt = attempts.get(cache_key)
    if not isinstance(last_attempt, (int, float)):
        return False
    return (time.time() - float(last_attempt)) < cooldown_seconds


def get_catalog(store: MutableMapping[str, Any], cache_key: str) -> ElevenLabsCatalog:
    if not cache_key:
        return ElevenLabsCatalog([], None, None)

    cache = _ensure_dict(store, _CACHE_KEY)
    entry = cache.get(cache_key)
    voices: List[Dict[str, Any]] = []
    updated_at: Optional[float] = None

    if isinstance(entry, dict):
        candidate = entry.get("voices")
        if isinstance(candidate, list):
            voices = candidate
        timestamp = entry.get("updated_at")
        if isinstance(timestamp, (int, float)):
            updated_at = float(timestamp)

    errors = _ensure_dict(store, _ERROR_KEY)
    err_value = errors.get(cache_key)
    error = err_value if isinstance(err_value, str) and err_value else None

    return ElevenLabsCatalog(voices=voices, error=error, updated_at=updated_at)


def refresh_catalog(
    store: MutableMapping[str, Any],
    *,
    cache_key: str,
    api_key: str,
    language_codes: Optional[Sequence[str]],
) -> ElevenLabsCatalog:
    if not cache_key:
        raise ValueError("Cache key is required to refresh ElevenLabs voices")
    if not api_key:
        raise ValueError("ElevenLabs API key is required")

    try:
        voices = fetch_elevenlabs_voices(api_key, language_codes=language_codes)
    except Exception as exc:  # pragma: no cover - network dependent
        errors = _ensure_dict(store, _ERROR_KEY)
        errors[cache_key] = str(exc)
        return ElevenLabsCatalog([], str(exc), None)

    cache = _ensure_dict(store, _CACHE_KEY)
    cache[cache_key] = {
        "voices": voices,
        "updated_at": time.time(),
        "language_codes": list(language_codes or []),
    }

    errors = _ensure_dict(store, _ERROR_KEY)
    errors.pop(cache_key, None)

    return get_catalog(store, cache_key)


def is_loading(store: MutableMapping[str, Any]) -> bool:
    return bool(store.get(_LOADING_KEY))


def set_loading(store: MutableMapping[str, Any], value: bool) -> None:
    store[_LOADING_KEY] = bool(value)


def clear_catalog(store: MutableMapping[str, Any], cache_key: str) -> None:
    if not cache_key:
        return
    cache = _ensure_dict(store, _CACHE_KEY)
    cache.pop(cache_key, None)
    errors = _ensure_dict(store, _ERROR_KEY)
    errors.pop(cache_key, None)
    attempts = _attempts(store)
    attempts.pop(cache_key, None)
