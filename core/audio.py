"""Utilities for post-generation text-to-speech (TTS) synthesis."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import requests

__all__ = [
    "AudioClipResult",
    "AudioSynthesisSummary",
    "ensure_audio_for_cards",
    "fetch_elevenlabs_voice",
    "fetch_elevenlabs_voices",
    "sentence_for_tts",
    "tts_asset_identity",
]


_CLOZE_RE = re.compile(r"\{\{c[12]::(.*?)(?:::[^}]*)?\}\}")
_WHITESPACE_RE = re.compile(r"\s+")

AUDIO_CACHE_DIR_ENV = "AUDIO_CACHE_DIR"


def _resolve_audio_cache_dir() -> Optional[Path]:
    configured = (os.getenv(AUDIO_CACHE_DIR_ENV) or "").strip()
    candidates: List[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path("cache") / "audio")
    candidates.append(Path(tempfile.gettempdir()) / "doedutch-cache" / "audio")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return None


AUDIO_CACHE_DIR = _resolve_audio_cache_dir()


def _disk_cache_path(key: str) -> Optional[Path]:
    if AUDIO_CACHE_DIR is None:
        return None
    return AUDIO_CACHE_DIR / f"{key}.bin"


def _load_disk_cache(key: str) -> Optional[bytes]:
    path = _disk_cache_path(key)
    if path is None or not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _store_disk_cache(key: str, data: bytes) -> None:
    path = _disk_cache_path(key)
    if path is None:
        return
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@dataclass
class AudioSynthesisSummary:
    """Aggregated statistics about a TTS run."""

    total_requests: int = 0
    cache_hits: int = 0
    word_success: int = 0
    sentence_success: int = 0
    word_skipped: int = 0
    sentence_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    fallback_switches: int = 0
    sentence_instruction_key: str = ""
    word_instruction_key: str = ""
    provider: str = ""
    voice: str = ""
    model_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_characters: int = 0
    total_requests_billed: int = 0
    clip_results: List["AudioClipResult"] = field(default_factory=list)

    def register_usage(self, model: str, kind: str, char_count: int, *, fallback_used: bool) -> None:
        """Track billed usage for model/voice combination."""
        if char_count <= 0:
            return
        model_key = model or "unknown"
        entry = self.model_usage.setdefault(
            model_key,
            {
                "chars": 0,
                "requests": 0,
                "fallback_requests": 0,
                "word_chars": 0,
                "sentence_chars": 0,
                "word_requests": 0,
                "sentence_requests": 0,
            },
        )
        entry["chars"] += char_count
        entry["requests"] += 1
        if fallback_used:
            entry["fallback_requests"] += 1
        if kind == "word":
            entry["word_chars"] += char_count
            entry["word_requests"] += 1
        else:
            entry["sentence_chars"] += char_count
            entry["sentence_requests"] += 1
        self.total_characters += char_count
        self.total_requests_billed += 1


@dataclass
class AudioClipResult:
    card_index: int
    kind: str
    text: str
    status: str
    filename: str = ""
    error: str = ""
    model: str = ""
    fallback_used: bool = False


def sentence_for_tts(cloze_text: str) -> str:
    """Return pronunciation-friendly sentence by removing cloze markup."""

    if not cloze_text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1) or ""
        return inner

    stripped = _CLOZE_RE.sub(_replace, cloze_text)
    cleaned = _WHITESPACE_RE.sub(" ", stripped).strip()
    return cleaned


def _slugify(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "audio"


def _payload_token(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha1(bytes(payload)).hexdigest()
    try:
        return json.dumps(payload, sort_keys=True)
    except Exception:
        return str(payload)


def _cache_key(model: str, voice: str, text: str, instructions: Optional[str] = None) -> str:
    digest = hashlib.sha256(
        f"{model}|{voice}|{instructions or ''}|{text}".encode("utf-8")
    ).hexdigest()
    return digest


def _filename(kind: str, voice: str, text: str) -> str:
    slug = _slugify(text)[:40]
    digest = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()[:8]
    return f"{kind}_{slug}__{voice}__{digest}.mp3"


def _openai_instruction_text(instruction_payload: Any) -> Optional[str]:
    if isinstance(instruction_payload, dict):
        raw_instr = instruction_payload.get("instructions")
        if isinstance(raw_instr, str):
            return raw_instr.strip() or None
    if isinstance(instruction_payload, str):
        return instruction_payload.strip() or None
    return None


def _elevenlabs_style_token(instruction_payload: Any) -> tuple[str, str]:
    voice_settings: Optional[Dict[str, Any]] = None
    spoken_language = "nl"
    if isinstance(instruction_payload, dict):
        vs = instruction_payload.get("voice_settings")
        if isinstance(vs, dict):
            voice_settings = vs
        sl = instruction_payload.get("spoken_language")
        if isinstance(sl, str) and sl:
            spoken_language = sl
    token_source = {
        "voice_settings": voice_settings,
        "spoken_language": spoken_language,
    }
    return _payload_token(token_source), spoken_language


def tts_asset_identity(
    *,
    provider: str,
    model: str,
    voice: str,
    kind: str,
    text: str,
    instruction_payload: Any = None,
) -> Dict[str, str]:
    """Return deterministic metadata for durable TTS asset reuse."""
    provider_key = (provider or "openai").strip().lower()
    model_key = (model or "").strip()
    voice_key = (voice or "").strip()
    kind_key = (kind or "").strip().lower()
    text_key = _WHITESPACE_RE.sub(" ", (text or "").strip())

    if provider_key == "openai":
        style_token = _payload_token(_openai_instruction_text(instruction_payload))
        spoken_language = ""
        output_format = "mp3"
    elif provider_key == "elevenlabs":
        style_token, spoken_language = _elevenlabs_style_token(instruction_payload)
        output_format = "mp3_44100_128"
    else:
        style_token = _payload_token(instruction_payload)
        spoken_language = ""
        output_format = "mp3"

    style_hash = hashlib.sha256(style_token.encode("utf-8")).hexdigest()
    text_hash = hashlib.sha256(text_key.encode("utf-8")).hexdigest()
    key_material = "|".join(
        [
            "tts-v1",
            provider_key,
            model_key,
            voice_key,
            kind_key,
            output_format,
            spoken_language,
            style_hash,
            text_hash,
        ]
    )
    return {
        "asset_key": hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        "provider": provider_key,
        "model": model_key,
        "voice": voice_key,
        "kind": kind_key,
        "text": text_key,
        "text_hash": text_hash,
        "style_hash": style_hash,
        "spoken_language": spoken_language,
        "output_format": output_format,
        "filename": _filename(kind_key, voice_key, text_key),
    }


def _is_model_not_found(error: Exception) -> bool:
    message = str(error).lower()
    return "model_not_found" in message or "does not exist" in message or "404" in message


def _status_code_from_error(error: Exception) -> Optional[int]:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_transient_tts_error(error: Exception) -> bool:
    status_code = _status_code_from_error(error)
    if status_code == 429 or (status_code is not None and 500 <= status_code <= 599):
        return True
    message = str(error).lower()
    transient_markers = (
        "429",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "503",
        "502",
        "504",
        "bad gateway",
        "service unavailable",
    )
    return any(marker in message for marker in transient_markers)


def _call_with_single_retry(callable_: Callable[[], bytes], *, retry_backoff: float = 0.75) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            return callable_()
        except Exception as err:  # pragma: no cover - network dependent
            last_error = err
            if attempt >= 1 or not _is_transient_tts_error(err):
                raise
            time.sleep(retry_backoff * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected retry state")


def _openai_tts_timeout_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("OPENAI_TTS_TIMEOUT_SECONDS", "12")))
    except Exception:
        return 12.0


def _extract_bytes(response: object) -> bytes:
    # New SDKs expose .read()
    read = getattr(response, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data
    # Fallback: streaming content list with base64
    content = getattr(response, "content", None)
    if isinstance(content, (list, tuple)):
        for item in content:
            audio = getattr(item, "audio", None)
            b64 = getattr(item, "b64_json", None)
            if isinstance(audio, (bytes, bytearray)):
                return bytes(audio)
            if isinstance(b64, str):
                return base64.b64decode(b64)
    audio_attr = getattr(response, "audio", None)
    if isinstance(audio_attr, str):
        try:
            return base64.b64decode(audio_attr)
        except Exception:
            pass
    if isinstance(audio_attr, (bytes, bytearray)):
        return bytes(audio_attr)
    raise ValueError("Unable to extract audio bytes from response")


def _synthesize(
    client: OpenAI,
    *,
    model: str,
    voice: str,
    text: str,
    format_: str = "mp3",
    instructions: Optional[str] = None,
) -> bytes:
    if not text:
        raise ValueError("Text for TTS is empty")

    timeout_seconds = _openai_tts_timeout_seconds()

    # Try streaming API first for best compatibility
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_}") as tmp:
        temp_path = Path(tmp.name)
    try:
        try:
            context = getattr(client.audio.speech, "with_streaming_response", None)
            if context is None:
                raise AttributeError
            with context.create(
                model=model,
                voice=voice,
                input=text,
                response_format=format_,
                instructions=instructions,
                timeout=timeout_seconds,
            ) as response:
                response.stream_to_file(str(temp_path))
                data = temp_path.read_bytes()
                return data
        except AttributeError:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=format_,
                instructions=instructions,
                timeout=timeout_seconds,
            )
            data = _extract_bytes(response)
            return data
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _synthesize_elevenlabs(
    *,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    voice_settings: Optional[Dict[str, Any]] = None,
    output_format: str = "mp3_44100_128",
    max_retries: int = 4,
    retry_backoff: float = 1.0,
    spoken_language: Optional[str] = None,
) -> bytes:
    if not api_key:
        raise ValueError("ElevenLabs API key is missing")
    if not voice:
        raise ValueError("ElevenLabs voice ID is missing")
    if not text:
        raise ValueError("Text for TTS is empty")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload: Dict[str, Any] = {
        "text": text,
        "model_id": model,
        "output_format": output_format,
    }
    if voice_settings:
        payload["voice_settings"] = voice_settings
    if spoken_language:
        payload["spoken_language"] = spoken_language

    attempt = 0
    delay = max(retry_backoff, 0.5)

    while attempt < max_retries:
        attempt += 1
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as err:  # pragma: no cover - network dependent
            if attempt >= max_retries:
                raise RuntimeError(f"ElevenLabs request failed (network): {err}") from err
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
            wait_s = delay
            if retry_after:
                try:
                    wait_s = float(retry_after)
                except ValueError:
                    pass
            if attempt >= max_retries:
                raise RuntimeError("ElevenLabs rate limit (429) after retries")
            time.sleep(max(wait_s, 0.5))
            delay = min(delay * 2, 10.0)
            continue

        if response.status_code in {500, 502, 503, 504} and attempt < max_retries:
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
            continue

        try:
            response.raise_for_status()
        except requests.RequestException as err:  # pragma: no cover - network dependent
            detail = response.text[:120] if response.text else str(err)
            raise RuntimeError(f"ElevenLabs request failed ({response.status_code}): {detail}") from err

        data = response.content
        if not data:
            if attempt >= max_retries:
                raise RuntimeError("ElevenLabs returned an empty audio payload")
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
            continue
        return data

    raise RuntimeError("ElevenLabs request failed after retries")


_NL_ALIASES: Set[str] = {
    "nl",
    "nl-nl",
    "nl_be",
    "nl-be",
    "nl_nl",
    "nld",
    "dut",
    "dutch",
    "nederlands",
    "flemish",
    "vlaams",
    "belgian-dutch",
    "netherlandic",
}


def _normalize_lang_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower()
    if not token:
        return ""
    token = token.replace("_", "-")
    token = token.replace("(", " ").replace(")", " ")
    token = "-".join(token.split())
    if token.startswith("nl-"):
        return "nl"
    if token in _NL_ALIASES:
        return "nl"
    if any(alias in token for alias in ("dutch", "nederlands", "flemish", "vlaams")):
        return "nl"
    return token


def _collect_language_tokens(voice: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    languages = voice.get("languages")
    if isinstance(languages, list):
        for entry in languages:
            if isinstance(entry, dict):
                tokens.append(_normalize_lang_token(entry.get("language_code")))
                tokens.append(_normalize_lang_token(entry.get("language")))
                tokens.append(_normalize_lang_token(entry.get("name")))
            else:
                tokens.append(_normalize_lang_token(entry))
    labels = voice.get("labels")
    if isinstance(labels, dict):
        for key, value in labels.items():
            tokens.append(_normalize_lang_token(key))
            tokens.append(_normalize_lang_token(value))
    tokens.append(_normalize_lang_token(voice.get("language")))
    tokens.append(_normalize_lang_token(voice.get("name")))
    tokens.append(_normalize_lang_token(voice.get("description")))
    tokens.append(_normalize_lang_token(voice.get("accent")))
    return [token for token in tokens if token]


_MULTILINGUAL_MARKERS: Set[str] = {
    "multilingual",
    "multi-language",
    "multi-lingual",
    "polyglot",
    "global",
    "universal",
}

_ENGLISH_ONLY_MARKERS: Set[str] = {
    "en",
    "eng",
    "en-us",
    "en-gb",
    "english-only",
    "us-english",
    "gb-english",
}

_LANGUAGE_TOKEN_HINTS: Set[str] = {
    "nl",
    "nederlands",
    "dutch",
    "flemish",
    "english",
    "en-",
    "german",
    "de",
    "deutsch",
    "spanish",
    "es",
    "italian",
    "it",
    "french",
    "fr",
    "portuguese",
    "pt",
    "polish",
    "pl",
    "turkish",
    "tr",
    "arabic",
    "ar",
    "persian",
    "farsi",
    "fa",
    "russian",
    "ru",
    "ukrainian",
    "uk",
    "spanish",
    "mexican",
    "brazilian",
    "swedish",
    "norwegian",
    "danish",
    "finnish",
    "czech",
    "slovak",
    "hungarian",
    "romanian",
    "serbian",
    "croatian",
    "bulgarian",
    "greek",
    "hebrew",
    "hindi",
    "urdu",
    "bengali",
    "thai",
    "vietnamese",
    "indonesian",
    "malay",
    "japanese",
    "korean",
    "mandarin",
    "cantonese",
    "chinese",
    "swahili",
    "afrikaans",
    "tigrinya",
    "amharic",
    "filipino",
    "tagalog",
    "multilingual",
    "polyglot",
    "global",
    "universal",
}


def _looks_like_language_token(token: str) -> bool:
    return any(hint in token for hint in _LANGUAGE_TOKEN_HINTS)


def _is_english_token(token: str) -> bool:
    if token in _ENGLISH_ONLY_MARKERS:
        return True
    return "english" in token or token.startswith("en-") or token.endswith("-english")


def _voice_supports_languages(
    voice: Dict[str, Any],
    desired: Set[str],
    *,
    model_is_multilingual: bool = False,
) -> bool:
    tokens = _collect_language_tokens(voice)

    if any(token in desired for token in tokens):
        return True

    languages_field = voice.get("languages")
    available_languages: Set[str] = set()
    if isinstance(languages_field, list):
        for entry in languages_field:
            if isinstance(entry, dict):
                available_languages.add(_normalize_lang_token(entry.get("language_code")))
                available_languages.add(_normalize_lang_token(entry.get("language")))
                available_languages.add(_normalize_lang_token(entry.get("name")))
            else:
                available_languages.add(_normalize_lang_token(entry))
    available_languages.discard("")

    if available_languages & desired:
        return True

    if not model_is_multilingual:
        return False

    if any(marker in token for token in tokens for marker in _MULTILINGUAL_MARKERS):
        return True

    language_tokens = [token for token in tokens if _looks_like_language_token(token)]
    if not language_tokens:
        return True

    return False


_LANGUAGE_LABELS: Dict[str, str] = {
    "nl": "Dutch",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "fa": "Persian",
    "ru": "Russian",
    "uk": "Ukrainian",
}


def _shared_voice_language_param(desired: Optional[Set[str]]) -> Optional[str]:
    if not desired:
        return None
    for code in desired:
        label = _LANGUAGE_LABELS.get(code)
        if label:
            return label
    return None


def _label_from_shared_voice(entry: Dict[str, Any]) -> str:
    name = str(entry.get("name") or "").strip()
    accent = str(entry.get("accent") or "").strip()
    descriptive = str(entry.get("descriptive") or "").strip()
    use_case = str(entry.get("use_case") or "").strip()
    language = str(entry.get("language") or "").strip()

    tag_candidates = [accent, descriptive, use_case, language]
    tag = next((candidate for candidate in tag_candidates if candidate and candidate.lower() not in {"standard", "neutral"}), "")
    if tag:
        return f"{name} — {tag}"
    return name


def fetch_elevenlabs_voice(
    api_key: str,
    voice_id: str,
    *,
    timeout: float = 20.0,
) -> Dict[str, str]:
    """Return metadata for one ElevenLabs voice ID available to the API key."""

    if not api_key:
        raise ValueError("ElevenLabs API key is required to fetch a voice")

    cleaned_voice_id = (voice_id or "").strip()
    if not cleaned_voice_id:
        raise ValueError("ElevenLabs voice ID is required")

    headers = {
        "xi-api-key": api_key,
        "Accept": "application/json",
    }
    try:
        response = requests.get(
            f"https://api.elevenlabs.io/v1/voices/{cleaned_voice_id}",
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as err:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to fetch ElevenLabs voice {cleaned_voice_id}: {err}") from err

    try:
        payload = response.json()
    except ValueError as err:  # pragma: no cover - network dependent
        raise RuntimeError("ElevenLabs voice response was not valid JSON") from err

    payload_voice_id = str(payload.get("voice_id") or cleaned_voice_id).strip()
    name = str(payload.get("name") or payload_voice_id).strip()
    if not payload_voice_id:
        raise RuntimeError("ElevenLabs voice response did not include a voice_id")

    return {"id": payload_voice_id, "label": name or payload_voice_id}


def fetch_elevenlabs_voices(
    api_key: str,
    *,
    language_codes: Optional[Sequence[str]] = None,
    timeout: float = 30.0,
) -> List[Dict[str, str]]:
    """Return ElevenLabs voices filtered by languages.

    Falls back to returning the full catalogue if the language filter yields
    no results so the UI can display at least the static defaults.
    """

    if not api_key:
        raise ValueError("ElevenLabs API key is required to fetch voices")

    headers = {
        "xi-api-key": api_key,
        "Accept": "application/json",
    }
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as err:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to list ElevenLabs voices: {err}") from err

    try:
        payload = response.json()
    except ValueError as err:  # pragma: no cover - network dependent
        raise RuntimeError("ElevenLabs voices response was not valid JSON") from err

    voices_raw = payload.get("voices")
    if not isinstance(voices_raw, list):
        return []

    desired: Optional[Set[str]] = None
    if language_codes:
        desired = {_normalize_lang_token(code) for code in language_codes if _normalize_lang_token(code)}
        if not desired:
            desired = None

    catalogue: List[Dict[str, str]] = []
    fallback_catalogue: List[Dict[str, str]] = []

    for entry in voices_raw:
        if not isinstance(entry, dict):
            continue
        voice_id = entry.get("voice_id")
        name = entry.get("name") or voice_id
        if not voice_id or not name:
            continue
        record = {"id": voice_id, "label": str(name)}
        if desired:
            if _voice_supports_languages(entry, desired, model_is_multilingual=True):
                catalogue.append(record)
            else:
                fallback_catalogue.append(record)
        else:
            catalogue.append(record)

    if desired:
        try:
            shared_params: Dict[str, Any] = {"page_size": 100}
            shared_language = _shared_voice_language_param(desired)
            if shared_language:
                shared_params["language"] = shared_language
            shared_response = requests.get(
                "https://api.elevenlabs.io/v1/shared-voices",
                headers=headers,
                params=shared_params,
                timeout=timeout,
            )
            shared_response.raise_for_status()
            shared_payload = shared_response.json()
            shared_raw = shared_payload.get("voices")
            if isinstance(shared_raw, list):
                for entry in shared_raw:
                    if not isinstance(entry, dict):
                        continue
                    voice_id = entry.get("voice_id")
                    name = entry.get("name") or voice_id
                    if not voice_id or not name:
                        continue
                    language_hints: Set[str] = {
                        _normalize_lang_token(entry.get("language")),
                        _normalize_lang_token(entry.get("accent")),
                        _normalize_lang_token(entry.get("locale")),
                    }
                    language_hints.discard("")
                    if desired and language_hints and not (language_hints & desired):
                        continue
                    if not _voice_supports_languages(entry, desired, model_is_multilingual=True):
                        continue
                    catalogue.append({"id": voice_id, "label": _label_from_shared_voice(entry)})
        except requests.RequestException:
            pass  # pragma: no cover - network dependent
        except ValueError:
            pass  # pragma: no cover - network dependent

    if not catalogue and desired:
        catalogue = fallback_catalogue

    seen: Set[str] = set()
    deduped: List[Dict[str, str]] = []
    for item in catalogue:
        voice_id = item.get("id")
        if not voice_id or voice_id in seen:
            continue
        seen.add(voice_id)
        deduped.append(item)

    return sorted(deduped, key=lambda item: item.get("label", "").lower())


def ensure_audio_for_cards(
    cards: Iterable[MutableMapping[str, str]],
    *,
    provider: str,
    voice: str,
    include_word: bool = True,
    include_sentence: bool = True,
    cache: Optional[MutableMapping[str, Tuple[str, bytes]]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    instruction_payloads: Optional[Dict[str, Any]] = None,
    instruction_keys: Optional[Dict[str, str]] = None,
    max_workers: int = 4,
    openai_client: Optional[OpenAI] = None,
    openai_model: Optional[str] = None,
    openai_fallback_model: Optional[str] = None,
    eleven_api_key: Optional[str] = None,
    eleven_model: Optional[str] = None,
) -> Tuple[Dict[str, bytes], AudioSynthesisSummary]:
    """Generate audio for cards, returning media map and summary.

    `cache` keeps `(filename, bytes)` per cache key so repeated texts reuse audio.
    """

    cache = cache if cache is not None else {}
    media: Dict[str, bytes] = {}
    provider_key = (provider or "openai").lower()
    summary = AudioSynthesisSummary(provider=provider_key, voice=voice)
    if instruction_keys:
        summary.sentence_instruction_key = instruction_keys.get("sentence", "")
        summary.word_instruction_key = instruction_keys.get("word", "")

    if provider_key == "openai":
        if openai_client is None:
            raise ValueError("OpenAI client must be provided for OpenAI TTS")
        primary_model = (openai_model or "").strip()
        if not primary_model:
            raise ValueError("OpenAI TTS model must be provided")
        models_order: List[str] = [primary_model]
        if openai_fallback_model:
            fallback_clean = openai_fallback_model.strip()
            if fallback_clean and fallback_clean not in models_order:
                models_order.append(fallback_clean)
    elif provider_key == "elevenlabs":
        if not eleven_api_key:
            raise ValueError("ElevenLabs API key must be provided for ElevenLabs TTS")
        effective_model = (eleven_model or "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
        models_order = [effective_model]
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")

    cards_list: List[MutableMapping[str, str]] = list(cards)
    for card in cards_list:
        card["AudioWord"] = ""
        card["AudioSentence"] = ""

    sentence_payload = instruction_payloads.get("sentence") if instruction_payloads else None
    word_payload = instruction_payloads.get("word") if instruction_payloads else None

    tasks: List[Tuple[int, str, str, Optional[Any]]] = []  # (card_index, kind, text, payload)
    for idx, card in enumerate(cards_list):
        woord = (card.get("L2_word") or "").strip()
        if include_word:
            if not woord:
                summary.word_skipped += 1
            else:
                tasks.append((idx, "word", woord, word_payload))
        sentence = sentence_for_tts(card.get("L2_cloze", "")) if include_sentence else ""
        if include_sentence:
            if not sentence:
                summary.sentence_skipped += 1
            else:
                tasks.append((idx, "sentence", sentence, sentence_payload))

    total_tasks = len(tasks)
    summary.total_requests = total_tasks
    if total_tasks == 0:
        if progress_cb:
            progress_cb(0, 0)
        return media, summary

    max_workers = max(1, min(int(max_workers or 1), total_tasks))
    if provider_key == "elevenlabs":
        max_workers = min(max_workers, 2)

    per_run_cache: Dict[str, Tuple[str, bytes]] = {}
    inflight: Dict[str, threading.Event] = {}
    cache_lock = threading.Lock()
    media_lock = threading.Lock()

    def _load_from_cache(key: str, filename_hint: Optional[str] = None) -> Optional[Tuple[str, bytes]]:
        with cache_lock:
            if key in per_run_cache:
                return per_run_cache[key]
            if key in cache:
                return cache[key]
        disk_bytes = _load_disk_cache(key)
        if disk_bytes is not None and filename_hint:
            value = (filename_hint, disk_bytes)
            with cache_lock:
                per_run_cache[key] = value
                cache[key] = value
            return value
        return None

    def _store_in_cache(key: str, value: Tuple[str, bytes]) -> None:
        with cache_lock:
            per_run_cache[key] = value
            cache[key] = value
        _store_disk_cache(key, value[1])

    def _claim_or_wait(key: str, filename_hint: str) -> Tuple[Optional[Tuple[str, bytes]], Optional[threading.Event], bool]:
        while True:
            cached = _load_from_cache(key, filename_hint)
            if cached is not None:
                return cached, None, False
            with cache_lock:
                event = inflight.get(key)
                if event is None:
                    event = threading.Event()
                    inflight[key] = event
                    return None, event, True
            event.wait()

    def _store_media(filename: str, data: bytes) -> None:
        with media_lock:
            media[filename] = data

    def _generate_single(
        kind: str, text: str, instruction_payload: Optional[Any]
    ) -> Tuple[str, bool, bool, Optional[str]]:
        last_error: Optional[Exception] = None
        fallback_flag = False
        for idx, model_id in enumerate(models_order):
            inflight_event: Optional[threading.Event] = None
            owns_event = False
            if provider_key == "openai":
                instruction_text: Optional[str] = None
                if isinstance(instruction_payload, dict):
                    raw_instr = instruction_payload.get("instructions")
                    if isinstance(raw_instr, str):
                        instruction_text = raw_instr.strip() or None
                elif isinstance(instruction_payload, str):
                    instruction_text = instruction_payload.strip() or None
                cache_token = _payload_token(instruction_text)
                cache_key = _cache_key(f"openai:{model_id}", voice, text, cache_token)
                filename_hint = _filename(kind, voice, text)
                cached, inflight_event, owns_event = _claim_or_wait(cache_key, filename_hint)
                cache_hit = False
                if cached is not None:
                    filename, data = cached
                    _store_media(filename, data)
                    cache_hit = True
                    return filename, (fallback_flag or idx > 0), cache_hit, model_id
                try:
                    data = _call_with_single_retry(
                        lambda: _synthesize(
                            openai_client,
                            model=model_id,
                            voice=voice,
                            text=text,
                            instructions=instruction_text,
                        )
                    )
                except Exception as err:  # pragma: no cover - network dependent
                    if owns_event and inflight_event is not None:
                        with cache_lock:
                            inflight.pop(cache_key, None)
                        inflight_event.set()
                    last_error = err
                    if idx < len(models_order) - 1 and _is_model_not_found(err):
                        fallback_flag = True
                        continue
                    raise
            else:  # ElevenLabs and other future providers following same pattern
                voice_settings: Optional[Dict[str, Any]] = None
                if isinstance(instruction_payload, dict):
                    vs = instruction_payload.get("voice_settings")
                    if isinstance(vs, dict):
                        voice_settings = vs
                spoken_language = None
                if isinstance(instruction_payload, dict):
                    sl = instruction_payload.get("spoken_language")
                    if isinstance(sl, str) and sl:
                        spoken_language = sl
                if not spoken_language:
                    spoken_language = "nl"
                token_source = {
                    "voice_settings": voice_settings,
                    "spoken_language": spoken_language,
                }
                cache_token = _payload_token(token_source)
                cache_key = _cache_key(f"elevenlabs:{model_id}", voice, text, cache_token)
                filename_hint = _filename(kind, voice, text)
                cached, inflight_event, owns_event = _claim_or_wait(cache_key, filename_hint)
                cache_hit = False
                if cached is not None:
                    filename, data = cached
                    _store_media(filename, data)
                    cache_hit = True
                    return filename, False, cache_hit, model_id
                try:
                    data = _call_with_single_retry(
                        lambda: _synthesize_elevenlabs(
                            api_key=eleven_api_key or "",
                            model=model_id,
                            voice=voice,
                            text=text,
                            voice_settings=voice_settings,
                            spoken_language=spoken_language,
                        )
                    )
                except Exception as err:  # pragma: no cover - network dependent
                    if owns_event and inflight_event is not None:
                        with cache_lock:
                            inflight.pop(cache_key, None)
                        inflight_event.set()
                    last_error = err
                    raise
            filename = _filename(kind, voice, text)
            _store_in_cache(cache_key, (filename, data))
            _store_media(filename, data)
            if owns_event and inflight_event is not None:
                with cache_lock:
                    inflight.pop(cache_key, None)
                inflight_event.set()
            return filename, (fallback_flag or idx > 0), False, model_id

        if last_error:
            raise last_error
        raise RuntimeError("Unable to synthesize audio")

    processed = 0

    def _worker(
        task: Tuple[int, str, str, Optional[Any]]
    ) -> Tuple[int, str, Optional[str], bool, bool, Optional[Exception], Optional[str]]:
        card_idx, kind, text, instruction_payload = task
        try:
            filename, used_fallback, cache_hit, model_used = _generate_single(kind, text, instruction_payload)
            return card_idx, kind, filename, used_fallback, cache_hit, None, model_used
        except Exception as err:  # pragma: no cover - network dependent
            return card_idx, kind, None, False, False, err, None

    if progress_cb:
        progress_cb(0, total_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            card_idx, kind, text, _payload = task
            card_idx_res, kind_res, filename, used_fallback, cache_hit, error, model_used = future.result()
            # Sanity: ensure indices align
            card_idx = card_idx_res
            kind = kind_res
            card = cards_list[card_idx]
            if error or not filename:
                error_text = str(error) if error else "Audio synthesis failed"
                summary.clip_results.append(
                    AudioClipResult(
                        card_index=card_idx,
                        kind=kind,
                        text=text,
                        status="failed",
                        filename=filename or "",
                        error=error_text,
                        model=model_used or "",
                        fallback_used=used_fallback,
                    )
                )
                if kind == "word":
                    summary.word_skipped += 1
                    label = (card.get("L2_word") or "").strip()
                    summary.errors.append(f"{label or 'word'}: {error_text}")
                else:
                    summary.sentence_skipped += 1
                    label = (card.get("L2_word") or "").strip()
                    summary.errors.append(f"Sentence for '{label}': {error_text}")
            else:
                clip_status = "cached" if cache_hit else "ok"
                summary.clip_results.append(
                    AudioClipResult(
                        card_index=card_idx,
                        kind=kind,
                        text=text,
                        status=clip_status,
                        filename=filename,
                        error="",
                        model=model_used or "",
                        fallback_used=used_fallback,
                    )
                )
                if kind == "word":
                    card["AudioWord"] = f"[sound:{filename}]"
                    summary.word_success += 1
                else:
                    card["AudioSentence"] = f"[sound:{filename}]"
                    summary.sentence_success += 1
                if cache_hit:
                    summary.cache_hits += 1
                if used_fallback:
                    summary.fallback_switches += 1
                if not cache_hit and model_used:
                    summary.register_usage(
                        model=model_used,
                        kind=kind,
                        char_count=len(text),
                        fallback_used=used_fallback,
                    )

            processed += 1
            if progress_cb:
                progress_cb(processed, total_tasks)

    if progress_cb and processed < total_tasks:
        progress_cb(total_tasks, total_tasks)

    return media, summary
