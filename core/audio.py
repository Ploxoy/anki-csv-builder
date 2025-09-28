"""Utilities for post-generation text-to-speech (TTS) synthesis."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import requests

__all__ = [
    "AudioSynthesisSummary",
    "ensure_audio_for_cards",
    "fetch_elevenlabs_voices",
    "sentence_for_tts",
]


_CLOZE_RE = re.compile(r"\{\{c[12]::(.*?)(?:::[^}]*)?\}\}")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class AudioSynthesisSummary:
    """Aggregated statistics about a TTS run."""

    total_requests: int = 0
    cache_hits: int = 0
    word_success: int = 0
    sentence_success: int = 0
    word_skipped: int = 0
    sentence_skipped: int = 0
    errors: List[str] = None  # type: ignore[assignment]
    fallback_switches: int = 0
    sentence_instruction_key: str = ""
    word_instruction_key: str = ""
    provider: str = ""
    voice: str = ""

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


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


def _is_model_not_found(error: Exception) -> bool:
    message = str(error).lower()
    return "model_not_found" in message or "does not exist" in message or "404" in message


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


def _normalize_lang_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower()
    if not token:
        return ""
    if token in {"nl", "nld", "nl_nl"}:
        return "nl"
    if token.startswith("nl-"):
        return "nl"
    if "dutch" in token or "flemish" in token:
        return "nl"
    return token


def _voice_supports_languages(voice: Dict[str, Any], desired: Set[str]) -> bool:
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
        for value in labels.values():
            tokens.append(_normalize_lang_token(value))
    tokens.append(_normalize_lang_token(voice.get("language")))
    tokens.append(_normalize_lang_token(voice.get("name")))
    for token in tokens:
        if token and token in desired:
            return True
    return False


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
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=timeout)
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
            if _voice_supports_languages(entry, desired):
                catalogue.append(record)
            else:
                fallback_catalogue.append(record)
        else:
            catalogue.append(record)

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
    cache_lock = threading.Lock()
    media_lock = threading.Lock()

    def _load_from_cache(key: str) -> Optional[Tuple[str, bytes]]:
        with cache_lock:
            if key in per_run_cache:
                return per_run_cache[key]
            if key in cache:
                return cache[key]
        return None

    def _store_in_cache(key: str, value: Tuple[str, bytes]) -> None:
        with cache_lock:
            per_run_cache[key] = value
            cache[key] = value

    def _store_media(filename: str, data: bytes) -> None:
        with media_lock:
            media[filename] = data

    def _generate_single(kind: str, text: str, instruction_payload: Optional[Any]) -> Tuple[str, bool, bool]:
        last_error: Optional[Exception] = None
        fallback_flag = False
        for idx, model_id in enumerate(models_order):
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
                cached = _load_from_cache(cache_key)
                cache_hit = False
                if cached is not None:
                    filename, data = cached
                    _store_media(filename, data)
                    cache_hit = True
                    return filename, (fallback_flag or idx > 0), cache_hit
                try:
                    data = _synthesize(
                        openai_client,
                        model=model_id,
                        voice=voice,
                        text=text,
                        instructions=instruction_text,
                    )
                except Exception as err:  # pragma: no cover - network dependent
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
                cached = _load_from_cache(cache_key)
                cache_hit = False
                if cached is not None:
                    filename, data = cached
                    _store_media(filename, data)
                    cache_hit = True
                    return filename, False, cache_hit
                try:
                    data = _synthesize_elevenlabs(
                        api_key=eleven_api_key or "",
                        model=model_id,
                        voice=voice,
                        text=text,
                        voice_settings=voice_settings,
                        spoken_language=spoken_language,
                    )
                except Exception as err:  # pragma: no cover - network dependent
                    last_error = err
                    raise
            filename = _filename(kind, voice, text)
            _store_in_cache(cache_key, (filename, data))
            _store_media(filename, data)
            return filename, (fallback_flag or idx > 0), False

        if last_error:
            raise last_error
        raise RuntimeError("Unable to synthesize audio")

    processed = 0

    def _worker(task: Tuple[int, str, str, Optional[Any]]) -> Tuple[int, str, Optional[str], bool, bool, Optional[Exception]]:
        card_idx, kind, text, instruction_payload = task
        try:
            filename, used_fallback, cache_hit = _generate_single(kind, text, instruction_payload)
            return card_idx, kind, filename, used_fallback, cache_hit, None
        except Exception as err:  # pragma: no cover - network dependent
            return card_idx, kind, None, False, False, err

    if progress_cb:
        progress_cb(0, total_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(futures):
            card_idx, kind, filename, used_fallback, cache_hit, error = future.result()
            card = cards_list[card_idx]
            if error or not filename:
                if kind == "word":
                    summary.word_skipped += 1
                    label = (card.get("L2_word") or "").strip()
                    summary.errors.append(f"{label or 'word'}: {error}") if error else None
                else:
                    summary.sentence_skipped += 1
                    label = (card.get("L2_word") or "").strip()
                    summary.errors.append(
                        f"Sentence for '{label}': {error}"
                    ) if error else None
            else:
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

            processed += 1
            if progress_cb:
                progress_cb(processed, total_tasks)

    if progress_cb and processed < total_tasks:
        progress_cb(total_tasks, total_tasks)

    return media, summary
