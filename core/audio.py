"""Utilities for post-generation text-to-speech (TTS) synthesis."""
from __future__ import annotations

import base64
import hashlib
import re
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

__all__ = [
    "AudioSynthesisSummary",
    "ensure_audio_for_cards",
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


def ensure_audio_for_cards(
    cards: Iterable[MutableMapping[str, str]],
    *,
    client: OpenAI,
    model: str,
    voice: str,
    fallback_model: Optional[str] = None,
    include_word: bool = True,
    include_sentence: bool = True,
    cache: Optional[MutableMapping[str, Tuple[str, bytes]]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    instructions: Optional[Dict[str, str]] = None,
    instruction_keys: Optional[Dict[str, str]] = None,
    max_workers: int = 4,
) -> Tuple[Dict[str, bytes], AudioSynthesisSummary]:
    """Generate audio for cards, returning media map and summary.

    `cache` keeps `(filename, bytes)` per cache key so repeated texts reuse audio.
    """

    cache = cache if cache is not None else {}
    media: Dict[str, bytes] = {}
    summary = AudioSynthesisSummary()
    if instruction_keys:
        summary.sentence_instruction_key = instruction_keys.get("sentence", "")
        summary.word_instruction_key = instruction_keys.get("word", "")

    models_order: List[str] = [model]
    if fallback_model and fallback_model not in models_order:
        models_order.append(fallback_model)

    cards_list: List[MutableMapping[str, str]] = list(cards)
    for card in cards_list:
        card["AudioWord"] = ""
        card["AudioSentence"] = ""

    sanitized_instructions: Dict[str, Optional[str]] = {}
    if instructions:
        for key, value in instructions.items():
            sanitized_instructions[key] = (value.strip() or None) if isinstance(value, str) else None

    tasks: List[Tuple[int, str, str, Optional[str]]] = []  # (card_index, kind, text, instruction)
    for idx, card in enumerate(cards_list):
        woord = (card.get("L2_word") or "").strip()
        if include_word:
            if not woord:
                summary.word_skipped += 1
            else:
                tasks.append((idx, "word", woord, sanitized_instructions.get("word") if instructions else None))
        sentence = sentence_for_tts(card.get("L2_cloze", "")) if include_sentence else ""
        if include_sentence:
            if not sentence:
                summary.sentence_skipped += 1
            else:
                tasks.append((idx, "sentence", sentence, sanitized_instructions.get("sentence") if instructions else None))

    total_tasks = len(tasks)
    summary.total_requests = total_tasks
    if total_tasks == 0:
        if progress_cb:
            progress_cb(0, 0)
        return media, summary

    max_workers = max(1, min(int(max_workers or 1), total_tasks))

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

    def _generate_single(kind: str, text: str, instruction_text: Optional[str]) -> Tuple[str, bool, bool]:
        last_error: Optional[Exception] = None
        fallback_flag = False
        for idx, model_id in enumerate(models_order):
            cache_key = _cache_key(model_id, voice, text, instruction_text)
            cached = _load_from_cache(cache_key)
            cache_hit = False
            if cached is not None:
                filename, data = cached
                _store_media(filename, data)
                cache_hit = True
                return filename, (fallback_flag or idx > 0), cache_hit
            try:
                data = _synthesize(
                    client,
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
            filename = _filename(kind, voice, text)
            _store_in_cache(cache_key, (filename, data))
            _store_media(filename, data)
            return filename, (fallback_flag or idx > 0), False

        if last_error:
            raise last_error
        raise RuntimeError("Unable to synthesize audio")

    processed = 0

    def _worker(task: Tuple[int, str, str, Optional[str]]) -> Tuple[int, str, Optional[str], bool, bool, Optional[Exception]]:
        card_idx, kind, text, instruction_text = task
        try:
            filename, used_fallback, cache_hit = _generate_single(kind, text, instruction_text)
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
