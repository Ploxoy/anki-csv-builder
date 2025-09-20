"""Utilities for post-generation text-to-speech (TTS) synthesis."""
from __future__ import annotations

import base64
import hashlib
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

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


def _cache_key(model: str, voice: str, text: str) -> str:
    digest = hashlib.sha256(f"{model}|{voice}|{text}".encode("utf-8")).hexdigest()
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
            with context.create(model=model, voice=voice, input=text) as response:
                response.stream_to_file(str(temp_path))
                data = temp_path.read_bytes()
                return data
        except AttributeError:
            response = client.audio.speech.create(model=model, voice=voice, input=text)
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
) -> Tuple[Dict[str, bytes], AudioSynthesisSummary]:
    """Generate audio for cards, returning media map and summary.

    `cache` keeps `(filename, bytes)` per cache key so repeated texts reuse audio.
    """

    cache = cache if cache is not None else {}
    media: Dict[str, bytes] = {}
    summary = AudioSynthesisSummary()

    models_order: List[str] = [model]
    if fallback_model and fallback_model not in models_order:
        models_order.append(fallback_model)

    # Pre-compute total number of requests (word + sentence per card if applicable)
    tasks: List[Tuple[str, str, str]] = []  # (kind, key, text)
    for card in cards:
        woord = (card.get("L2_word") or "").strip()
        if include_word and woord:
            tasks.append(("word", "L2_word", woord))
        sentence = sentence_for_tts(card.get("L2_cloze", ""))
        if include_sentence and sentence:
            tasks.append(("sentence", "L2_cloze", sentence))
    total_tasks = len(tasks)
    if total_tasks == 0:
        return media, summary

    if progress_cb:
        progress_cb(0, total_tasks)

    processed = 0
    per_run_cache: Dict[str, Tuple[str, bytes]] = {}

    cards_list = list(cards)
    for card in cards_list:
        # Reset fields before generation to avoid stale placeholders
        card["AudioWord"] = ""
        card["AudioSentence"] = ""

    def _generate_audio(kind: str, text: str, card_label: str) -> Tuple[Optional[str], bool]:
        nonlocal summary
        last_error: Optional[Exception] = None
        fallback_flag = False
        for idx, model_id in enumerate(models_order):
            key = _cache_key(model_id, voice, text)
            filename: Optional[str] = None
            data: Optional[bytes] = None
            fallback_context = fallback_flag or idx > 0

            if key in cache:
                summary.cache_hits += 1
                filename, data = cache[key]
            elif key in per_run_cache:
                filename, data = per_run_cache[key]
            else:
                try:
                    data = _synthesize(client, model=model_id, voice=voice, text=text)
                except Exception as err:  # pragma: no cover - network dependent
                    last_error = err
                    if idx < len(models_order) - 1 and _is_model_not_found(err):
                        fallback_flag = True
                        continue
                    raise
                filename = _filename(kind, voice, text)
                per_run_cache[key] = (filename, data)
                cache[key] = (filename, data)

            if data and filename:
                media[filename] = data
                return filename, fallback_context

        if last_error:
            raise last_error
        return None, fallback_flag

    for card in cards_list:
        woord = (card.get("L2_word") or "").strip()
        if include_word:
            if not woord:
                summary.word_skipped += 1
            else:
                text_word = woord
                try:
                    filename, used_fallback = _generate_audio("word", text_word, woord)
                    if filename:
                        card["AudioWord"] = f"[sound:{filename}]"
                        summary.word_success += 1
                        if used_fallback:
                            summary.fallback_switches += 1
                    else:
                        summary.word_skipped += 1
                except Exception as err:  # pragma: no cover - network dependent
                    summary.errors.append(f"{woord}: {err}")
                    summary.word_skipped += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total_tasks)

        sentence_clean = sentence_for_tts(card.get("L2_cloze", "")) if include_sentence else ""
        if include_sentence:
            if not sentence_clean:
                summary.sentence_skipped += 1
            else:
                try:
                    filename, used_fallback = _generate_audio("sentence", sentence_clean, card.get("L2_word", ""))
                    if filename:
                        card["AudioSentence"] = f"[sound:{filename}]"
                        summary.sentence_success += 1
                        if used_fallback:
                            summary.fallback_switches += 1
                    else:
                        summary.sentence_skipped += 1
                except Exception as err:  # pragma: no cover - network dependent
                    summary.errors.append(
                        f"Sentence for '{card.get('L2_word','')}': {err}"
                    )
                    summary.sentence_skipped += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total_tasks)

    summary.total_requests = total_tasks
    if progress_cb:
        progress_cb(total_tasks, total_tasks)
    return media, summary
