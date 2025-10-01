"""Tests covering audio catalogue and orchestration helpers."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from core import audio


class _DummyResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - matches requests API
        return

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_fetch_elevenlabs_voices_filters_and_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "voices": [
            {
                "voice_id": "A",
                "name": "Anne",
                "languages": [{"language_code": "nl"}],
            },
            {
                "voice_id": "B",
                "name": "Ben",
                "labels": {"accent": "Flemish"},
            },
            {
                "voice_id": "C",
                "name": "Carl",
                "languages": [{"language_code": "de"}],
            },
            {
                "voice_id": "A",
                "name": "Anne Duplicate",
                "languages": [{"language_code": "nl"}],
            },
        ]
    }

    def fake_get(url: str, headers: Dict[str, str], timeout: float):  # type: ignore[override]
        assert "xi-api-key" in headers
        assert url.endswith("/voices")
        return _DummyResponse(payload)

    monkeypatch.setattr(audio.requests, "get", fake_get)

    voices = audio.fetch_elevenlabs_voices("secret-key", language_codes=["nl"])

    assert voices == [
        {"id": "A", "label": "Anne"},
        {"id": "B", "label": "Ben"},
    ]

    ids = {voice["id"] for voice in voices}
    assert ids == {"A", "B"}


def test_ensure_audio_for_cards_reuses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [
        {"L2_word": "fiets", "L2_cloze": "{{c1::fiets}} is nieuw"},
        {"L2_word": "fiets", "L2_cloze": "{{c1::fiets}} is nieuw"},
    ]

    calls: list[str] = []

    def fake_synthesize(*args: Any, **kwargs: Any) -> bytes:
        # args: (client, model, voice, text, instructions)
        text = kwargs.get("text") if "text" in kwargs else args[3]
        calls.append(text)
        return f"audio::{text}".encode("utf-8")

    monkeypatch.setattr(audio, "_synthesize", fake_synthesize)

    progress_steps: list[tuple[int, int]] = []

    def progress_cb(current: int, total: int) -> None:
        progress_steps.append((current, total))

    media_map, summary = audio.ensure_audio_for_cards(
        cards,
        provider="openai",
        voice="voice-nl",
        include_word=True,
        include_sentence=True,
        cache={},
        progress_cb=progress_cb,
        instruction_payloads={"sentence": {"instructions": "kort"}},
        instruction_keys={"sentence": "Dutch_sentence_default", "word": "Dutch_word_default"},
        max_workers=4,
        openai_client=object(),
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    assert len(calls) == 2  # initial word + sentence; duplicates served from cache
    assert summary.total_requests == 4
    assert summary.word_success == 2
    assert summary.sentence_success == 2
    assert summary.cache_hits == 2
    assert cards[0]["AudioWord"].startswith("[sound:")
    assert cards[0]["AudioSentence"].startswith("[sound:")
    assert media_map
    assert progress_steps[-1] == (4, 4)
