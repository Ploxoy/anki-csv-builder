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

    shared_payload = {
        "voices": [
            {
                "voice_id": "S1",
                "name": "Shared Voice",
                "language": "Dutch",
                "accent": "Standard",
            }
        ]
    }

    def fake_get(url: str, headers: Dict[str, str], timeout: float, params: Dict[str, Any] | None = None):  # type: ignore[override]
        assert "xi-api-key" in headers
        if url.endswith("/voices"):
            return _DummyResponse(payload)
        assert url.endswith("/shared-voices")
        assert params is not None
        return _DummyResponse(shared_payload)

    monkeypatch.setattr(audio.requests, "get", fake_get)

    voices = audio.fetch_elevenlabs_voices("secret-key", language_codes=["nl"])

    assert voices == [
        {"id": "A", "label": "Anne"},
        {"id": "B", "label": "Ben"},
        {"id": "S1", "label": "Shared Voice — Dutch"},
    ]


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


def test_ensure_audio_for_cards_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [{"L2_word": "woord", "L2_cloze": "Dit {{c1::woord}}."}]

    calls: list[str] = []

    def fake_synthesize(*_, model: str, **__):
        calls.append(model)
        if model == "primary-tts":
            raise RuntimeError("model_not_found: primary")
        return b"audio-data"

    monkeypatch.setattr(audio, "_synthesize", fake_synthesize)

    media_map, summary = audio.ensure_audio_for_cards(
        cards,
        provider="openai",
        voice="test-voice",
        include_word=True,
        include_sentence=False,
        cache={},
        progress_cb=None,
        instruction_payloads=None,
        instruction_keys=None,
        max_workers=1,
        openai_client=object(),
        openai_model="primary-tts",
        openai_fallback_model="fallback-tts",
    )

    assert summary.word_success == 1
    assert summary.fallback_switches == 1
    assert summary.cache_hits == 0
    assert calls == ["primary-tts", "fallback-tts"]
    assert cards[0]["AudioWord"].startswith("[sound:")
    assert media_map


def test_ensure_audio_cache_hits_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: dict[str, tuple[str, bytes]] = {}
    cards = [{"L2_word": "woord", "L2_cloze": "Dit {{c1::woord}}."}]

    monkeypatch.setattr(audio, "_synthesize", lambda *_, **__: b"audio-data")

    audio.ensure_audio_for_cards(
        cards,
        provider="openai",
        voice="test-voice",
        include_word=True,
        include_sentence=False,
        cache=cache,
        progress_cb=None,
        instruction_payloads=None,
        instruction_keys=None,
        max_workers=1,
        openai_client=object(),
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    cards_again = [{"L2_word": "woord", "L2_cloze": "Dit {{c1::woord}}."}]
    _, summary_second = audio.ensure_audio_for_cards(
        cards_again,
        provider="openai",
        voice="test-voice",
        include_word=True,
        include_sentence=False,
        cache=cache,
        progress_cb=None,
        instruction_payloads=None,
        instruction_keys=None,
        max_workers=1,
        openai_client=object(),
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    assert summary_second.cache_hits == 1
    assert summary_second.word_success == 1
