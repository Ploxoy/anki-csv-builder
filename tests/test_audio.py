"""Tests covering audio catalogue and orchestration helpers."""
from __future__ import annotations

from typing import Any, Dict
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core import audio


class _DummyResponse:
    def __init__(self, payload: Any):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - matches requests API
        return

    def json(self) -> Any:
        return self._payload


class _DummyErrorResponse:
    def __init__(self, message: str):
        self._message = message
        self.status_code = 400

    def raise_for_status(self) -> None:
        raise audio.requests.HTTPError(self._message)

    def json(self) -> Dict[str, Any]:  # pragma: no cover - should not be used
        return {}


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


def test_fetch_elevenlabs_voice_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_get(url: str, headers: Dict[str, str], timeout: float):  # type: ignore[override]
        seen["url"] = url
        seen["headers"] = headers
        seen["timeout"] = timeout
        return _DummyResponse({"voice_id": "voice-123", "name": "Library Voice"})

    monkeypatch.setattr(audio.requests, "get", fake_get)

    voice = audio.fetch_elevenlabs_voice("secret-key", "voice-123", timeout=7)

    assert seen["url"] == "https://api.elevenlabs.io/v1/voices/voice-123"
    assert seen["headers"]["xi-api-key"] == "secret-key"
    assert seen["timeout"] == 7
    assert voice == {"id": "voice-123", "label": "Library Voice"}


def test_fetch_elevenlabs_voice_falls_back_to_v2_voice_search(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, Any]] = []

    def fake_get(url: str, headers: Dict[str, str], timeout: float, params: Any = None):  # type: ignore[override]
        seen.append((url, params))
        if url.endswith("/v1/voices/library-voice"):
            return _DummyErrorResponse("400 Bad Request")
        assert url == "https://api.elevenlabs.io/v2/voices"
        return _DummyResponse({"voices": [{"voice_id": "library-voice", "name": "Saved Library Voice"}]})

    monkeypatch.setattr(audio.requests, "get", fake_get)

    voice = audio.fetch_elevenlabs_voice("secret-key", "library-voice")

    assert seen[0][0] == "https://api.elevenlabs.io/v1/voices/library-voice"
    assert seen[1][0] == "https://api.elevenlabs.io/v2/voices"
    assert ("voice_ids", "library-voice") in seen[1][1]
    assert voice == {"id": "library-voice", "label": "Saved Library Voice"}


def test_fetch_elevenlabs_models_returns_tts_models_sorted_by_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "model_id": "eleven_v3",
            "can_do_text_to_speech": True,
            "model_rates": {"character_cost_multiplier": 2},
        },
        {
            "model_id": "eleven_flash_v2_5",
            "can_do_text_to_speech": True,
            "model_rates": {"character_cost_multiplier": 0.5},
        },
        {
            "model_id": "scribe_v1",
            "can_do_text_to_speech": False,
            "model_rates": {"character_cost_multiplier": 0.1},
        },
    ]

    def fake_get(url: str, headers: Dict[str, str], timeout: float):  # type: ignore[override]
        assert url == "https://api.elevenlabs.io/v1/models"
        assert headers["xi-api-key"] == "secret-key"
        return _DummyResponse(payload)  # type: ignore[arg-type]

    monkeypatch.setattr(audio.requests, "get", fake_get)

    assert audio.fetch_elevenlabs_models("secret-key") == ["eleven_flash_v2_5", "eleven_v3"]


def _set_temp_audio_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cache_dir = tmp_path / "audio-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(audio, "AUDIO_CACHE_DIR", cache_dir, raising=False)


def test_ensure_audio_for_cards_reuses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)
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


def test_ensure_audio_for_cards_uses_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)
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


def test_ensure_audio_cache_hits_across_runs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)
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


def test_disk_cache_survives_session(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)

    cards = [{"L2_word": "woord", "L2_cloze": "Dit {{c1::woord}}."}]

    synth_calls: list[str] = []

    def fake_synthesize(*_, **__):
        synth_calls.append("call")
        return b"audio-data"

    monkeypatch.setattr(audio, "_synthesize", fake_synthesize)

    audio.ensure_audio_for_cards(
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
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    # Second run should hit disk cache even with a fresh in-memory cache
    def fail_synthesize(*_, **__):  # pragma: no cover - ensures we don't call synth again
        raise AssertionError("synthesize should not be called when disk cache is used")

    monkeypatch.setattr(audio, "_synthesize", fail_synthesize)

    _, summary_second = audio.ensure_audio_for_cards(
        [{"L2_word": "woord", "L2_cloze": "Dit {{c1::woord}}."}],
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
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    assert summary_second.cache_hits == 1
    assert synth_calls  # initial run executed


def test_ensure_audio_retries_once_on_transient_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)

    calls = {"count": 0}

    def flaky_synthesize(*_, **__) -> bytes:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return b"audio-data"

    monkeypatch.setattr(audio, "_synthesize", flaky_synthesize)

    cards = [{"L2_word": "fiets", "L2_cloze": ""}]
    _, summary = audio.ensure_audio_for_cards(
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
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    assert calls["count"] == 2
    assert summary.word_success == 1
    assert summary.errors == []
    assert summary.clip_results[0].status == "ok"


def test_ensure_audio_does_not_retry_on_validation_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)

    calls = {"count": 0}

    def invalid_synthesize(*_, **__) -> bytes:
        calls["count"] += 1
        raise ValueError("voice is invalid")

    monkeypatch.setattr(audio, "_synthesize", invalid_synthesize)

    cards = [{"L2_word": "fiets", "L2_cloze": ""}]
    _, summary = audio.ensure_audio_for_cards(
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
        openai_model="gpt-tts",
        openai_fallback_model=None,
    )

    assert calls["count"] == 1
    assert summary.word_success == 0
    assert summary.clip_results[0].status == "failed"
    assert "voice is invalid" in summary.clip_results[0].error


def test_ensure_audio_elevenlabs_cache_hit_keeps_clip_status(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_temp_audio_cache(monkeypatch, tmp_path)
    cache: dict[str, tuple[str, bytes]] = {}

    calls = {"count": 0}

    def fake_elevenlabs(*_, **__) -> bytes:
        calls["count"] += 1
        return b"eleven-data"

    monkeypatch.setattr(audio, "_synthesize_elevenlabs", fake_elevenlabs)

    cards_first = [{"L2_word": "aanraken", "L2_cloze": ""}]
    audio.ensure_audio_for_cards(
        cards_first,
        provider="elevenlabs",
        voice="voice-eleven",
        include_word=True,
        include_sentence=False,
        cache=cache,
        progress_cb=None,
        instruction_payloads=None,
        instruction_keys=None,
        max_workers=1,
        eleven_api_key="secret",
        eleven_model="eleven_multilingual_v2",
    )

    def fail_elevenlabs(*_, **__) -> bytes:  # pragma: no cover - should never run on cache hit
        raise AssertionError("ElevenLabs synth should not run on cache hit")

    monkeypatch.setattr(audio, "_synthesize_elevenlabs", fail_elevenlabs)

    cards_second = [{"L2_word": "aanraken", "L2_cloze": ""}]
    _, summary_second = audio.ensure_audio_for_cards(
        cards_second,
        provider="elevenlabs",
        voice="voice-eleven",
        include_word=True,
        include_sentence=False,
        cache=cache,
        progress_cb=None,
        instruction_payloads=None,
        instruction_keys=None,
        max_workers=1,
        eleven_api_key="secret",
        eleven_model="eleven_multilingual_v2",
    )

    assert calls["count"] == 1
    assert summary_second.cache_hits == 1
    assert summary_second.word_success == 1
    assert summary_second.clip_results[0].status == "cached"
