from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import api.main as api_main
from core.api_schemas import TTSPreviewRequest, TTSRequest, TTSVoiceCheckRequest
from core.audio import AudioClipResult, AudioSynthesisSummary
from core import audio as audio_core


def _dummy_request() -> Request:
    return Request({"type": "http", "headers": []})


@pytest.fixture
def patch_api_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "_require_user", lambda request, x_api_key: "user-test")
    monkeypatch.setattr(api_main, "_openai_client_or_500", lambda: object())
    monkeypatch.setattr(api_main, "log_usage_events", lambda **kwargs: None)
    monkeypatch.setattr(api_main, "store_run_media_assets", lambda **kwargs: (len(kwargs.get("media_files") or {}), None))
    monkeypatch.setattr(api_main, "load_audio_assets", lambda **kwargs: ({}, None))
    monkeypatch.setattr(api_main, "store_audio_assets", lambda **kwargs: (len(list(kwargs.get("assets") or [])), None))
    monkeypatch.setattr(api_main, "touch_audio_assets", lambda **kwargs: None)


def test_api_tts_returns_clip_level_status_and_error(monkeypatch: pytest.MonkeyPatch, patch_api_auth: None) -> None:
    def fake_ensure(cards: list[dict[str, Any]], **kwargs: Any):
        cards[0]["AudioWord"] = "[sound:word_fiets.mp3]"
        summary = AudioSynthesisSummary(provider="openai", voice="alloy")
        summary.word_success = 1
        summary.sentence_success = 0
        summary.cache_hits = 0
        summary.total_characters = 20
        summary.errors = ["Sentence for 'fiets': upstream timeout"]
        summary.clip_results = [
            AudioClipResult(
                card_index=0,
                kind="word",
                text="fiets",
                status="ok",
                filename="word_fiets.mp3",
                model="gpt-4o-tts",
            ),
            AudioClipResult(
                card_index=1,
                kind="sentence",
                text="Dit is een fiets.",
                status="failed",
                error="upstream timeout",
                model="gpt-4o-tts",
            ),
        ]
        return {"word_fiets.mp3": b"abc"}, summary

    monkeypatch.setattr(api_main, "ensure_audio_for_cards", fake_ensure)

    payload = TTSRequest(
        run_id="run-1",
        provider="openai",
        model="gpt-4o-tts",
        voice="alloy",
        items=[
            {"card_id": "row-1", "type": "word", "text": "fiets"},
            {"card_id": "row-1", "type": "sentence", "text": "Dit is een fiets."},
        ],
    )

    result = api_main.api_tts(payload, request=_dummy_request(), x_api_key=None)

    assert result.summary.ok == 1
    assert result.summary.failed == 1
    assert result.summary.cached == 0
    assert result.audios[0].status == "ok"
    assert result.audios[0].filename == "word_fiets.mp3"
    assert result.audios[0].audio_b64 is not None
    assert result.audios[1].status == "failed"
    assert result.audios[1].error == "upstream timeout"
    assert result.audios[1].filename is None
    assert result.storage is not None
    assert result.storage.persisted is True
    assert result.storage.stored_clips == 1


def test_api_tts_voice_check_validates_elevenlabs_voice(
    monkeypatch: pytest.MonkeyPatch, patch_api_auth: None
) -> None:
    seen: dict[str, str] = {}

    monkeypatch.setattr(api_main, "_elevenlabs_key_or_500", lambda: "eleven-secret")

    def fake_fetch(api_key: str, voice_id: str):
        seen["api_key"] = api_key
        seen["voice_id"] = voice_id
        return {"id": voice_id, "label": "Library Voice"}

    monkeypatch.setattr(api_main, "fetch_elevenlabs_voice", fake_fetch)

    payload = TTSVoiceCheckRequest(provider="elevenlabs", voice_id="voice-123")

    result = api_main.api_tts_voice_check(payload, request=_dummy_request(), x_api_key=None)

    assert seen == {"api_key": "eleven-secret", "voice_id": "voice-123"}
    assert result.provider == "elevenlabs"
    assert result.id == "voice-123"
    assert result.label == "Library Voice"
    assert result.valid is True


def test_api_tts_preview_returns_audio_without_storage(
    monkeypatch: pytest.MonkeyPatch, patch_api_auth: None
) -> None:
    logged: list[dict[str, Any]] = []

    def fake_log_usage_events(**kwargs: Any) -> None:
        logged.extend(kwargs.get("events") or [])

    def fake_ensure(cards: list[dict[str, Any]], **kwargs: Any):
        assert kwargs["provider"] == "openai"
        assert kwargs["voice"] == "alloy"
        cards[0]["AudioSentence"] = "[sound:sentence_preview.mp3]"
        summary = AudioSynthesisSummary(provider="openai", voice="alloy")
        summary.sentence_success = 1
        summary.total_characters = 32
        summary.total_requests = 1
        summary.total_requests_billed = 1
        summary.model_usage = {"gpt-4o-tts": {"chars": 32, "requests": 1}}
        summary.clip_results = [
            AudioClipResult(
                card_index=0,
                kind="sentence",
                text="Dit is een voorbeeld.",
                status="ok",
                filename="sentence_preview.mp3",
                model="gpt-4o-tts",
            )
        ]
        return {"sentence_preview.mp3": b"preview-audio"}, summary

    monkeypatch.setattr(api_main, "ensure_audio_for_cards", fake_ensure)
    monkeypatch.setattr(api_main, "log_usage_events", fake_log_usage_events)

    payload = TTSPreviewRequest(
        provider="openai",
        model="gpt-4o-tts",
        voice="alloy",
        text="Dit is een voorbeeld.",
    )

    result = api_main.api_tts_preview(payload, request=_dummy_request(), x_api_key=None)

    assert result.provider == "openai"
    assert result.model == "gpt-4o-tts"
    assert result.voice == "alloy"
    assert result.filename == "sentence_preview.mp3"
    assert result.audio_b64
    assert result.summary.ok == 1
    assert logged
    assert logged[0]["kind"] == "audio_preview"


def test_api_tts_retries_transient_error_once(
    monkeypatch: pytest.MonkeyPatch, patch_api_auth: None, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "audio-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(audio_core, "AUDIO_CACHE_DIR", cache_dir, raising=False)

    calls = {"count": 0}

    def flaky_synthesize(*args: Any, **kwargs: Any) -> bytes:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return b"audio-data"

    monkeypatch.setattr(audio_core, "_synthesize", flaky_synthesize)

    payload = TTSRequest(
        run_id="run-2",
        provider="openai",
        model="gpt-4o-tts",
        voice="alloy",
        items=[{"card_id": "row-1", "type": "word", "text": "fiets"}],
    )

    result = api_main.api_tts(payload, request=_dummy_request(), x_api_key=None)

    assert calls["count"] == 2
    assert result.audios[0].status == "ok"
    assert result.summary.failed == 0


def test_api_tts_does_not_retry_validation_error(
    monkeypatch: pytest.MonkeyPatch, patch_api_auth: None, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "audio-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(audio_core, "AUDIO_CACHE_DIR", cache_dir, raising=False)

    calls = {"count": 0}

    def invalid_synthesize(*args: Any, **kwargs: Any) -> bytes:
        calls["count"] += 1
        raise ValueError("voice is invalid")

    monkeypatch.setattr(audio_core, "_synthesize", invalid_synthesize)

    payload = TTSRequest(
        run_id="run-3",
        provider="openai",
        model="gpt-4o-tts",
        voice="alloy",
        items=[{"card_id": "row-1", "type": "word", "text": "fiets"}],
    )

    result = api_main.api_tts(payload, request=_dummy_request(), x_api_key=None)

    assert calls["count"] == 1
    assert result.audios[0].status == "failed"
    assert result.summary.failed == 1


def test_api_tts_reuses_durable_audio_asset_without_provider_call(
    monkeypatch: pytest.MonkeyPatch, patch_api_auth: None
) -> None:
    touched: list[str] = []

    def fail_ensure(*args: Any, **kwargs: Any):
        raise AssertionError("provider synthesis should not run for durable cache hits")

    def fake_load_audio_assets(*, asset_keys: list[str]):
        key = asset_keys[0]
        return {
            key: {
                "asset_key": key,
                "provider": "openai",
                "model": "gpt-4o-tts",
                "voice": "alloy",
                "kind": "word",
                "filename": "word_fiets__alloy__cached.mp3",
                "content": b"cached-audio",
            }
        }, None

    def fake_touch_audio_assets(*, asset_keys: list[str]) -> None:
        touched.extend(asset_keys)

    monkeypatch.setattr(api_main, "ensure_audio_for_cards", fail_ensure)
    monkeypatch.setattr(api_main, "load_audio_assets", fake_load_audio_assets)
    monkeypatch.setattr(api_main, "touch_audio_assets", fake_touch_audio_assets)

    payload = TTSRequest(
        run_id="run-cached",
        provider="openai",
        model="gpt-4o-tts",
        voice="alloy",
        items=[{"card_id": "row-1", "type": "word", "text": "fiets"}],
    )

    result = api_main.api_tts(payload, request=_dummy_request(), x_api_key=None)

    assert result.summary.ok == 1
    assert result.summary.failed == 0
    assert result.summary.cached == 1
    assert result.audios[0].status == "cached"
    assert result.audios[0].filename == "word_fiets__alloy__cached.mp3"
    assert result.audios[0].audio_b64 is not None
    assert result.timing["durable_cache_hits"] == 1
    assert result.timing["total_requests"] == 0
    assert touched


def test_filter_openai_tts_models_returns_live_discovered_list() -> None:
    models = [
        "gpt-4o-mini-tts",
        "gpt-4o-tts",
        "gpt-4o-mini-tts",
        "gpt-5",
        "whisper-1",
        "gpt-4o-transcribe",
    ]
    filtered = api_main._filter_openai_tts_models(models)
    assert filtered == ["gpt-4o-mini-tts", "gpt-4o-tts"]


def test_filter_openai_tts_models_uses_fallback_when_discovery_empty() -> None:
    filtered = api_main._filter_openai_tts_models([])
    assert filtered
    assert api_main.AUDIO_TTS_MODEL in filtered
