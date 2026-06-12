from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import api.main as api_main
from core.api_schemas import ExportDeckRequest


def _dummy_request() -> Request:
    return Request({"type": "http", "headers": []})


@pytest.fixture
def patch_api_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "_require_user", lambda request, x_api_key: "user-test")


def _payload(media_map: dict[str, str] | None = None, use_persisted_media: bool = False) -> ExportDeckRequest:
    card = {
        "L2_word": "fiets",
        "L2_cloze": "Ik zie een {{c1::fiets}}.",
        "L1_sentence": "I see a bicycle.",
        "L2_collocations": "een fiets zien; op de fiets; nieuwe fiets",
        "L2_definition": "een voertuig met twee wielen",
        "L1_gloss": "bicycle",
    }
    if media_map is not None or use_persisted_media:
        card["AudioWord"] = "[sound:word_fiets.mp3]"
    return ExportDeckRequest(
        run_id="run-1",
        l1="EN",
        cefr="B1",
        profile="balanced",
        model="gpt-4.1-mini",
        deck_name="Dutch",
        guid_policy="stable",
        include_basic_reversed=False,
        include_basic_typein=False,
        use_persisted_media=use_persisted_media,
        media_map=media_map,
        cards=[card],
    )


def test_api_export_apkg_streams_attachment(monkeypatch: pytest.MonkeyPatch, patch_api_auth: None) -> None:
    monkeypatch.setattr(api_main, "HAS_GENANKI", True)
    monkeypatch.setattr(api_main, "build_anki_package", lambda *args, **kwargs: b"apkg-bytes")

    result = api_main.api_export_apkg(_payload(), request=_dummy_request(), x_api_key=None)

    assert isinstance(result, StreamingResponse)
    assert result.media_type == "application/octet-stream"
    assert 'attachment; filename="Dutch.apkg"' == result.headers.get("content-disposition")
    assert result.headers.get("x-card-count") == "1"


def test_api_export_apkg_rejects_large_request(monkeypatch: pytest.MonkeyPatch, patch_api_auth: None) -> None:
    monkeypatch.setattr(api_main, "HAS_GENANKI", True)
    big_media = {"word.mp3": "A" * (api_main.EXPORT_REQUEST_SOFT_LIMIT_BYTES + 1024)}

    with pytest.raises(HTTPException) as exc_info:
        api_main.api_export_apkg(_payload(media_map=big_media), request=_dummy_request(), x_api_key=None)

    assert exc_info.value.status_code == 413
    assert "too large for Vercel" in str(exc_info.value.detail)


def test_api_export_apkg_uses_persisted_media(monkeypatch: pytest.MonkeyPatch, patch_api_auth: None) -> None:
    monkeypatch.setattr(api_main, "HAS_GENANKI", True)
    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        captured["media_files"] = kwargs.get("media_files")
        return b"apkg-bytes"

    monkeypatch.setattr(api_main, "build_anki_package", fake_build)
    monkeypatch.setattr(
        api_main,
        "load_run_media_assets",
        lambda **kwargs: ({"word_fiets.mp3": b"persisted-audio"}, None),
    )

    result = api_main.api_export_apkg(
        _payload(media_map=None, use_persisted_media=True),
        request=_dummy_request(),
        x_api_key=None,
    )

    assert isinstance(result, StreamingResponse)
    assert captured["media_files"] == {"word_fiets.mp3": b"persisted-audio"}
