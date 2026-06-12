from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import api.main as api_main
from core.api_schemas import GenerateRequest


def _dummy_request() -> Request:
    return Request({"type": "http", "headers": []})


def _card(word: str = "fiets") -> dict[str, Any]:
    return {
        "L2_word": word,
        "L2_cloze": f"Ik heb een {{{{c1::{word}}}}}.",
        "L1_sentence": "I have a bike.",
        "L2_collocations": "op de fiets",
        "L2_definition": "een voertuig met twee wielen",
        "L1_gloss": "bike",
        "meta": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "request": {"prompt_tokens": 10, "completion_tokens": 20, "cached_tokens": 0, "total_elapsed_ms": 1000},
        },
    }


def _payload(*, reuse: bool) -> GenerateRequest:
    return GenerateRequest(
        run_id="run-cache",
        prompt_version="p0",
        provider="openai",
        model="gpt-4.1-mini",
        cefr="B1",
        profile="balanced",
        l1="EN",
        flags={"reuse_text_cache": reuse},
        items=[{"id": "row-1", "woord": "fiets", "def_nl": "", "translation": ""}],
    )


def test_api_generate_reuses_saved_card_without_generation_call(monkeypatch):
    def fail_generate_card(*args: Any, **kwargs: Any):
        raise AssertionError("generate_card should not be called for text cache hits")

    monkeypatch.setattr(api_main, "_require_user", lambda request, x_api_key: "user-test")
    monkeypatch.setattr(api_main, "_openai_client_or_500", lambda: (_ for _ in ()).throw(AssertionError("client not needed")))
    monkeypatch.setattr(api_main, "generate_card", fail_generate_card)
    monkeypatch.setattr(api_main, "log_usage_events", lambda **kwargs: None)
    monkeypatch.setattr(api_main, "touch_generated_card_assets", lambda **kwargs: None)
    monkeypatch.setattr(
        api_main,
        "load_generated_card_assets",
        lambda **kwargs: ({kwargs["asset_keys"][0]: {"card_json": _card(), "status": "ok"}}, None),
    )

    result = api_main.api_generate(_payload(reuse=True), request=_dummy_request(), x_api_key=None)

    assert result.items[0].status == "ok"
    assert result.items[0].card is not None
    assert result.items[0].card.L2_word == "fiets"
    assert result.timing["text_cache_hits"] == 1
    assert result.timing["text_assets_stored"] == 0


def test_api_generate_stores_successful_card_on_cache_miss(monkeypatch):
    stored_assets: list[dict[str, Any]] = []

    def fake_generate_card(*args: Any, **kwargs: Any):
        return SimpleNamespace(card=_card("huis"), signal_usage={}, signal_last=None)

    def fake_store_generated_card_asset(*, asset: dict[str, Any]):
        stored_assets.append(asset)
        return True, None

    monkeypatch.setattr(api_main, "_require_user", lambda request, x_api_key: "user-test")
    monkeypatch.setattr(api_main, "_openai_client_or_500", lambda: object())
    monkeypatch.setattr(api_main, "generate_card", fake_generate_card)
    monkeypatch.setattr(api_main, "log_usage_events", lambda **kwargs: None)
    monkeypatch.setattr(api_main, "load_generated_card_assets", lambda **kwargs: ({}, None))
    monkeypatch.setattr(api_main, "store_generated_card_asset", fake_store_generated_card_asset)

    result = api_main.api_generate(_payload(reuse=True), request=_dummy_request(), x_api_key=None)

    assert result.items[0].status == "ok"
    assert result.items[0].card is not None
    assert result.items[0].card.L2_word == "huis"
    assert result.timing["text_cache_hits"] == 0
    assert result.timing["text_assets_stored"] == 1
    assert stored_assets
    assert stored_assets[0]["card_json"]["L2_word"] == "huis"
