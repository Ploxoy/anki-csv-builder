from types import SimpleNamespace

import pytest

from app.run_report import build_run_report, reset_run_report


def _card(meta: dict, error: str = "") -> dict:
    return {
        "L2_word": "woord",
        "L2_cloze": "{{c1::woord}}",
        "L1_sentence": "",
        "L2_collocations": "",
        "L2_definition": "",
        "L1_gloss": "",
        "error": error,
        "meta": meta,
    }


def test_build_run_report_aggregates_metrics() -> None:
    state = SimpleNamespace()
    state.results = [
        _card(
                {
                    "repair_attempted": False,
                    "response_format_removed": False,
                    "temperature_removed": False,
                    "model": "gpt-5",
                    "level": "B1",
                    "request": {
                        "response_format_used": True,
                        "retries": 0,
                        "cached_tokens": 400,
                        "prompt_tokens": 400,
                        "completion_tokens": 120,
                        "total_tokens": 520,
                    },
                }
            ),
            _card(
                {
                    "repair_attempted": True,
                "response_format_removed": True,
                "temperature_removed": True,
                "repair_response_format_removed": True,
                "model": "gpt-5",
                "level": "B2",
                    "request": {
                        "response_format_used": True,
                        "retries": 1,
                        "response_format_error": "unexpected keyword argument 'text'",
                        "cached_tokens": 1100,
                        "prompt_tokens": 500,
                        "completion_tokens": 150,
                        "total_tokens": 650,
                    },
                },
                error="validation_failed: missing collocations",
            ),
        _card({"repair_attempted": False}, error="flagged_precheck"),
    ]
    state.sig_usage = {"omdat": 2, "maar": 1}
    state.sig_last = "maar"
    state.audio_summary = {
        "total_requests": 4,
        "cache_hits": 1,
        "word_success": 2,
        "sentence_success": 1,
        "word_skipped": 0,
        "sentence_skipped": 1,
        "errors": [],
        "provider": "openai",
        "voice": "alloy",
        "total_characters": 1000,
        "total_requests_billed": 2,
        "model_usage": {
            "gpt-4o-mini-tts": {
                "chars": 1000,
                "requests": 2,
                "fallback_requests": 0,
                "word_chars": 400,
                "sentence_chars": 600,
                "word_requests": 1,
                "sentence_requests": 1,
            }
        },
    }
    state.run_stats = {
        "batches": 2,
        "items": 3,
        "elapsed": 6.0,
        "errors": 1,
        "transient": 1,
        "start_ts": 123.0,
    }

    report = build_run_report(state)

    assert report["generation"]["total"] == 3
    assert report["generation"]["valid"] == 1
    assert report["generation"]["errored"] == 1  # flagged counted separately
    assert report["generation"]["flagged_precheck"] == 1
    assert report["generation"]["repair_attempted"] == 1
    assert report["generation"]["retries"] == 1
    assert report["response_format"]["schema_removed"] == 1
    assert report["response_format"]["temperature_removed"] == 1
    assert report["response_format"]["repair_schema_removed"] == 1
    tokens = report["tokens"]
    assert tokens["cached"] == 1500
    assert tokens["prompt"] == 900
    assert tokens["completion"] == 270
    assert tokens["total"] == 1170
    assert report["audio"]["total_characters"] == 1000
    assert report["cost"]["text"]["estimated_usd"] == pytest.approx(0.0513, rel=1e-6)
    assert report["cost"]["audio"]["estimated_usd"] == pytest.approx(0.015, rel=1e-6)
    assert report["cost"]["estimated_usd"] == pytest.approx(0.0663, rel=1e-6)
    assert report["cost"]["notes"] is None
    assert report["signalwords"]["total_found"] == 3
    assert report["signalwords"]["last"] == "maar"
    assert report["timing"]["batches"] == 2
    assert report["audio"]["total_requests"] == 4
    assert state.run_report == report


def test_reset_run_report() -> None:
    state = SimpleNamespace(run_report={"foo": "bar"})
    reset_run_report(state)
    assert state.run_report == {}
