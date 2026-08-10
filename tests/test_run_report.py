from types import SimpleNamespace

import pytest

from core.run_report import build_run_report, reset_run_report, resolve_text_pricing


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
                "raw_response_length": 1000,
                "card_text_length": 620,
                "raw_response_truncated": False,
                "request": {
                    "response_format_used": True,
                    "retries": 0,
                    "cached_tokens": 400,
                    "prompt_tokens": 400,
                    "completion_tokens": 120,
                    "total_tokens": 520,
                    "prompt_cache_key": "doedutch:card:aaaa",
                    "cache_prefix_hash": "abc123",
                    "cache_prefix_estimated_tokens": 1200,
                    "cache_prefix_cacheable": True,
                    "prompt_cache_retention": "24h",
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
                "raw_response_length": 1200,
                "card_text_length": 640,
                "raw_response_truncated": True,
                "repair_response_length": 800,
                "request": {
                    "response_format_used": True,
                    "retries": 1,
                    "response_format_error": "unexpected keyword argument 'text'",
                    "cached_tokens": 1100,
                        "prompt_tokens": 500,
                        "completion_tokens": 150,
                        "total_tokens": 650,
                        "prompt_cache_key": "doedutch:card:bbbb",
                        "prompt_cache_key_removed": True,
                        "prompt_cache_retention_removed": True,
                        "cache_prefix_hash": "abc123",
                        "cache_prefix_estimated_tokens": 1200,
                        "cache_prefix_cacheable": True,
                        "prompt_cache_retention": "24h",
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
            "gpt-4o-mini-tts-2025-12-15": {
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
    cache_diag = report["prompting"]["cache_diagnostics"]
    assert cache_diag["requests"] == 2
    assert cache_diag["cache_key_attached"] == 2
    assert cache_diag["cache_key_removed_by_sdk"] == 1
    assert cache_diag["cache_retention_removed_by_sdk"] == 1
    assert cache_diag["prefix_cacheable_requests"] == 2
    assert cache_diag["prefix_hash_unique"] == 1
    assert cache_diag["retentions"] == {"24h": 2}
    tokens = report["tokens"]
    assert tokens["cached"] == 1500
    assert tokens["prompt"] == 900
    assert tokens["completion"] == 270
    assert tokens["total"] == 1170
    chars = tokens["chars"]
    assert chars["raw"] == 2200
    assert chars["final"] == 1260
    assert chars["raw_trimmed_count"] == 1
    assert report["audio"]["total_characters"] == 1000
    assert report["cost"]["text"]["estimated_usd"] == pytest.approx(0.003825, rel=1e-6)
    assert report["cost"]["audio"]["estimated_usd"] == pytest.approx(0.00015, rel=1e-6)
    assert report["cost"]["estimated_usd"] == pytest.approx(0.003975, rel=1e-6)
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


def test_resolve_text_pricing_supports_current_flagship_models() -> None:
    assert resolve_text_pricing("gpt-5.6-sol") == {"input": 5.0, "cached_input": 0.5, "cache_write": 6.25, "output": 30.0}
    assert resolve_text_pricing("gpt-5.6") == {"input": 5.0, "cached_input": 0.5, "cache_write": 6.25, "output": 30.0}
    assert resolve_text_pricing("gpt-5.6-terra") == {"input": 2.0, "cached_input": 0.2, "cache_write": 2.5, "output": 12.0}
    assert resolve_text_pricing("gpt-5.6-luna") == {"input": 0.2, "cached_input": 0.02, "cache_write": 0.25, "output": 1.2}
    assert resolve_text_pricing("gpt-5.6-terra-2026-08-10") == {"input": 2.0, "cached_input": 0.2, "cache_write": 2.5, "output": 12.0}
    assert resolve_text_pricing("gpt-5.5") == {"input": 5.0, "cached_input": 0.5, "output": 30.0}
    assert resolve_text_pricing("gpt-5.5-pro") == {"input": 30.0, "output": 180.0}
    assert resolve_text_pricing("gpt-5.4") == {"input": 2.5, "cached_input": 0.25, "output": 15.0}
    assert resolve_text_pricing("gpt-5.4-mini") == {"input": 0.75, "cached_input": 0.075, "output": 4.5}
    assert resolve_text_pricing("gpt-5.4-nano") == {"input": 0.2, "cached_input": 0.02, "output": 1.25}
    assert resolve_text_pricing("gpt-5.4-pro") == {"input": 30.0, "output": 180.0}


def test_build_run_report_uses_cached_input_pricing_when_available() -> None:
    state = SimpleNamespace()
    state.results = [
        _card(
            {
                "model": "gpt-5.6-terra",
                "request": {
                    "prompt_tokens": 1000,
                    "cached_tokens": 400,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                },
            }
        )
    ]
    state.sig_usage = {}
    state.audio_summary = {}
    state.run_stats = {}

    report = build_run_report(state)

    # Terra: 600 uncached input * $2/M + 400 cached input * $0.20/M + 200 output * $12/M.
    assert report["cost"]["text"]["estimated_usd"] == pytest.approx(0.00368, rel=1e-6)
    assert report["cost"]["text"]["by_model"]["gpt-5.6-terra"]["cached_tokens"] == 400
