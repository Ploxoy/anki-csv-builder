"""Run report aggregation and UI helpers."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional

import streamlit as st

from .run_status import ensure_run_stats


RunReport = Dict[str, Any]


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    try:
        return getattr(state, key)
    except AttributeError:
        if isinstance(state, Mapping):
            return state.get(key, default)
        if hasattr(state, "get"):
            return state.get(key, default)  # type: ignore[no-any-return]
    return default


def _state_set(state: Any, key: str, value: Any) -> None:
    try:
        setattr(state, key, value)
    except Exception:
        if isinstance(state, MutableMapping):
            state[key] = value  # type: ignore[index]
        else:
            try:
                state[key] = value  # type: ignore[index]
            except Exception:
                pass


def reset_run_report(state: Any) -> None:
    """Clear any previously stored run report."""
    _state_set(state, "run_report", {})


def build_run_report(state: Any) -> RunReport:
    """Aggregate generation/audio metrics into a run report dict."""

    results = list(_state_get(state, "results", []) or [])
    stats = ensure_run_stats(state)
    usage_raw = _state_get(state, "sig_usage", {}) or {}
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    sig_usage: Dict[str, int] = {str(k): int(v) for k, v in usage_raw.items() if isinstance(v, int) or isinstance(v, float)}
    sig_usage = {k: int(v) for k, v in sig_usage.items()}
    sig_last = _state_get(state, "sig_last")
    audio_summary = _state_get(state, "audio_summary") or {}
    if not isinstance(audio_summary, dict):
        audio_summary = {}

    total = len(results)
    valid = 0
    errored = 0
    flagged = 0
    repairs = 0
    schema_removed = 0
    repair_schema_removed = 0
    temp_removed = 0
    schema_attempted = 0
    retries = 0
    cached_tokens_total = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    response_format_errors: Counter[str] = Counter()
    models_counter: Counter[str] = Counter()
    levels_counter: Counter[str] = Counter()

    for card in results:
        if not isinstance(card, dict):
            continue
        meta = card.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        err = str(card.get("error") or "").strip()
        if err:
            if err == "flagged_precheck":
                flagged += 1
            else:
                errored += 1
        else:
            valid += 1

        if meta.get("repair_attempted"):
            repairs += 1
        if meta.get("response_format_removed"):
            schema_removed += 1
        if meta.get("temperature_removed"):
            temp_removed += 1
        if meta.get("repair_response_format_removed"):
            repair_schema_removed += 1

        req = meta.get("request") or {}
        if isinstance(req, dict):
            if req.get("response_format_used"):
                schema_attempted += 1
            if int(req.get("retries", 0) or 0) > 0:
                retries += 1
            cached_tokens_total += int(req.get("cached_tokens", 0) or 0)
            prompt_tokens_total += int(req.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(req.get("completion_tokens", 0) or 0)
            total_tokens_total += int(req.get("total_tokens", 0) or 0)
            err_reason = req.get("response_format_error") or meta.get("response_format_error")
            if isinstance(err_reason, str) and err_reason:
                response_format_errors[err_reason] += 1
        else:
            err_reason = meta.get("response_format_error")
            if isinstance(err_reason, str) and err_reason:
                response_format_errors[err_reason] += 1

        model = meta.get("model")
        if isinstance(model, str) and model:
            models_counter[model] += 1
        level = meta.get("level")
        if isinstance(level, str) and level:
            levels_counter[level] += 1

    elapsed = float(stats.get("elapsed", 0.0) or 0.0)
    batches = int(stats.get("batches", 0) or 0)
    items_done = int(stats.get("items", total) or total)
    start_ts = stats.get("start_ts")
    transient_errors = int(stats.get("transient", 0) or 0)
    avg_batch = elapsed / batches if batches else 0.0
    per_second = items_done / elapsed if elapsed else 0.0

    report: RunReport = {
        "generation": {
            "total": total,
            "valid": valid,
            "errored": errored,
            "flagged_precheck": flagged,
            "repair_attempted": repairs,
            "retries": retries,
            "models": dict(models_counter),
            "levels": dict(levels_counter),
        },
        "response_format": {
            "schema_attempted": schema_attempted,
            "schema_removed": schema_removed,
            "repair_schema_removed": repair_schema_removed,
            "temperature_removed": temp_removed,
            "errors": dict(response_format_errors),
        },
        "tokens": {
            "prompt": prompt_tokens_total,
            "completion": completion_tokens_total,
            "total": total_tokens_total,
            "cached": cached_tokens_total,
        },
        "signalwords": {
            "usage": dict(sig_usage),
            "total_found": int(sum(sig_usage.values())),
            "last": sig_last,
        },
        "timing": {
            "batches": batches,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_batch": avg_batch,
            "items_per_second": per_second,
            "start_timestamp": start_ts,
            "transient_errors": transient_errors,
        },
        "audio": audio_summary,
        "cost": {
            "estimated_usd": None,
            "notes": "Token usage tracked; cost estimation coming later.",
        },
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    _state_set(state, "run_report", report)
    return report


def ensure_run_report(state: Any) -> RunReport:
    """Ensure a run report exists and return it."""
    report = _state_get(state, "run_report")
    if not isinstance(report, dict):
        return build_run_report(state)
    return report


def render_run_report_section(state: Any) -> None:
    """Render the Run report block in the Streamlit UI."""
    report = ensure_run_report(state)
    generation = report.get("generation", {})
    total = int(generation.get("total", 0) or 0)

    if total == 0:
        st.info("Run report will appear here after you generate at least one batch.")
        return

    valid = int(generation.get("valid", 0) or 0)
    errored = int(generation.get("errored", 0) or 0)
    flagged = int(generation.get("flagged_precheck", 0) or 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Cards total", total)
    col2.metric("Valid cards", valid)
    col3.metric("Errored cards", errored)
    st.caption(f"Flagged pre-check: {flagged} • Repairs attempted: {generation.get('repair_attempted', 0)} • Retries: {generation.get('retries', 0)}")

    rf = report.get("response_format", {})
    st.write(
        f"**Response format:** attempted {rf.get('schema_attempted', 0)}, "
        f"removed {rf.get('schema_removed', 0)}, "
        f"repair removed {rf.get('repair_schema_removed', 0)}, "
        f"temperature removed {rf.get('temperature_removed', 0)}."
    )
    rf_errors = rf.get("errors", {})
    if rf_errors:
        st.write("Schema issues:", rf_errors)

    token_stats = report.get("tokens", {})
    st.write(
        "**Tokens:** prompt {prompt} • completion {completion} • total {total} • cached {cached}".format(
            prompt=token_stats.get("prompt", 0),
            completion=token_stats.get("completion", 0),
            total=token_stats.get("total", 0),
            cached=token_stats.get("cached", 0),
        )
    )

    signal = report.get("signalwords", {})
    usage = signal.get("usage", {})
    if usage:
        st.write("Top signal words:", dict(sorted(usage.items(), key=lambda item: item[1], reverse=True)[:5]))
    else:
        st.write("Signal words: no usages recorded.")

    timing = report.get("timing", {})
    st.write(
        f"**Timing:** batches {timing.get('batches', 0)}, "
        f"elapsed {timing.get('elapsed_seconds', 0):.1f}s, "
        f"{timing.get('items_per_second', 0):.2f} cards/s, "
        f"avg batch {timing.get('avg_seconds_per_batch', 0):.1f}s, "
        f"transient errors {timing.get('transient_errors', 0)}."
    )

    audio = report.get("audio", {})
    if audio:
        st.write(
            "Audio summary:",
            {
                "requests": audio.get("total_requests"),
                "cache_hits": audio.get("cache_hits"),
                "word_success": audio.get("word_success"),
                "sentence_success": audio.get("sentence_success"),
                "word_skipped": audio.get("word_skipped"),
                "sentence_skipped": audio.get("sentence_skipped"),
                "provider": audio.get("provider"),
            },
        )
    else:
        st.write("Audio: not synthesized yet.")

    st.write("Cost estimate: token usage not tracked yet.")

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Download run report (JSON)",
        data=payload.encode("utf-8"),
        file_name="run_report.json",
        mime="application/json",
    )
