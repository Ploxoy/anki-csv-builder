"""Run report aggregation and UI helpers."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import pandas as pd
import streamlit as st

from config.pricing import AUDIO_MODEL_PRICING_USD_PER_1K_CHAR, MODEL_PRICING_USD_PER_1K

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


def _resolve_pricing(model_id: Optional[str]) -> Optional[Dict[str, float]]:
    if not model_id:
        return None
    model_id = str(model_id)
    for key in sorted(MODEL_PRICING_USD_PER_1K.keys(), key=len, reverse=True):
        if model_id.startswith(key):
            return MODEL_PRICING_USD_PER_1K[key]
    return None


def _resolve_audio_pricing(model_id: Optional[str]) -> Optional[float]:
    if not model_id:
        return None
    model_id = str(model_id)
    for key in sorted(AUDIO_MODEL_PRICING_USD_PER_1K_CHAR.keys(), key=len, reverse=True):
        if model_id.startswith(key):
            return AUDIO_MODEL_PRICING_USD_PER_1K_CHAR[key]
    return None


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
    fallback_cards = 0
    prompt_tokens_primary_total = 0
    completion_tokens_primary_total = 0
    total_tokens_primary_total = 0
    cached_tokens_primary_total = 0
    prompt_tokens_repair_total = 0
    completion_tokens_repair_total = 0
    total_tokens_repair_total = 0
    cached_tokens_repair_total = 0
    response_format_errors: Counter[str] = Counter()
    models_counter: Counter[str] = Counter()
    levels_counter: Counter[str] = Counter()
    fallback_per_model: Counter[str] = Counter()
    per_model_usage: Dict[str, Dict[str, int]] = {}

    def _model_usage(model_name: str) -> Dict[str, int]:
        if not model_name:
            model_name = "unknown"
        entry = per_model_usage.get(model_name)
        if entry is None:
            entry = {
                "calls": 0,
                "repair_calls": 0,
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "cached": 0,
                "prompt_repair": 0,
                "completion_repair": 0,
                "total_repair": 0,
                "cached_repair": 0,
                "fallbacks": 0,
            }
            per_model_usage[model_name] = entry
        return entry

    for card in results:
        if not isinstance(card, dict):
            continue
        meta = card.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        err = str(card.get("error") or "").strip()
        model = meta.get("model")
        if isinstance(model, str) and model:
            models_counter[model] += 1
        else:
            model = "unknown"
        level = meta.get("level")
        if isinstance(level, str) and level:
            levels_counter[level] += 1
        model_usage = _model_usage(model)
        model_usage["calls"] += 1

        if err:
            if err == "flagged_precheck":
                flagged += 1
            else:
                errored += 1
        else:
            valid += 1

        if meta.get("repair_attempted"):
            repairs += 1
            model_usage["repair_calls"] += 1
        if meta.get("response_format_removed"):
            schema_removed += 1
            fallback_cards += 1
            model_usage["fallbacks"] += 1
            fallback_per_model[model] += 1
        if meta.get("temperature_removed"):
            temp_removed += 1
            fallback_cards += 1
            model_usage["fallbacks"] += 1
            fallback_per_model[model] += 1
        if meta.get("repair_response_format_removed"):
            repair_schema_removed += 1

        req = meta.get("request") or {}
        if isinstance(req, dict):
            if req.get("response_format_used"):
                schema_attempted += 1
            if int(req.get("retries", 0) or 0) > 0:
                retries += 1
            cached_primary = int(req.get("cached_tokens", 0) or 0)
            prompt_primary = int(req.get("prompt_tokens", 0) or 0)
            completion_primary = int(req.get("completion_tokens", 0) or 0)
            total_primary = int(req.get("total_tokens", 0) or 0)
            cached_tokens_primary_total += cached_primary
            prompt_tokens_primary_total += prompt_primary
            completion_tokens_primary_total += completion_primary
            total_tokens_primary_total += total_primary
            model_usage["cached"] += cached_primary
            model_usage["prompt"] += prompt_primary
            model_usage["completion"] += completion_primary
            model_usage["total"] += total_primary
            err_reason = req.get("response_format_error") or meta.get("response_format_error")
            if isinstance(err_reason, str) and err_reason:
                response_format_errors[err_reason] += 1
            repair_info = req.get("repair_usage") or {}
            if isinstance(repair_info, dict):
                repair_prompt = int(repair_info.get("prompt_tokens", 0) or 0)
                repair_completion = int(repair_info.get("completion_tokens", 0) or 0)
                repair_total = int(repair_info.get("total_tokens", 0) or 0)
                repair_cached = int(repair_info.get("cached_tokens", 0) or 0)
                if repair_prompt or repair_completion or repair_total or repair_cached:
                    prompt_tokens_repair_total += repair_prompt
                    completion_tokens_repair_total += repair_completion
                    total_tokens_repair_total += repair_total
                    cached_tokens_repair_total += repair_cached
                    model_usage["prompt_repair"] += repair_prompt
                    model_usage["completion_repair"] += repair_completion
                    model_usage["total_repair"] += repair_total
                    model_usage["cached_repair"] += repair_cached
                if int(repair_info.get("retries", 0) or 0) > 0:
                    retries += 1
        else:
            err_reason = meta.get("response_format_error")
            if isinstance(err_reason, str) and err_reason:
                response_format_errors[err_reason] += 1

    elapsed = float(stats.get("elapsed", 0.0) or 0.0)
    batches = int(stats.get("batches", 0) or 0)
    items_done = int(stats.get("items", total) or total)
    start_ts = stats.get("start_ts")
    transient_errors = int(stats.get("transient", 0) or 0)
    avg_batch = elapsed / batches if batches else 0.0
    per_second = items_done / elapsed if elapsed else 0.0

    prompt_total = prompt_tokens_primary_total + prompt_tokens_repair_total
    completion_total = completion_tokens_primary_total + completion_tokens_repair_total
    tokens_total = total_tokens_primary_total + total_tokens_repair_total
    cached_tokens_total = cached_tokens_primary_total + cached_tokens_repair_total

    tokens_by_model: Dict[str, Dict[str, int]] = {}
    for model_name, usage in per_model_usage.items():
        tokens_by_model[model_name] = {
            "calls": usage["calls"],
            "repair_calls": usage["repair_calls"],
            "prompt": usage["prompt"],
            "prompt_repair": usage["prompt_repair"],
            "completion": usage["completion"],
            "completion_repair": usage["completion_repair"],
            "total": usage["total"],
            "total_repair": usage["total_repair"],
            "cached": usage["cached"],
            "cached_repair": usage["cached_repair"],
            "fallbacks": usage["fallbacks"],
        }

    text_cost_by_model: Dict[str, Dict[str, Optional[float]]] = {}
    text_total_cost = 0.0
    text_cost_available = False
    text_missing_pricing: list[str] = []
    for model_name, usage in per_model_usage.items():
        prompt_all = usage["prompt"] + usage["prompt_repair"]
        completion_all = usage["completion"] + usage["completion_repair"]
        total_all = usage["total"] + usage["total_repair"]
        pricing = _resolve_pricing(model_name if model_name != "unknown" else None)
        estimated = None
        if pricing and (prompt_all or completion_all):
            estimated = (
                (prompt_all / 1000.0) * pricing["input"]
                + (completion_all / 1000.0) * pricing["output"]
            )
            text_total_cost += estimated
            text_cost_available = True
        else:
            if model_name not in ("unknown",):
                text_missing_pricing.append(model_name)
        text_cost_by_model[model_name] = {
            "prompt_tokens": prompt_all,
            "completion_tokens": completion_all,
            "total_tokens": total_all,
            "estimated_usd": estimated,
        }

    if text_cost_available:
        text_total_cost = round(text_total_cost, 6)

    fallback_rate = fallback_cards / total if total else 0.0
    schema_removed_rate = schema_removed / schema_attempted if schema_attempted else 0.0
    repair_schema_removed_rate = repair_schema_removed / repairs if repairs else 0.0
    repair_rate_value = repairs / total if total else 0.0
    errored_rate = errored / total if total else 0.0
    flagged_rate = flagged / total if total else 0.0

    audio_cost_by_model: Dict[str, Dict[str, Optional[float]]] = {}
    audio_total_cost = 0.0
    audio_cost_available = False
    audio_missing_pricing: list[str] = []
    if audio_summary:
        model_usage_audio = audio_summary.get("model_usage") or {}
        if isinstance(model_usage_audio, dict):
            for model_name, usage in model_usage_audio.items():
                if not isinstance(usage, dict):
                    continue
                chars = int(usage.get("chars", 0) or 0)
                pricing = _resolve_audio_pricing(model_name)
                estimated_audio = None
                if pricing and chars:
                    estimated_audio = (chars / 1000.0) * pricing
                    audio_total_cost += estimated_audio
                    audio_cost_available = True
                else:
                    if pricing is None and model_name not in ("unknown",):
                        audio_missing_pricing.append(model_name)
                audio_cost_by_model[model_name] = {
                    "characters": chars,
                    "estimated_usd": estimated_audio,
                    "requests": int(usage.get("requests", 0) or 0),
                }
    if audio_cost_available:
        audio_total_cost = round(audio_total_cost, 6)

    combined_cost = 0.0
    combined_cost_available = False
    if text_cost_available:
        combined_cost += text_total_cost
        combined_cost_available = True
    if audio_cost_available:
        combined_cost += audio_total_cost
        combined_cost_available = True
    if combined_cost_available:
        combined_cost = round(combined_cost, 6)

    cost_notes_parts: List[str] = []
    if text_missing_pricing:
        cost_notes_parts.append(
            "No text pricing configured for: " + ", ".join(sorted(set(text_missing_pricing)))
        )
    if audio_missing_pricing:
        cost_notes_parts.append(
            "No audio pricing configured for: " + ", ".join(sorted(set(audio_missing_pricing)))
        )

    report: RunReport = {
        "generation": {
            "total": total,
            "valid": valid,
            "errored": errored,
            "flagged_precheck": flagged,
            "repair_attempted": repairs,
            "repair_rate": repair_rate_value,
            "errored_rate": errored_rate,
            "flagged_rate": flagged_rate,
            "retries": retries,
            "models": dict(models_counter),
            "levels": dict(levels_counter),
        },
        "response_format": {
            "schema_attempted": schema_attempted,
            "schema_removed": schema_removed,
            "schema_removed_rate": schema_removed_rate,
            "repair_schema_removed": repair_schema_removed,
            "repair_schema_removed_rate": repair_schema_removed_rate,
            "temperature_removed": temp_removed,
            "errors": dict(response_format_errors),
            "fallback_cards": fallback_cards,
            "fallback_rate": fallback_rate,
            "fallback_by_model": dict(fallback_per_model),
        },
        "tokens": {
            "prompt": prompt_total,
            "completion": completion_total,
            "total": tokens_total,
            "cached": cached_tokens_total,
            "primary": {
                "prompt": prompt_tokens_primary_total,
                "completion": completion_tokens_primary_total,
                "total": total_tokens_primary_total,
                "cached": cached_tokens_primary_total,
            },
            "repair": {
                "prompt": prompt_tokens_repair_total,
                "completion": completion_tokens_repair_total,
                "total": total_tokens_repair_total,
                "cached": cached_tokens_repair_total,
            },
            "by_model": tokens_by_model,
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
            "estimated_usd": combined_cost if combined_cost_available else None,
            "text": {
                "estimated_usd": text_total_cost if text_cost_available else None,
                "by_model": text_cost_by_model,
            },
            "audio": {
                "estimated_usd": audio_total_cost if audio_cost_available else None,
                "by_model": audio_cost_by_model,
            },
            "notes": " | ".join(cost_notes_parts) if cost_notes_parts else None,
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
    repair_attempted = int(generation.get("repair_attempted", 0) or 0)
    repair_rate = float(generation.get("repair_rate", 0) or 0.0)
    flagged_rate = float(generation.get("flagged_rate", 0) or 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Cards total", total)
    col2.metric("Valid cards", valid)
    col3.metric("Errored cards", errored)
    st.caption(
        f"Flagged pre-check: {flagged} ({flagged_rate:.1%}) • "
        f"Repairs attempted: {repair_attempted} ({repair_rate:.1%}) • "
        f"Retries (any stage): {generation.get('retries', 0)}"
    )

    rf = report.get("response_format", {})
    fallback_rate = float(rf.get("fallback_rate", 0) or 0.0)
    schema_attempted = int(rf.get("schema_attempted", 0) or 0)
    schema_removed = int(rf.get("schema_removed", 0) or 0)
    schema_removed_rate = float(rf.get("schema_removed_rate", 0) or 0.0)
    repair_schema_removed = int(rf.get("repair_schema_removed", 0) or 0)
    repair_schema_removed_rate = float(rf.get("repair_schema_removed_rate", 0) or 0.0)
    temp_removed = int(rf.get("temperature_removed", 0) or 0)
    fallback_cards = int(rf.get("fallback_cards", 0) or 0)
    st.write(
        "**Response format:** "
        f"attempted {schema_attempted}, "
        f"removed {schema_removed} ({schema_removed_rate:.1%}), "
        f"repair removed {repair_schema_removed} ({repair_schema_removed_rate:.1%}), "
        f"temperature removed {temp_removed}. "
        f"Fallback cards (schema or temp): {fallback_cards} ({fallback_rate:.1%})."
    )
    rf_errors = rf.get("errors", {})
    if rf_errors:
        st.write("Schema issues:", rf_errors)

    token_stats = report.get("tokens", {})
    repair_tokens = token_stats.get("repair", {})
    st.write(
        "**Tokens:** "
        f"prompt {token_stats.get('prompt', 0)} "
        f"(repair {repair_tokens.get('prompt', 0)}), "
        f"completion {token_stats.get('completion', 0)} "
        f"(repair {repair_tokens.get('completion', 0)}), "
        f"total {token_stats.get('total', 0)} "
        f"(repair {repair_tokens.get('total', 0)}), "
        f"cached {token_stats.get('cached', 0)}."
    )
    tokens_by_model = token_stats.get("by_model", {})
    cost_section = report.get("cost", {})
    text_cost_section = cost_section.get("text", {}) if isinstance(cost_section, dict) else {}
    text_cost_by_model = text_cost_section.get("by_model", {}) if isinstance(text_cost_section, dict) else {}
    if tokens_by_model:
        rows = []
        for model_name, data in tokens_by_model.items():
            prompt_all = (data.get("prompt", 0) or 0) + (data.get("prompt_repair", 0) or 0)
            completion_all = (data.get("completion", 0) or 0) + (data.get("completion_repair", 0) or 0)
            total_all = (data.get("total", 0) or 0) + (data.get("total_repair", 0) or 0)
            rows.append(
                {
                    "Model": model_name,
                    "Calls": data.get("calls", 0),
                    "Repair calls": data.get("repair_calls", 0),
                    "Prompt tokens": prompt_all,
                    "Completion tokens": completion_all,
                    "Total tokens": total_all,
                    "Fallback cards": data.get("fallbacks", 0),
                    "Cost (USD)": (
                        text_cost_by_model.get(model_name, {}).get("estimated_usd")
                        if text_cost_by_model
                        else None
                    ),
                }
            )
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            df.style.format({"Cost (USD)": lambda v: "" if pd.isna(v) else f"{v:.6f}"}),
            use_container_width=True,
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
                "voice": audio.get("voice"),
                "characters_billed": audio.get("total_characters"),
                "requests_billed": audio.get("total_requests_billed"),
            },
        )
        audio_usage = audio.get("model_usage") if isinstance(audio, dict) else None
        audio_cost_by_model = cost_section.get("audio", {}).get("by_model", {}) if isinstance(cost_section, dict) else {}
        if isinstance(audio_usage, dict) and audio_usage:
            audio_rows = []
            for model_name, data in audio_usage.items():
                if not isinstance(data, dict):
                    continue
                audio_rows.append(
                    {
                        "Model": model_name,
                        "Requests billed": data.get("requests", 0),
                        "Fallback requests": data.get("fallback_requests", 0),
                        "Characters": data.get("chars", 0),
                        "Word chars": data.get("word_chars", 0),
                        "Sentence chars": data.get("sentence_chars", 0),
                        "Cost (USD)": (
                            audio_cost_by_model.get(model_name, {}).get("estimated_usd")
                            if audio_cost_by_model
                            else None
                        ),
                    }
                )
            if audio_rows:
                df_audio = pd.DataFrame(audio_rows).set_index("Model")
                st.dataframe(
                    df_audio.style.format({"Cost (USD)": lambda v: "" if pd.isna(v) else f"{v:.6f}"}),
                    use_container_width=True,
                )
    else:
        st.write("Audio: not synthesized yet.")

    est_cost = cost_section.get("estimated_usd") if isinstance(cost_section, dict) else None
    text_cost_est = text_cost_section.get("estimated_usd") if isinstance(text_cost_section, dict) else None
    audio_cost_section = cost_section.get("audio", {}) if isinstance(cost_section, dict) else {}
    audio_cost_est = audio_cost_section.get("estimated_usd") if isinstance(audio_cost_section, dict) else None
    if est_cost is not None:
        cost_parts = []
        if text_cost_est is not None:
            cost_parts.append(f"text ${text_cost_est:.4f}")
        if audio_cost_est is not None:
            cost_parts.append(f"audio ${audio_cost_est:.4f}")
        details = f" ({', '.join(cost_parts)})" if cost_parts else ""
        st.write(f"**Cost estimate:** ${est_cost:.4f} USD{details}.")
    else:
        st.write(
            "Cost estimate unavailable: configure pricing in `config.settings.MODEL_PRICING_USD_PER_1K` "
            "and `AUDIO_MODEL_PRICING_USD_PER_1K_CHAR`."
        )
    notes = cost_section.get("notes") if isinstance(cost_section, dict) else None
    if notes:
        st.write(notes)

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Download run report (JSON)",
        data=payload.encode("utf-8"),
        file_name="run_report.json",
        mime="application/json",
    )
