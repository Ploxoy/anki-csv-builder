"""Run report aggregation (Streamlit-agnostic).

This module centralizes run report building so it can be reused by both the
Streamlit UI and the FastAPI service without importing Streamlit.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from config.pricing import AUDIO_MODEL_PRICING_USD_PER_1M_CHAR, MODEL_PRICING_USD_PER_1M

RunReport = Dict[str, Any]


_DEFAULT_RUN_STATS: Dict[str, Any] = {
    "batches": 0,
    "items": 0,
    "elapsed": 0.0,
    "errors": 0,
    "transient": 0,
    "start_ts": None,
}


def _ensure_run_stats(state: Any) -> Dict[str, Any]:
    stats = getattr(state, "run_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    merged: Dict[str, Any] = dict(_DEFAULT_RUN_STATS)
    merged.update({k: v for k, v in stats.items() if k in _DEFAULT_RUN_STATS})
    try:
        state.run_stats = merged
    except Exception:
        pass
    return merged


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


def resolve_text_pricing(model_id: Optional[str]) -> Optional[Dict[str, float]]:
    if not model_id:
        return None
    model_id = str(model_id)
    for key in sorted(MODEL_PRICING_USD_PER_1M.keys(), key=len, reverse=True):
        if model_id.startswith(key):
            return MODEL_PRICING_USD_PER_1M[key]
    return None


def resolve_audio_pricing(model_id: Optional[str]) -> Optional[float]:
    if not model_id:
        return None
    model_id = str(model_id)
    for key in sorted(AUDIO_MODEL_PRICING_USD_PER_1M_CHAR.keys(), key=len, reverse=True):
        if model_id.startswith(key):
            return AUDIO_MODEL_PRICING_USD_PER_1M_CHAR[key]
    return None


def reset_run_report(state: Any) -> None:
    """Clear any previously stored run report."""
    _state_set(state, "run_report", {})


def build_run_report(state: Any) -> RunReport:
    """Aggregate generation/audio metrics into a run report dict."""

    results = list(_state_get(state, "results", []) or [])
    stats = _ensure_run_stats(state)
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
    providers_counter: Counter[str] = Counter()
    levels_counter: Counter[str] = Counter()
    fallback_per_model: Counter[str] = Counter()
    per_model_usage: Dict[str, Dict[str, int]] = {}
    per_provider_usage: Dict[str, Dict[str, int]] = {}
    model_providers: Dict[str, set[str]] = {}
    completion_chars_raw_total = 0
    completion_chars_final_total = 0
    raw_trimmed_total = 0
    instructions_truncated = 0
    cache_diag_requests = 0
    cache_key_attached = 0
    cache_key_removed = 0
    cache_retention_removed = 0
    cacheable_prefix_count = 0
    cache_prefix_tokens_sum = 0
    cache_prefix_tokens_max = 0
    cache_prefix_hashes: Counter[str] = Counter()
    cache_retentions: Counter[str] = Counter()

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
                "completion_chars_raw": 0,
                "completion_chars_final": 0,
                "completion_chars_raw_max": 0,
                "completion_chars_final_max": 0,
                "raw_trimmed": 0,
            }
            per_model_usage[model_name] = entry
        return entry

    def _provider_usage(provider_name: str) -> Dict[str, int]:
        if not provider_name:
            provider_name = "unknown"
        entry = per_provider_usage.get(provider_name)
        if entry is None:
            entry = {
                "calls": 0,
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "cached": 0,
                "prompt_repair": 0,
                "completion_repair": 0,
                "total_repair": 0,
                "cached_repair": 0,
            }
            per_provider_usage[provider_name] = entry
        return entry

    for card in results:
        if not isinstance(card, dict):
            continue
        meta = card.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        err = str(card.get("error") or "").strip()
        provider = meta.get("provider")
        model = meta.get("model")
        if isinstance(model, str) and model:
            models_counter[model] += 1
        else:
            model = "unknown"
        if isinstance(provider, str) and provider:
            providers_counter[provider] += 1
        else:
            provider = "unknown"
        if model not in model_providers:
            model_providers[model] = set()
        model_providers[model].add(provider)
        level = meta.get("level")
        if isinstance(level, str) and level:
            levels_counter[level] += 1
        model_usage = _model_usage(model)
        provider_usage = _provider_usage(provider)
        model_usage["calls"] += 1
        provider_usage["calls"] += 1

        raw_len = int(meta.get("raw_response_length", 0) or 0)
        final_len = int(meta.get("card_text_length", 0) or 0)
        if raw_len > 0:
            completion_chars_raw_total += raw_len
            model_usage["completion_chars_raw"] += raw_len
            model_usage["completion_chars_raw_max"] = max(model_usage["completion_chars_raw_max"], raw_len)
        if final_len > 0:
            completion_chars_final_total += final_len
            model_usage["completion_chars_final"] += final_len
            model_usage["completion_chars_final_max"] = max(model_usage["completion_chars_final_max"], final_len)
        if bool(meta.get("raw_response_truncated")):
            raw_trimmed_total += 1
            model_usage["raw_trimmed"] += 1

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
        if isinstance(req, dict) and req:
            cache_diag_requests += 1
            if req.get("response_format_used"):
                schema_attempted += 1
            if int(req.get("retries", 0) or 0) > 0:
                retries += 1
            if bool(req.get("instructions_truncated")):
                instructions_truncated += 1
            if req.get("prompt_cache_key"):
                cache_key_attached += 1
            if bool(req.get("prompt_cache_key_removed")):
                cache_key_removed += 1
            if bool(req.get("prompt_cache_retention_removed")):
                cache_retention_removed += 1
            prefix_est_tokens = int(req.get("cache_prefix_estimated_tokens", 0) or 0)
            if prefix_est_tokens > 0:
                cache_prefix_tokens_sum += prefix_est_tokens
                cache_prefix_tokens_max = max(cache_prefix_tokens_max, prefix_est_tokens)
            if bool(req.get("cache_prefix_cacheable")):
                cacheable_prefix_count += 1
            prefix_hash = req.get("cache_prefix_hash")
            if isinstance(prefix_hash, str) and prefix_hash:
                cache_prefix_hashes[prefix_hash] += 1
            retention = req.get("prompt_cache_retention")
            if isinstance(retention, str) and retention:
                cache_retentions[retention] += 1
            cached_primary = int(req.get("cached_tokens", 0) or 0)
            prompt_primary = int(req.get("prompt_tokens", 0) or 0)
            completion_primary = int(req.get("completion_tokens", 0) or 0)
            total_primary = int(req.get("total_tokens", 0) or 0)
            cached_tokens_primary_total += cached_primary
            prompt_tokens_primary_total += prompt_primary
            completion_tokens_primary_total += completion_primary
            total_tokens_primary_total += total_primary
            for usage_bucket in (model_usage, provider_usage):
                usage_bucket["cached"] += cached_primary
                usage_bucket["prompt"] += prompt_primary
                usage_bucket["completion"] += completion_primary
                usage_bucket["total"] += total_primary
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
                    for usage_bucket in (model_usage, provider_usage):
                        usage_bucket["prompt_repair"] += repair_prompt
                        usage_bucket["completion_repair"] += repair_completion
                        usage_bucket["total_repair"] += repair_total
                        usage_bucket["cached_repair"] += repair_cached
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
    tokens_by_provider: Dict[str, Dict[str, int]] = {}
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
            "completion_chars_raw": usage["completion_chars_raw"],
            "completion_chars_final": usage["completion_chars_final"],
            "completion_chars_raw_max": usage["completion_chars_raw_max"],
            "completion_chars_final_max": usage["completion_chars_final_max"],
            "raw_trimmed": usage["raw_trimmed"],
        }
    for provider_name, usage in per_provider_usage.items():
        tokens_by_provider[provider_name] = {
            "calls": usage["calls"],
            "prompt": usage["prompt"],
            "prompt_repair": usage["prompt_repair"],
            "completion": usage["completion"],
            "completion_repair": usage["completion_repair"],
            "total": usage["total"],
            "total_repair": usage["total_repair"],
            "cached": usage["cached"],
            "cached_repair": usage["cached_repair"],
        }

    text_cost_by_model: Dict[str, Dict[str, Optional[float]]] = {}
    text_total_cost = 0.0
    text_cost_available = False
    text_missing_pricing: list[str] = []
    for model_name, usage in per_model_usage.items():
        prompt_all = usage["prompt"] + usage["prompt_repair"]
        completion_all = usage["completion"] + usage["completion_repair"]
        total_all = usage["total"] + usage["total_repair"]
        pricing = resolve_text_pricing(model_name if model_name != "unknown" else None)
        estimated = None
        if pricing and (prompt_all or completion_all):
            estimated = (prompt_all / 1_000_000.0) * pricing["input"] + (completion_all / 1_000_000.0) * pricing["output"]
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
    cache_prefix_tokens_avg = (
        cache_prefix_tokens_sum / cache_diag_requests if cache_diag_requests else 0.0
    )

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
                pricing = resolve_audio_pricing(model_name)
                estimated_audio = None
                if pricing and chars:
                    estimated_audio = (chars / 1_000_000.0) * pricing
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

    # Build normalized usage events (for future API billing)
    usage_events: List[Dict[str, Any]] = []

    for model_name, usage in per_model_usage.items():
        prompt_all = usage["prompt"] + usage["prompt_repair"]
        completion_all = usage["completion"] + usage["completion_repair"]
        cached_all = usage["cached"] + usage["cached_repair"]
        estimated_usd = text_cost_by_model.get(model_name, {}).get("estimated_usd")
        providers_for_model = sorted(model_providers.get(model_name, []) or ["unknown"])
        provider_for_event = providers_for_model[0] if providers_for_model else "unknown"
        usage_events.append(
            {
                "kind": "text",
                "provider": provider_for_event,
                "model": model_name,
                "input_tokens": prompt_all,
                "output_tokens": completion_all,
                "cached_tokens": cached_all,
                "audio_chars": None,
                "audio_tokens": None,
                "seconds": None,
                "raw_cost_usd": estimated_usd,
                "raw_cost_eur": None,
                "charged_cost_eur": None,
                "markup_tier": None,
                "markup_multiplier": None,
                "request_id": None,
                "elapsed_ms": None,
            }
        )

    audio_provider_for_event = audio_summary.get("provider") if isinstance(audio_summary, dict) else None
    if audio_cost_by_model:
        for model_name, data in audio_cost_by_model.items():
            chars = data.get("characters")
            estimated_usd = data.get("estimated_usd")
            usage_events.append(
                {
                    "kind": "audio",
                    "provider": audio_provider_for_event or "unknown",
                    "model": model_name,
                    "input_tokens": None,
                    "output_tokens": None,
                    "cached_tokens": None,
                    "audio_chars": chars,
                    "audio_tokens": None,
                    "seconds": None,
                    "raw_cost_usd": estimated_usd,
                    "raw_cost_eur": None,
                    "charged_cost_eur": None,
                    "markup_tier": None,
                    "markup_multiplier": None,
                    "request_id": None,
                    "elapsed_ms": None,
                }
            )

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
        cost_notes_parts.append("No text pricing configured for: " + ", ".join(sorted(set(text_missing_pricing))))
    if audio_missing_pricing:
        cost_notes_parts.append("No audio pricing configured for: " + ", ".join(sorted(set(audio_missing_pricing))))

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
            "providers": dict(providers_counter),
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
        "prompting": {
            "instructions_truncated": instructions_truncated,
            "cache_diagnostics": {
                "requests": cache_diag_requests,
                "cache_key_attached": cache_key_attached,
                "cache_key_removed_by_sdk": cache_key_removed,
                "cache_retention_removed_by_sdk": cache_retention_removed,
                "prefix_cacheable_requests": cacheable_prefix_count,
                "prefix_est_tokens_avg": cache_prefix_tokens_avg,
                "prefix_est_tokens_max": cache_prefix_tokens_max,
                "prefix_hash_unique": len(cache_prefix_hashes),
                "retentions": dict(cache_retentions),
            },
        },
        "tokens": {
            "prompt": prompt_total,
            "completion": completion_total,
            "total": tokens_total,
            "cached": cached_tokens_total,
            "chars": {
                "raw": completion_chars_raw_total,
                "final": completion_chars_final_total,
                "raw_trimmed_count": raw_trimmed_total,
            },
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
            "by_provider": tokens_by_provider,
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
        "usage_events": usage_events,
    }

    _state_set(state, "run_report", report)
    return report


def ensure_run_report(state: Any) -> RunReport:
    """Ensure a run report exists and return it."""
    report = _state_get(state, "run_report")
    if not isinstance(report, dict):
        return build_run_report(state)
    return report
