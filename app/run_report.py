"""Run report aggregation and UI helpers."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.run_report import RunReport, build_run_report, ensure_run_report, reset_run_report


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

    prompting = report.get("prompting", {})
    if prompting and prompting.get("instructions_truncated", 0):
        st.warning(
            f"Instructions were truncated for {prompting.get('instructions_truncated', 0)} request(s) in this run (UI log view only; full prompt was sent to the API)."
        )

    token_stats = report.get("tokens", {})
    repair_tokens = token_stats.get("repair", {})
    char_stats = token_stats.get("chars", {})
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
    if isinstance(char_stats, dict) and char_stats:
        st.caption(
            "Completion chars: raw {raw} • final (sanitized fields) {final} • truncated responses {trimmed}.".format(
                raw=char_stats.get("raw", 0),
                final=char_stats.get("final", 0),
                trimmed=char_stats.get("raw_trimmed_count", 0),
            )
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
                    "Raw chars": data.get("completion_chars_raw", 0),
                    "Final chars": data.get("completion_chars_final", 0),
                    "Max raw chars": data.get("completion_chars_raw_max", 0),
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
            width="stretch",
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
                    width="stretch",
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
            "Cost estimate unavailable: configure pricing in `config.pricing.MODEL_PRICING_USD_PER_1M` "
            "and `AUDIO_MODEL_PRICING_USD_PER_1M_CHAR`."
        )
    notes = cost_section.get("notes") if isinstance(cost_section, dict) else None
    if notes:
        st.write(notes)

    def _with_ext(name: str, ext: str) -> str:
        n = (name or "").strip()
        if not n:
            return f"run_report{ext}"
        if not n.lower().endswith(ext):
            return n + ext
        return n

    report_name = st.text_input(
        "Report file name",
        value="run_report.json",
        key="run_report_name",
        help="The suggested download name. Your browser decides the folder.",
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        f"⬇️ Download {report_name or 'run_report.json'}",
        data=payload.encode("utf-8"),
        file_name=_with_ext(report_name, ".json"),
        mime="application/json",
    )
