"""Main generation, preview, audio, and export UI for the app."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from core.export_anki import HAS_GENANKI, build_anki_package
from core.export_csv import generate_csv
from core.generation import GenerationSettings, generate_card
from core.llm_clients import create_client

from . import ui_helpers
from .sidebar import SidebarConfig
from .tts_panel import render_audio_panel


@dataclass
class AudioConfig:
    providers: Dict[str, Dict[str, Any]]
    default_provider: str


@dataclass
class ExportConfig:
    csv_delimiter: str
    csv_lineterminator: str
    anki_model_id: int
    anki_deck_id: int
    anki_model_name: str
    anki_deck_name: str
    front_template: str
    back_template: str
    css: str


@dataclass
class GenerationRunContext:
    client: Any
    max_tokens: Optional[int]
    temperature: Optional[float]
    allow_response_format: bool
    force_flagged: bool


def render_generation_page(
    settings: SidebarConfig,
    *,
    signalword_groups: Optional[Dict],
    signalwords_b1: List[str],
    signalwords_b2_plus: List[str],
    api_delay: float,
    audio_config: AudioConfig,
    export_config: ExportConfig,
) -> None:
    """Render generation controls, preview, audio panel, and exports."""

    if not st.session_state.input_data:
        return

    state = st.session_state
    state.setdefault("current_index", 0)
    state.setdefault("run_active", False)
    state.setdefault("auto_continue", False)

    total = len(state.input_data)
    processed = len(state.get("results", []))
    run_stats = state.get("run_stats") or {
        "batches": 0,
        "items": 0,
        "elapsed": 0.0,
        "errors": 0,
        "transient": 0,
        "start_ts": None,
    }
    state.run_stats = run_stats

    summary = st.empty()
    if run_stats["start_ts"]:
        total_elapsed = max(0.001, time.time() - run_stats["start_ts"])
        rate = run_stats["items"] / total_elapsed
        valid_now = sum(1 for card in state.get("results", []) if not card.get("error"))
        summary.caption(
            f"Run: batches {run_stats['batches']} • processed {processed}/{total} • valid {valid_now} • "
            f"elapsed {total_elapsed:.1f}s • {rate:.2f}/s • errors {run_stats['errors']} (transient {run_stats['transient']})"
        )
    else:
        valid_now = sum(1 for card in state.get("results", []) if not card.get("error"))
        summary.caption(f"Run: processed {processed}/{total} • valid {valid_now}")

    overall_caption = st.empty()
    overall = st.progress(0)
    overall.progress(min(1.0, processed / max(total, 1)))
    overall_caption.caption(f"Overall: {processed}/{total} processed")

    col_start, col_next, col_stop, col_rerun = st.columns([1, 1, 1, 1])
    start_run = col_start.button("Start run", type="primary")
    next_batch = col_next.button("Next batch")
    stop_run = col_stop.button("Stop run")
    rerun_errored = col_rerun.button("Re‑run errored only")

    def _process_batch() -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        run_ctx = _prepare_generation_run(state, settings)
        if run_ctx is None:
            return
        client = run_ctx.client
        ui_helpers.init_signalword_state()
        ui_helpers.init_response_format_cache()
        if not state.get("anki_run_id"):
            state.anki_run_id = str(int(time.time()))
        state.model_id = settings.model

        start_idx = int(state.current_index or 0)
        end_idx = min(start_idx + int(state.get("batch_size", 5)), total)
        if start_idx >= total:
            return
        indices = list(range(start_idx, end_idx))
        input_snapshot = list(state.input_data)

        def _worker(idx: int, row: dict) -> tuple[int, dict]:
            if not run_ctx.force_flagged and not row.get("_flag_ok", True):
                return idx, _flagged_card(row, idx, include_reason=True, preserve_fields=False)
            try:
                seed = random.randint(0, 2**31 - 1)
                gen_settings = _build_generation_settings(settings, run_ctx, seed)
                gen_result = generate_card(
                    client=client,
                    row=row,
                    settings=gen_settings,
                    signalword_groups=signalword_groups,
                    signalwords_b1=signalwords_b1,
                    signalwords_b2_plus=signalwords_b2_plus,
                    signal_usage=None,
                    signal_last=None,
                )
                card = gen_result.card
                meta = card.get("meta", {}) or {}
                meta["input_index"] = idx
                card["meta"] = meta
                return idx, card
            except Exception as exc:  # pragma: no cover
                return idx, _exception_card(exc, row, idx)

        workers = int(state.get("max_workers", 3))
        batch_header = st.empty()
        batch_header.caption(f"Batch {start_idx+1}–{end_idx} of {total} • size {len(indices)} • workers {workers}")
        batch_prog = st.progress(0)
        batch_status = st.empty()
        batch_start_ts = time.time()
        results_map: Dict[int, dict] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker, idx, input_snapshot[idx]): idx for idx in indices}
            for future in as_completed(futures):
                idx, card = future.result()
                results_map[idx] = card
                completed += 1
                batch_prog.progress(min(1.0, completed / max(len(indices), 1)))
                elapsed = max(0.001, time.time() - batch_start_ts)
                active = max(0, min(workers, len(indices) - completed))
                queued = max(0, len(indices) - completed - active)
                rate = completed / elapsed
                batch_status.caption(
                    f"Done {completed}/{len(indices)} • Active ~{active} • Queued ~{queued} • {elapsed:.1f}s • {rate:.2f}/s"
                )
                done_tasks = start_idx + completed
                overall.progress(min(1.0, done_tasks / max(total, 1)))
                overall_caption.caption(f"Overall: {done_tasks}/{total} processed")
                if api_delay > 0:
                    time.sleep(api_delay)

        usage = dict(state.get("sig_usage", {}))
        last = state.get("sig_last")
        batch_errors = 0
        batch_transient = 0
        for idx in indices:
            card = results_map.get(idx)
            if card is None:
                continue
            state.results.append(card)
            meta = card.get("meta", {}) or {}
            if meta.get("response_format_removed"):
                cache = set(state.get("no_response_format_models", set()))
                notified = set(state.get("no_response_format_notified", set()))
                if settings.model not in cache:
                    cache.add(settings.model)
                    state.no_response_format_models = cache
                if settings.model not in notified:
                    notified.add(settings.model)
                    state.no_response_format_notified = notified
                    detail = meta.get("response_format_error")
                    message = (
                        f"Model {settings.model} ignored schema (text.format); falling back to text parsing for this session."
                    )
                    if detail:
                        message += f"\nReason: {detail}"
                    st.info(message, icon="ℹ️")
            found = meta.get("signalword_found")
            if found:
                usage[found] = usage.get(found, 0) + 1
                last = found
            err_text = (card.get("error") or "").lower()
            if err_text:
                batch_errors += 1
                if any(code in err_text for code in ("429", "rate", "timeout", "502", "503")):
                    batch_transient += 1

        state.sig_usage = usage
        state.sig_last = last
        state.current_index = end_idx
        overall_count = len(state.results)
        overall.progress(min(1.0, overall_count / max(total, 1)))
        overall_caption.caption(f"Overall: {overall_count}/{total} processed")
        batch_elapsed = max(0.001, time.time() - batch_start_ts)
        batch_status.caption(f"Batch finished in {batch_elapsed:.1f}s • {len(indices)/batch_elapsed:.2f}/s")

        if not state.run_stats.get("start_ts"):
            state.run_stats["start_ts"] = batch_start_ts
        state.run_stats["batches"] += 1
        state.run_stats["items"] += len(indices)
        state.run_stats["elapsed"] += batch_elapsed
        state.run_stats["errors"] += batch_errors
        state.run_stats["transient"] += batch_transient
        total_elapsed = max(0.001, time.time() - state.run_stats["start_ts"])
        rate = state.run_stats["items"] / total_elapsed
        summary.caption(
            f"Run: batches {state.run_stats['batches']} • processed {overall_count}/{total} • "
            f"elapsed {total_elapsed:.1f}s • {rate:.2f}/s • errors {state.run_stats['errors']} "
            f"(transient {state.run_stats['transient']})"
        )

        if batch_transient >= 2 and state.get("max_workers", 3) > 1:
            state.max_workers = int(state.get("max_workers", 3)) - 1
            st.info(
                f"Transient errors detected ({batch_transient}); reducing max workers to {state.max_workers} for next batch.",
                icon="⚠️",
            )

    def _rerun_errored_only() -> None:
        err_indices = []
        for card in state.get("results", []):
            meta = card.get("meta", {}) or {}
            idx = meta.get("input_index")
            if card.get("error") and isinstance(idx, int):
                err_indices.append(idx)

        if not err_indices:
            st.info("No errored cards to re-run.")
            return

        run_ctx = _prepare_generation_run(state, settings)
        if run_ctx is None:
            return
        client = run_ctx.client

        def _worker(idx: int, row: dict) -> tuple[int, dict]:
            try:
                if not run_ctx.force_flagged and not row.get("_flag_ok", True):
                    return idx, _flagged_card(row, idx, include_reason=False, preserve_fields=True)
                seed = random.randint(0, 2**31 - 1)
                gen_settings = _build_generation_settings(settings, run_ctx, seed)
                gen_result = generate_card(
                    client=client,
                    row=row,
                    settings=gen_settings,
                    signalword_groups=signalword_groups,
                    signalwords_b1=signalwords_b1,
                    signalwords_b2_plus=signalwords_b2_plus,
                    signal_usage=None,
                    signal_last=None,
                )
                card = gen_result.card
                meta = card.get("meta", {}) or {}
                meta["input_index"] = idx
                card["meta"] = meta
                return idx, card
            except Exception as exc:  # pragma: no cover
                return idx, _exception_card(exc, row, idx)

        st.info(f"Re-running {len(err_indices)} errored items…")
        bar = st.progress(0)
        results_map: Dict[int, dict] = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(state.get("max_workers", 3))) as ex:
            futures = {ex.submit(_worker, idx, state.input_data[idx]): idx for idx in err_indices}
            done = 0
            for future in as_completed(futures):
                idx, card = future.result()
                results_map[idx] = card
                done += 1
                bar.progress(min(1.0, done / max(len(err_indices), 1)))

        new_results: List[dict] = []
        for card in state.get("results", []):
            meta = card.get("meta", {}) or {}
            idx = meta.get("input_index")
            if isinstance(idx, int) and idx in results_map:
                new_results.append(results_map[idx])
            else:
                new_results.append(card)
        state.results = new_results

        usage: Dict[str, int] = {}
        last = None
        for card in state.results:
            meta = card.get("meta", {}) or {}
            found = meta.get("signalword_found")
            if found:
                usage[found] = usage.get(found, 0) + 1
                last = found
        state.sig_usage = usage
        state.sig_last = last
        st.success("Errored items re-run completed.")

    if start_run:
        if not settings.api_key:
            st.error("Provide OPENAI_API_KEY via Secrets, environment variable, or the input field.")
        else:
            state.results = []
            state.audio_media = {}
            state.audio_summary = None
            state.current_index = 0
            state.run_stats = {"batches": 0, "items": 0, "elapsed": 0.0, "errors": 0, "transient": 0, "start_ts": None}
            state.run_active = True
            client = create_client(settings.api_key)
            if client is not None:
                ui_helpers.probe_response_format_support(client, settings.model)
            _process_batch()
            if state.get("auto_advance") and state.current_index < total:
                state.auto_continue = True
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
    elif next_batch:
        if not state.get("run_active"):
            state.run_active = True
        if not settings.api_key:
            st.error("Provide OPENAI_API_KEY before running batches.")
        else:
            _process_batch()
            if state.get("auto_advance") and state.current_index < total:
                state.auto_continue = True
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
    elif stop_run:
        state.run_active = False
        state.auto_continue = False
        st.info("Run paused. Use Next batch or Start run to continue.")
    elif rerun_errored:
        if not settings.api_key:
            st.error("Provide OPENAI_API_KEY before retrying errors.")
        else:
            _rerun_errored_only()

    if state.get("auto_continue") and state.get("run_active") and state.current_index < total:
        if not state.get("auto_advance"):
            state.auto_continue = False
        elif not settings.api_key:
            state.auto_continue = False
            state.run_active = False
            st.error("Provide OPENAI_API_KEY before running batches.")
        else:
            state.auto_continue = False
            _process_batch()
            if state.get("auto_advance") and state.current_index < total:
                state.auto_continue = True
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

    preview_container = st.container()
    st.divider()

    render_audio_panel(audio_config=audio_config, settings=settings)

    with preview_container:
        st.subheader("📋 Preview (all)")
        total_rows = len(state.results)
        total_errors = sum(1 for card in state.results if card.get("error"))
        total_valid = total_rows - total_errors
        st.caption(f"Preview: valid {total_valid} • errors {total_errors} • rows {total_rows}")
        filt_col1, filt_col2 = st.columns([1, 1])
        with filt_col1:
            st.checkbox(
                "Show only errors",
                value=state.get("preview_only_errors", False),
                key="preview_only_errors",
            )
        with filt_col2:
            next_err_click = st.button("Next error")

        if next_err_click and total_errors:
            err_indices = [idx for idx, card in enumerate(state.results) if card.get("error")]
            ptr = int(state.get("err_ptr", -1))
            candidates = [idx for idx in err_indices if idx > ptr]
            target = candidates[0] if candidates else err_indices[0]
            state.err_ptr = target
            target_card = state.results[target]
            st.warning(
                f"Next error at row {target+1}/{total_rows}: "
                f"{target_card.get('L2_word', '')} — {target_card.get('error', '')}"
            )

        if state.results:
            preview_cards = state.results
            if state.get("preview_only_errors"):
                preview_cards = [card for card in state.results if card.get("error")]
            preview_df = pd.DataFrame(preview_cards)
            st.dataframe(preview_df, width="stretch")
        else:
            st.info("No results yet — run a batch to populate preview.")

    st.divider()

    csv_extras = {
        "level": state.get("level", settings.level),
        "profile": state.get("prompt_profile", settings.profile),
        "model": state.get("model_id", settings.model),
        "L1": state.get("L1_code", settings.L1_code),
    }
    include_errored = st.sidebar.checkbox("Include errored cards in exports", value=False)
    export_cards = state.results if include_errored else [card for card in state.results if not card.get("error")]
    exportable_count = len(export_cards)
    saved_count = len(state.results)
    st.caption(
        f"Exportable cards: {exportable_count} / saved results: {saved_count} / total input: {len(state.input_data)}"
    )

    csv_data = generate_csv(
        export_cards,
        settings.L1_meta,
        delimiter=export_config.csv_delimiter,
        line_terminator=export_config.csv_lineterminator,
        include_header=state.get("csv_with_header", True),
        include_extras=True,
        anki_field_header=settings.csv_anki_header,
        extras_meta=csv_extras,
    )

    state.last_csv_data = csv_data
    st.download_button(
        label="📥 Download anki_cards.csv",
        data=csv_data,
        file_name="anki_cards.csv",
        mime="text/csv",
        key="download_csv",
    )

    if HAS_GENANKI:
        try:
            front_html = export_config.front_template.replace("{L1_LABEL}", settings.L1_meta["label"])
            tags_meta = {
                "level": state.get("level", settings.level),
                "profile": state.get("prompt_profile", settings.profile),
                "model": state.get("model_id", settings.model),
                "L1": state.get("L1_code", settings.L1_code),
            }
            anki_bytes = build_anki_package(
                export_cards,
                l1_label=settings.L1_meta["label"],
                guid_policy=state.get("anki_guid_policy", "stable"),
                run_id=state.get("anki_run_id", str(int(time.time()))),
                model_id=export_config.anki_model_id,
                model_name=export_config.anki_model_name,
                deck_id=export_config.anki_deck_id,
                deck_name=export_config.anki_deck_name,
                front_template=front_html,
                back_template=export_config.back_template,
                css=export_config.css,
                tags_meta=tags_meta,
                media_files=state.get("audio_media"),
            )
            state.last_anki_package = anki_bytes
            st.download_button(
                label="🧩 Download Anki deck (.apkg)",
                data=anki_bytes,
                file_name="dutch_cloze.apkg",
                mime="application/octet-stream",
                key="download_apkg",
            )
        except Exception as exc:
            st.error(f"Failed to build .apkg: {exc}")
    else:
        st.info("To enable .apkg export, add 'genanki' to requirements.txt and restart the app.")

    last_meta = (state.results or [{}])[-1].get("meta", {}) if state.results else {}
    req_dbg = last_meta.get("request") if isinstance(last_meta, dict) else None
    with st.expander("🐞 Debug: last model request", expanded=False):
        if req_dbg:
            st.json(req_dbg)
            st.caption(
                f"response_format_removed={last_meta.get('response_format_removed')} | "
                f"temperature_removed={last_meta.get('temperature_removed')}"
            )
            try:
                import openai as _openai  # type: ignore

                st.caption(f"openai SDK version: {_openai.__version__}")
            except Exception:
                pass
        else:
            st.caption("No recent request captured yet.")


def _prepare_generation_run(state: Any, settings: SidebarConfig) -> Optional[GenerationRunContext]:
    """Create a reusable generation context or report a missing SDK."""

    client = create_client(settings.api_key)
    if client is None:
        st.error("OpenAI SDK not available; install the openai package to continue.")
        return None

    max_tokens = 3000 if settings.limit_tokens else None
    temperature = settings.temperature if ui_helpers.should_pass_temperature(settings.model) else None
    no_rf_models = set(state.get("no_response_format_models", set()))
    force_schema = state.get("force_schema_checkbox", False)
    allow_response_format = settings.model not in no_rf_models or force_schema
    force_flagged = state.get("force_flagged", False)

    return GenerationRunContext(
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        allow_response_format=allow_response_format,
        force_flagged=force_flagged,
    )


def _build_generation_settings(
    settings: SidebarConfig, run_ctx: GenerationRunContext, seed: int
) -> GenerationSettings:
    """Construct GenerationSettings using shared runtime context."""

    return GenerationSettings(
        model=settings.model,
        L1_code=settings.L1_code,
        L1_name=settings.L1_meta["name"],
        level=settings.level,
        profile=settings.profile,
        temperature=run_ctx.temperature,
        max_output_tokens=run_ctx.max_tokens,
        allow_response_format=run_ctx.allow_response_format,
        signalword_seed=seed,
    )


def _flagged_card(
    row: Dict[str, Any],
    idx: int,
    *,
    include_reason: bool,
    preserve_fields: bool,
) -> Dict[str, Any]:
    """Return a placeholder card for flagged items."""

    card = {
        "L2_word": row.get("woord", ""),
        "L2_cloze": "",
        "L1_sentence": "",
        "L2_collocations": "",
        "L2_definition": "",
        "L1_gloss": "",
        "L1_hint": "",
        "AudioSentence": "",
        "AudioWord": "",
        "error": "flagged_precheck",
    }
    if preserve_fields:
        card["L2_definition"] = row.get("def_nl", "")
        card["L1_gloss"] = row.get("translation", "")

    meta: Dict[str, Any] = {"input_index": idx}
    if include_reason:
        meta["flag_reason"] = row.get("_flag_reason", "")
    card["meta"] = meta
    return card


def _exception_card(exc: Exception, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Produce a fallback card when generation raises an exception."""

    return {
        "L2_word": row.get("woord", ""),
        "L2_cloze": "",
        "L1_sentence": "",
        "L2_collocations": "",
        "L2_definition": row.get("def_nl", ""),
        "L1_gloss": row.get("translation", ""),
        "L1_hint": "",
        "AudioSentence": "",
        "AudioWord": "",
        "error": f"exception: {exc}",
        "meta": {"input_index": idx},
    }
