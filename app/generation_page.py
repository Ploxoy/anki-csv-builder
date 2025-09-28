"""Main generation, preview, audio, and export UI for the app."""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

from core.audio import ensure_audio_for_cards, fetch_elevenlabs_voices, sentence_for_tts
from core.export_anki import HAS_GENANKI, build_anki_package
from core.export_csv import generate_csv
from core.generation import GenerationSettings, generate_card
from core.llm_clients import create_client

from . import ui_helpers
from .sidebar import SidebarConfig


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

        client = OpenAI(api_key=settings.api_key)
        max_tokens = 3000 if settings.limit_tokens else None
        effective_temp = settings.temperature if ui_helpers.should_pass_temperature(settings.model) else None
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

        no_rf_models = set(state.get("no_response_format_models", set()))
        force_schema = state.get("force_schema_checkbox", False)
        force_flagged = state.get("force_flagged", False)

        def _make_settings(seed: int) -> GenerationSettings:
            return GenerationSettings(
                model=settings.model,
                L1_code=settings.L1_code,
                L1_name=settings.L1_meta["name"],
                level=settings.level,
                profile=settings.profile,
                temperature=effective_temp,
                max_output_tokens=max_tokens,
                allow_response_format=(settings.model not in no_rf_models or force_schema),
                signalword_seed=seed,
            )

        def _worker(idx: int, row: dict) -> tuple[int, dict]:
            if not force_flagged and not row.get("_flag_ok", True):
                return idx, {
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
                    "meta": {"flag_reason": row.get("_flag_reason", ""), "input_index": idx},
                }
            try:
                seed = random.randint(0, 2**31 - 1)
                gen_settings = _make_settings(seed)
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
                return idx, {
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

        client = OpenAI(api_key=settings.api_key)
        max_tokens = 3000 if settings.limit_tokens else None
        effective_temp = settings.temperature if ui_helpers.should_pass_temperature(settings.model) else None
        no_rf_models = set(state.get("no_response_format_models", set()))
        force_schema = state.get("force_schema_checkbox", False)
        force_flagged = state.get("force_flagged", False)

        def _make_settings(seed: int) -> GenerationSettings:
            return GenerationSettings(
                model=settings.model,
                L1_code=settings.L1_code,
                L1_name=settings.L1_meta["name"],
                level=settings.level,
                profile=settings.profile,
                temperature=effective_temp,
                max_output_tokens=max_tokens,
                allow_response_format=(settings.model not in no_rf_models or force_schema),
                signalword_seed=seed,
            )

        def _worker(idx: int, row: dict) -> tuple[int, dict]:
            try:
                if not force_flagged and not row.get("_flag_ok", True):
                    return idx, {
                        "error": "flagged_precheck",
                        "meta": {"input_index": idx},
                        "L2_word": row.get("woord", ""),
                        "L2_cloze": "",
                        "L1_sentence": "",
                        "L2_collocations": "",
                        "L2_definition": row.get("def_nl", ""),
                        "L1_gloss": row.get("translation", ""),
                        "L1_hint": "",
                        "AudioSentence": "",
                        "AudioWord": "",
                    }
                seed = random.randint(0, 2**31 - 1)
                gen_settings = _make_settings(seed)
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
                return idx, {
                    "error": f"exception: {exc}",
                    "meta": {"input_index": idx},
                    "L2_word": row.get("woord", ""),
                    "L2_cloze": "",
                    "L1_sentence": "",
                    "L2_collocations": "",
                    "L2_definition": row.get("def_nl", ""),
                    "L1_gloss": row.get("translation", ""),
                    "L1_hint": "",
                    "AudioSentence": "",
                    "AudioWord": "",
                }

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
            client = create_client(settings.api_key)  # noqa: F821 - dynamic import for probe only
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

    with st.expander("🔊 Audio (optional)", expanded=state.get("audio_panel_expanded", False)):
        state.audio_panel_expanded = True
        audio_summary = state.get("audio_summary")
        audio_cache = state.get("audio_cache", {})

        providers = audio_config.providers or {}
        provider_keys = list(providers.keys())
        default_provider = audio_config.default_provider

        def _provider_label(key: str) -> str:
            data = providers.get(key, {})
            if isinstance(data, dict):
                label = data.get("label")
                if isinstance(label, str) and label:
                    return label
            return key

        selected_provider = ""
        if not provider_keys:
            st.warning("No TTS providers configured in settings.")
        else:
            current_provider = state.get("audio_provider") or default_provider or provider_keys[0]
            if current_provider not in providers:
                current_provider = provider_keys[0]
            provider_index = provider_keys.index(current_provider) if current_provider in provider_keys else 0
            selected_provider = st.selectbox(
                "TTS provider",
                options=provider_keys,
                index=provider_index,
                format_func=_provider_label,
                key="audio_provider",
            )

        provider_data = providers.get(selected_provider, {}) if selected_provider else {}
        provider_type = str(provider_data.get("type", selected_provider or "")) if isinstance(provider_data, dict) else ""

        if provider_type == "elevenlabs":
            if "elevenlabs_api_key" not in state or not state.get("elevenlabs_api_key"):
                secret = ui_helpers.get_secret("ELEVENLABS_API_KEY")
                if secret:
                    state.elevenlabs_api_key = secret
            eleven_api_key = st.text_input(
                "ElevenLabs API Key",
                type="password",
                value=state.get("elevenlabs_api_key", ""),
                key="elevenlabs_api_key",
                help="Stored in ELEVENLABS_API_KEY env variable or Streamlit secrets.",
            )
        else:
            eleven_api_key = state.get("elevenlabs_api_key")

        dynamic_voice_list: List[Dict[str, str]] = []
        dynamic_voice_error: Optional[str] = None
        fetched_this_run = False
        if provider_type == "elevenlabs" and eleven_api_key:
            api_key_hash = hashlib.sha1(eleven_api_key.encode("utf-8")).hexdigest()
            cached_hash = state.get("elevenlabs_voice_catalog_key")
            cached_catalog = state.get("elevenlabs_voice_catalog")
            if not isinstance(cached_catalog, list) or cached_hash != api_key_hash:
                try:
                    dynamic_voice_list = fetch_elevenlabs_voices(
                        eleven_api_key,
                        language_codes=provider_data.get("voice_language_codes"),
                    )
                    state.elevenlabs_voice_catalog = dynamic_voice_list
                    state.elevenlabs_voice_catalog_key = api_key_hash
                    state.elevenlabs_voice_catalog_error = None
                    fetched_this_run = True
                except Exception as exc:  # pragma: no cover - network dependent
                    dynamic_voice_error = str(exc)
                    state.elevenlabs_voice_catalog = None
                    state.elevenlabs_voice_catalog_error = dynamic_voice_error
            else:
                dynamic_voice_list = cached_catalog
            if dynamic_voice_error is None:
                cached_error = state.get("elevenlabs_voice_catalog_error")
                if isinstance(cached_error, str) and cached_error:
                    dynamic_voice_error = cached_error
        else:
            state.elevenlabs_voice_catalog_error = None

        static_voices = provider_data.get("voices") if isinstance(provider_data, dict) else []
        voices_list = dynamic_voice_list if dynamic_voice_list else static_voices

        default_voice = str(provider_data.get("voice_default", "")) if isinstance(provider_data, dict) else ""
        if provider_type == "elevenlabs" and voices_list:
            first_voice = voices_list[0]
            if isinstance(first_voice, dict):
                default_voice = str(first_voice.get("id", "")) or default_voice
        voice_map = state.get("audio_voice_map")
        if not isinstance(voice_map, dict):
            voice_map = {}
            state.audio_voice_map = voice_map

        voice_ids = [voice.get("id") for voice in voices_list if isinstance(voice, dict) and voice.get("id")]

        if selected_provider and state.get("_audio_provider_last") != selected_provider:
            stored_voice = voice_map.get(selected_provider)
            if stored_voice and stored_voice in voice_ids:
                state.audio_voice = stored_voice
            elif default_voice and default_voice in voice_ids:
                state.audio_voice = default_voice
            elif voice_ids:
                state.audio_voice = voice_ids[0]
            state.audio_include_word = bool(provider_data.get("include_word_default", True))
            state.audio_include_sentence = bool(provider_data.get("include_sentence_default", True))
            sentence_default = provider_data.get("sentence_default", "") if isinstance(provider_data, dict) else ""
            word_default = provider_data.get("word_default", "") if isinstance(provider_data, dict) else ""
            if isinstance(sentence_default, str) and sentence_default:
                state.audio_sentence_instruction = sentence_default
            if isinstance(word_default, str) and word_default:
                state.audio_word_instruction = word_default
            state._audio_provider_last = selected_provider

        if fetched_this_run and provider_type == "elevenlabs" and voice_ids:
            current_voice = state.get("audio_voice")
            if current_voice not in voice_ids:
                state.audio_voice = voice_ids[0]

        if voice_ids:
            current_voice = state.get("audio_voice") or voice_ids[0]
            if current_voice not in voice_ids:
                current_voice = voice_ids[0]
            voice_widget_key = f"audio_voice__{selected_provider}" if selected_provider else "audio_voice__default"
            widget_default = current_voice
            if voice_widget_key not in st.session_state or st.session_state.get(voice_widget_key) not in voice_ids:
                st.session_state[voice_widget_key] = widget_default

            def _voice_label(vid: str) -> str:
                return next(
                    (voice.get("label", vid) for voice in voices_list if isinstance(voice, dict) and voice.get("id") == vid),
                    vid,
                )

            st.selectbox(
                "Voice",
                options=voice_ids,
                format_func=_voice_label,
                key=voice_widget_key,
            )
            selected_voice = st.session_state.get(voice_widget_key, voice_ids[0])
            state.audio_voice = selected_voice
            if selected_provider:
                voice_map[selected_provider] = selected_voice
        else:
            if selected_provider:
                st.warning("No voices configured for this provider.")
            selected_voice = ""

        if provider_type == "elevenlabs" and dynamic_voice_error:
            st.warning(f"Failed to load ElevenLabs voices: {dynamic_voice_error}")
        elif provider_type == "elevenlabs" and not dynamic_voice_list and static_voices:
            st.info("Showing default ElevenLabs voices because none matched the requested language.")

        max_audio_workers = st.slider(
            "Parallel audio workers",
            min_value=1,
            max_value=6,
            value=int(state.get("audio_workers", 3)),
            step=1,
            help="How many TTS requests to run in parallel.",
            key="audio_workers",
        )
        if provider_type == "elevenlabs" and max_audio_workers > 2:
            st.caption("ElevenLabs rate limits are strict — backend caps workers at 2 to avoid 429 errors.")

        include_word = st.checkbox(
            "Include word audio",
            value=state.get("audio_include_word"),
            key="audio_include_word",
        )
        include_sentence = st.checkbox(
            "Include sentence audio",
            value=state.get("audio_include_sentence"),
            key="audio_include_sentence",
        )

        sentence_styles_map = provider_data.get("sentence_styles") if isinstance(provider_data, dict) else {}
        word_styles_map = provider_data.get("word_styles") if isinstance(provider_data, dict) else {}

        def _style_label(style_map: Dict[str, Any], key: str) -> str:
            option = style_map.get(key) if isinstance(style_map, dict) else None
            if isinstance(option, dict):
                label = option.get("label")
                if isinstance(label, str) and label:
                    return label
            return key

        sentence_options = list(sentence_styles_map.keys()) if isinstance(sentence_styles_map, dict) else []
        if sentence_options:
            sentence_current = state.get("audio_sentence_instruction") or provider_data.get("sentence_default")
            if sentence_current not in sentence_options:
                sentence_current = provider_data.get("sentence_default") if provider_data.get("sentence_default") in sentence_options else sentence_options[0]
                state.audio_sentence_instruction = sentence_current
            sentence_choice = st.selectbox(
                "Sentence style",
                options=sentence_options,
                format_func=lambda key: _style_label(sentence_styles_map, key),
                key="audio_sentence_instruction",
            )
            sentence_caption = st.empty()
            sentence_caption.caption(
                str(sentence_styles_map.get(sentence_choice, {}).get("description", "")) or " "
            )
        else:
            sentence_choice = ""
            if selected_provider:
                st.info("No sentence styles configured for this provider.")

        word_options = list(word_styles_map.keys()) if isinstance(word_styles_map, dict) else []
        if word_options:
            word_current = state.get("audio_word_instruction") or provider_data.get("word_default")
            if word_current not in word_options:
                word_current = provider_data.get("word_default") if provider_data.get("word_default") in word_options else word_options[0]
                state.audio_word_instruction = word_current
            word_choice = st.selectbox(
                "Word style",
                options=word_options,
                format_func=lambda key: _style_label(word_styles_map, key),
                key="audio_word_instruction",
            )
            word_caption = st.empty()
            word_caption.caption(
                str(word_styles_map.get(word_choice, {}).get("description", "")) or " "
            )
        else:
            word_choice = ""
            if selected_provider:
                st.info("No word styles configured for this provider.")

        cards = state.results
        unique_words = set()
        unique_sentences = set()
        for card in cards:
            woord_text = (card.get("L2_word") or "").strip()
            if woord_text:
                unique_words.add(woord_text)
            sentence_text = sentence_for_tts(card.get("L2_cloze", ""))
            if sentence_text:
                unique_sentences.add(sentence_text)

        requests_estimate = 0
        if include_word:
            requests_estimate += len(unique_words)
        if include_sentence:
            requests_estimate += len(unique_sentences)

        st.caption(
            "Estimated requests: "
            f"{requests_estimate} (unique words — {len(unique_words)}, sentences — {len(unique_sentences)})."
        )

        if audio_summary:
            cache_hits = audio_summary.get("cache_hits", 0)
            fallback_hits = audio_summary.get("fallback_switches", 0)
            msg = (
                f"Done: words — {audio_summary.get('word_success', 0)}, "
                f"sentences — {audio_summary.get('sentence_success', 0)}."
            )
            if cache_hits:
                msg += f" Cache hits: {cache_hits}."
            if fallback_hits:
                msg += f" Fallback used: {fallback_hits}×."
            summary_provider_key = audio_summary.get("provider") or selected_provider
            summary_provider = providers.get(summary_provider_key, {}) if isinstance(providers, dict) else {}
            st.success(msg)
            st.caption(
                "Requests: {req} • Word skips: {w_skip} • Sentence skips: {s_skip}".format(
                    req=audio_summary.get("total_requests", 0),
                    w_skip=audio_summary.get("word_skipped", 0),
                    s_skip=audio_summary.get("sentence_skipped", 0),
                )
            )
            errors = audio_summary.get("errors") or []
            if errors:
                preview_err = "; ".join(errors[:3])
                if len(errors) > 3:
                    preview_err += " …"
                st.warning(f"Audio issues: {preview_err}")

            summary_sentence_styles = summary_provider.get("sentence_styles", {}) if isinstance(summary_provider, dict) else {}
            summary_word_styles = summary_provider.get("word_styles", {}) if isinstance(summary_provider, dict) else {}
            styles = []
            sent_key = audio_summary.get("sentence_instruction_key") or ""
            word_key = audio_summary.get("word_instruction_key") or ""
            if sent_key:
                styles.append(f"sentence: {_style_label(summary_sentence_styles, sent_key)}")
            if word_key:
                styles.append(f"word: {_style_label(summary_word_styles, word_key)}")
            if styles:
                st.caption("Styles → " + "; ".join(styles))
            provider_label = _provider_label(summary_provider_key) if summary_provider_key else _provider_label(selected_provider)
            if provider_label:
                st.caption(f"Provider → {provider_label}")
            if audio_summary.get("voice"):
                st.caption(f"Voice → {audio_summary['voice']}")

        button_disabled = requests_estimate == 0 or not selected_voice or not selected_provider
        instruction_payloads = {
            "sentence": sentence_styles_map.get(sentence_choice, {}).get("payload") if sentence_choice else None,
            "word": word_styles_map.get(word_choice, {}).get("payload") if word_choice else None,
        }
        instruction_keys = {"sentence": sentence_choice, "word": word_choice}

        generate_audio = st.button(
            "🔊 Generate audio",
            type="primary",
            disabled=button_disabled,
            key="generate_audio_button",
        )

        if generate_audio:
            if button_disabled:
                st.info("No text to synthesize — enable word or sentence above.")
            elif provider_type == "openai" and not settings.api_key:
                st.error("OPENAI_API_KEY is required for OpenAI TTS synthesis.")
            elif provider_type == "elevenlabs" and not eleven_api_key:
                st.error("Provide ELEVENLABS_API_KEY (environment, secrets, or field above) for ElevenLabs TTS.")
            else:
                openai_client = OpenAI(api_key=settings.api_key) if provider_type == "openai" else None
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0.0)

                def _progress(done: int, total_requested: int) -> None:
                    if total_requested <= 0:
                        pct = 0.0
                    else:
                        pct = min(1.0, done / total_requested)
                    progress_bar.progress(pct)
                    status_placeholder.text(
                        f"Audio progress: {done}/{total_requested} ({pct * 100:.0f}%)"
                        if total_requested > 0
                        else "Audio progress: 0/0 (0%)"
                    )

                try:
                    media_map, summary_obj = ensure_audio_for_cards(
                        state.results,
                        provider=provider_type,
                        voice=selected_voice,
                        include_word=include_word,
                        include_sentence=include_sentence,
                        cache=audio_cache,
                        progress_cb=_progress,
                        instruction_payloads=instruction_payloads,
                        instruction_keys=instruction_keys,
                        max_workers=int(state.get("audio_workers", 3)),
                        openai_client=openai_client,
                        openai_model=str(provider_data.get("model")) if provider_type == "openai" and provider_data.get("model") else None,
                        openai_fallback_model=str(provider_data.get("fallback_model")) if provider_type == "openai" and provider_data.get("fallback_model") else None,
                        eleven_api_key=eleven_api_key if provider_type == "elevenlabs" else None,
                        eleven_model=str(provider_data.get("model")) if provider_type == "elevenlabs" and provider_data.get("model") else None,
                    )
                    state.audio_media = media_map
                    state.audio_summary = asdict(summary_obj)

                    success_msg = (
                        f"Done: words — {summary_obj.word_success}, "
                        f"sentences — {summary_obj.sentence_success}."
                    )
                    if summary_obj.cache_hits:
                        success_msg += f" Cache hits: {summary_obj.cache_hits}."
                    if summary_obj.fallback_switches:
                        success_msg += f" Fallback used: {summary_obj.fallback_switches}×."
                    styles_now = []
                    if summary_obj.sentence_instruction_key:
                        styles_now.append(f"sentence: {_style_label(sentence_styles_map, summary_obj.sentence_instruction_key)}")
                    if summary_obj.word_instruction_key:
                        styles_now.append(f"word: {_style_label(word_styles_map, summary_obj.word_instruction_key)}")
                    if styles_now:
                        success_msg += " | Styles → " + "; ".join(styles_now)
                    provider_label = _provider_label(summary_obj.provider or provider_type)
                    if provider_label:
                        success_msg += f" | Provider → {provider_label}"
                    st.success(success_msg)
                    if summary_obj.errors:
                        err_preview = "; ".join(summary_obj.errors[:3])
                        if len(summary_obj.errors) > 3:
                            err_preview += " …"
                        st.warning(f"Audio issues: {err_preview}")
                    st.caption(
                        "Requests: {req} • Word skips: {w_skip} • Sentence skips: {s_skip}".format(
                            req=summary_obj.total_requests,
                            w_skip=summary_obj.word_skipped,
                            s_skip=summary_obj.sentence_skipped,
                        )
                    )
                    state.results = [dict(card) for card in state.results]
                    status_placeholder.text(
                        f"Audio progress: {summary_obj.total_requests}/{summary_obj.total_requests} (100%)"
                        if summary_obj.total_requests
                        else "Audio progress: 0/0 (0%)"
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Audio synthesis failed: {exc}")
                    status_placeholder.text("Audio generation failed.")
                finally:
                    progress_placeholder.empty()

        if st.button("Hide audio options", key="hide_audio_panel"):
            state.audio_panel_expanded = False

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
