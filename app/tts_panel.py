"""Audio (TTS) panel rendering for the Streamlit app."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st
from openai import OpenAI

from core.audio import ensure_audio_for_cards, fetch_elevenlabs_voices, sentence_for_tts

from . import ui_helpers

try:  # pragma: no cover - optional import for type hints only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - Python <3.11 fallback
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - to avoid circular import at runtime
    from .sidebar import SidebarConfig
    from .ui_models import AudioConfig


def _provider_label(providers: Dict[str, Dict[str, Any]], key: str) -> str:
    data = providers.get(key, {})
    if isinstance(data, dict):
        label = data.get("label")
        if isinstance(label, str) and label:
            return label
    return key


def _compute_unique_texts(cards: List[Dict[str, Any]], include_word: bool, include_sentence: bool) -> Dict[str, Any]:
    unique_words: set[str] = set()
    unique_sentences: set[str] = set()
    for card in cards:
        woord_text = (card.get("L2_word") or "").strip()
        if include_word and woord_text:
            unique_words.add(woord_text)
        if include_sentence:
            sentence_text = sentence_for_tts(card.get("L2_cloze", ""))
            if sentence_text:
                unique_sentences.add(sentence_text)
    return {
        "unique_words": unique_words,
        "unique_sentences": unique_sentences,
        "requests": (len(unique_words) if include_word else 0)
        + (len(unique_sentences) if include_sentence else 0),
    }


def _voice_label(voices_list: List[Dict[str, Any]], vid: str) -> str:
    return next(
        (voice.get("label", vid) for voice in voices_list if isinstance(voice, dict) and voice.get("id") == vid),
        vid,
    )


def _elevenlabs_cache_key(api_key: str, language_codes: Optional[Sequence[str]]) -> str:
    normalized_codes = ",".join(sorted(code.strip().lower() for code in (language_codes or []) if code))
    key_hash = hashlib.sha1(api_key.encode("utf-8")).hexdigest()
    return f"{key_hash}:{normalized_codes or 'all'}"


def _ensure_state_dict(state: Any, key: str) -> Dict[str, Any]:
    data = state.get(key)
    if not isinstance(data, dict):
        data = {}
        state[key] = data
    return data


_ELEVENLABS_API_WIDGET_KEY = "elevenlabs_api_key_input"


def _refresh_elevenlabs_voice_catalog(
    *,
    api_key: str,
    provider_data: Dict[str, Any],
    cache_key: str,
    voice_cache: Dict[str, Any],
    voice_errors: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[str]]:
    """Fetch voices from ElevenLabs and update cache structures."""

    language_codes = provider_data.get("voice_language_codes")
    try:
        fetched_voices = fetch_elevenlabs_voices(api_key, language_codes=language_codes)
    except Exception as exc:  # pragma: no cover - network dependent
        voice_errors[cache_key] = str(exc)
        return [], None, voice_errors[cache_key]

    timestamp = time.time()
    voice_cache[cache_key] = {
        "voices": fetched_voices,
        "updated_at": timestamp,
        "language_codes": list(language_codes or []),
    }
    voice_errors.pop(cache_key, None)
    return fetched_voices, timestamp, None


def render_audio_panel(
    *,
    audio_config: "AudioConfig",
    settings: "SidebarConfig",
) -> None:
    """Render the audio expander with provider + voice selection and synthesis."""

    state = st.session_state
    providers = audio_config.providers or {}
    provider_keys = list(providers.keys())
    default_provider = audio_config.default_provider

    secret_elevenlabs = ui_helpers.get_secret("ELEVENLABS_API_KEY")

    with st.expander("🔊 Audio (optional)", expanded=state.get("audio_panel_expanded", False)):
        state.audio_panel_expanded = True
        audio_summary = state.get("audio_summary")
        audio_cache = state.get("audio_cache", {})

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
                format_func=lambda key: _provider_label(providers, key),
                key="audio_provider",
            )

        provider_data = providers.get(selected_provider, {}) if selected_provider else {}
        provider_type = str(provider_data.get("type", selected_provider or "")) if isinstance(provider_data, dict) else ""

        prev_provider = state.get("_audio_provider_last")
        provider_changed = bool(selected_provider) and selected_provider != prev_provider
        if provider_changed and prev_provider:
            voice_map_prev = state.get("audio_voice_map")
            if not isinstance(voice_map_prev, dict):
                voice_map_prev = {}
            prev_voice_value = state.get("audio_voice")
            if prev_voice_value:
                voice_map_prev[prev_provider] = prev_voice_value
            state.audio_voice_map = voice_map_prev
            st.session_state.pop(f"audio_voice__{selected_provider}", None)

        if provider_type == "elevenlabs":
            stored_key = (state.get("elevenlabs_api_key", "") or "").strip()
            if not stored_key and secret_elevenlabs:
                secret_clean = secret_elevenlabs.strip()
                if secret_clean:
                    state.elevenlabs_api_key = secret_clean
                    stored_key = secret_clean
                    st.caption("Using ELEVENLABS_API_KEY from secrets.")

            if stored_key:
                st.caption("ElevenLabs key stored for this session.")
                placeholder = "Paste a new ElevenLabs API key to replace the stored value."
            else:
                placeholder = "Paste your ElevenLabs API key to enable this provider."

            widget_value = state.get(_ELEVENLABS_API_WIDGET_KEY)
            if stored_key and not widget_value:
                state[_ELEVENLABS_API_WIDGET_KEY] = stored_key

            manual_entry = st.text_input(
                "ElevenLabs API Key",
                type="password",
                key=_ELEVENLABS_API_WIDGET_KEY,
                help="Stored in ELEVENLABS_API_KEY env variable or enter manually for this session.",
                placeholder=placeholder,
            )
            manual_entry = (manual_entry or "").strip()
            if manual_entry:
                state.elevenlabs_api_key = manual_entry
                stored_key = manual_entry
                st.caption("ElevenLabs key stored for this session.")

            eleven_api_key = stored_key
        else:
            eleven_api_key = state.get("elevenlabs_api_key")

        dynamic_voice_list: List[Dict[str, Any]] = []
        dynamic_voice_error: Optional[str] = None
        last_loaded_at: Optional[float] = None
        voices_list: List[Dict[str, Any]] = []

        if provider_type == "elevenlabs":
            voice_cache = _ensure_state_dict(state, "elevenlabs_voice_catalog_cache")
            voice_errors = _ensure_state_dict(state, "elevenlabs_voice_catalog_errors")
            voice_meta = _ensure_state_dict(state, "elevenlabs_voice_catalog_meta")
            cache_key = (
                _elevenlabs_cache_key(eleven_api_key, provider_data.get("voice_language_codes"))
                if eleven_api_key
                else ""
            )

            cached_entry = voice_cache.get(cache_key) if cache_key else None
            if isinstance(cached_entry, dict):
                cached_voices = cached_entry.get("voices")
                if isinstance(cached_voices, list):
                    dynamic_voice_list = cached_voices
                updated_at = cached_entry.get("updated_at")
                if isinstance(updated_at, (int, float)):
                    last_loaded_at = float(updated_at)

            dynamic_voice_error = voice_errors.get(cache_key) if cache_key else None

            loading_flag = bool(state.get("elevenlabs_voice_catalog_loading"))
            button_label = "Load ElevenLabs voices" if not dynamic_voice_list else "Refresh ElevenLabs voices"
            load_button = st.button(
                button_label,
                key="elevenlabs_voice_catalog_trigger",
                disabled=loading_flag or not eleven_api_key,
                help="Fetch the latest ElevenLabs voices for the provided API key.",
            )

            meta_entry = voice_meta.get(cache_key, {}) if cache_key else {}
            should_autoload = (
                bool(eleven_api_key)
                and bool(cache_key)
                and not dynamic_voice_list
                and not loading_flag
                and not meta_entry.get("autoloaded")
            )

            fetch_reason: Optional[str] = None
            if load_button and not eleven_api_key:
                ui_helpers.toast(
                    "Provide ELEVENLABS_API_KEY to load ElevenLabs voices.",
                    icon="⚠️",
                    variant="warning",
                )
            elif load_button and eleven_api_key:
                fetch_reason = "manual"
            elif should_autoload:
                fetch_reason = "auto"

            if fetch_reason and eleven_api_key and cache_key:
                state.elevenlabs_voice_catalog_loading = True
                with st.spinner("Loading ElevenLabs voices…"):
                    fetched_voices, refreshed_at, fetch_error = _refresh_elevenlabs_voice_catalog(
                        api_key=eleven_api_key,
                        provider_data=provider_data,
                        cache_key=cache_key,
                        voice_cache=voice_cache,
                        voice_errors=voice_errors,
                    )
                state.elevenlabs_voice_catalog_loading = False

                if fetch_error:
                    dynamic_voice_error = fetch_error
                else:
                    dynamic_voice_list = fetched_voices
                    last_loaded_at = refreshed_at

                meta_update = dict(meta_entry)
                if fetch_reason == "auto":
                    meta_update["autoloaded"] = True
                meta_update["last_attempt"] = time.time()
                meta_update["last_status"] = "error" if fetch_error else "success"
                voice_meta[cache_key] = meta_update

            if not dynamic_voice_list and eleven_api_key and not loading_flag and dynamic_voice_error is None:
                st.caption('Click "Load ElevenLabs voices" to fetch the voice catalogue.')

            if last_loaded_at:
                formatted_time = time.strftime("%H:%M:%S", time.localtime(last_loaded_at))
                if dynamic_voice_error:
                    st.caption(
                        f"Using cached ElevenLabs voices from {formatted_time}; last refresh failed."
                    )
                else:
                    st.caption(f"ElevenLabs voices cached at {formatted_time}.")

            voices_list = dynamic_voice_list if dynamic_voice_list else []
        else:
            voices_list = []

        static_voices = provider_data.get("voices") if isinstance(provider_data, dict) else []
        if not voices_list and static_voices:
            voices_list = static_voices

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

        if selected_provider and provider_changed:
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

        if voice_ids:
            current_voice = state.get("audio_voice") or voice_ids[0]
            if current_voice not in voice_ids:
                current_voice = voice_ids[0]
            voice_widget_key = (
                f"audio_voice__{selected_provider}" if selected_provider else "audio_voice__default"
            )
            widget_default = current_voice
            if voice_widget_key not in st.session_state or st.session_state.get(voice_widget_key) not in voice_ids:
                st.session_state[voice_widget_key] = widget_default

            st.selectbox(
                "Voice",
                options=voice_ids,
                format_func=lambda vid: _voice_label(voices_list, vid),
                key=voice_widget_key,
            )
            selected_voice = st.session_state.get(voice_widget_key, voice_ids[0])
            state.audio_voice = selected_voice
            if selected_provider:
                voice_map[selected_provider] = selected_voice
        else:
            if provider_type == "elevenlabs":
                st.info(
                    'Provide ELEVENLABS_API_KEY above and click "Load ElevenLabs voices" to fetch the catalogue.'
                )
            elif selected_provider:
                st.info("No voices available for this provider yet.")
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

        sentence_options = list(sentence_styles_map.keys()) if isinstance(sentence_styles_map, dict) else []
        if sentence_options:
            sentence_current = state.get("audio_sentence_instruction") or provider_data.get("sentence_default")
            if sentence_current not in sentence_options:
                sentence_current = (
                    provider_data.get("sentence_default")
                    if provider_data.get("sentence_default") in sentence_options
                    else sentence_options[0]
                )
                state.audio_sentence_instruction = sentence_current
            sentence_choice = st.selectbox(
                "Sentence style",
                options=sentence_options,
                format_func=lambda key: sentence_styles_map.get(key, {}).get("label", key),
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
                word_current = (
                    provider_data.get("word_default")
                    if provider_data.get("word_default") in word_options
                    else word_options[0]
                )
                state.audio_word_instruction = word_current
            word_choice = st.selectbox(
                "Word style",
                options=word_options,
                format_func=lambda key: word_styles_map.get(key, {}).get("label", key),
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
        unique_info = _compute_unique_texts(cards, include_word, include_sentence)
        st.caption(
            "Estimated requests: "
            f"{unique_info['requests']} (unique words — {len(unique_info['unique_words'])}, "
            f"sentences — {len(unique_info['unique_sentences'])})."
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
                styles.append(
                    f"sentence: {summary_sentence_styles.get(sent_key, {}).get('label', sent_key)}"
                )
            if word_key:
                styles.append(f"word: {summary_word_styles.get(word_key, {}).get('label', word_key)}")
            if styles:
                st.caption("Styles → " + "; ".join(styles))
            provider_label = _provider_label(providers, summary_provider_key) if summary_provider_key else _provider_label(providers, selected_provider)
            if provider_label:
                st.caption(f"Provider → {provider_label}")
            if audio_summary.get("voice"):
                st.caption(f"Voice → {audio_summary['voice']}")

        button_disabled = unique_info["requests"] == 0 or not selected_voice or not selected_provider
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
                    pct = (done / total_requested) if total_requested > 0 else 0.0
                    progress_bar.progress(min(1.0, pct))
                    status_placeholder.text(
                        f"Audio progress: {done}/{total_requested} ({pct * 100:.0f}%)" if total_requested > 0 else "Audio progress: 0/0 (0%)"
                    )

                try:
                    media_map, summary_obj = ensure_audio_for_cards(
                        st.session_state.results,
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
                        styles_now.append(
                            f"sentence: {sentence_styles_map.get(summary_obj.sentence_instruction_key, {}).get('label', summary_obj.sentence_instruction_key)}"
                        )
                    if summary_obj.word_instruction_key:
                        styles_now.append(
                            f"word: {word_styles_map.get(summary_obj.word_instruction_key, {}).get('label', summary_obj.word_instruction_key)}"
                        )
                    if styles_now:
                        success_msg += " | Styles → " + "; ".join(styles_now)
                    provider_label = _provider_label(providers, summary_obj.provider or provider_type)
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
