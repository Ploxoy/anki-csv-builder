"""Audio (TTS) panel rendering for the Streamlit app."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
from openai import OpenAI

from core.audio import ensure_audio_for_cards, sentence_for_tts

from . import audio_catalog, ui_helpers
from .audio_state import AudioPanelState

try:  # pragma: no cover - optional import for type hints only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - Python <3.11 fallback
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - to avoid circular import at runtime
    from .sidebar import SidebarConfig
    from .ui_models import AudioConfig


_ELEVENLABS_API_WIDGET_KEY = "elevenlabs_api_key_input"


def _provider_label(providers: Dict[str, Dict[str, Any]], key: str) -> str:
    data = providers.get(key, {})
    if isinstance(data, dict):
        label = data.get("label")
        if isinstance(label, str) and label:
            return label
    return key


def _voice_label(voices_list: List[Dict[str, Any]], vid: str) -> str:
    return next(
        (voice.get("label", vid) for voice in voices_list if isinstance(voice, dict) and voice.get("id") == vid),
        vid,
    )


def _compute_unique_texts(cards: Sequence[Dict[str, Any]], include_word: bool, include_sentence: bool) -> Dict[str, Any]:
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


def render_audio_panel(
    *,
    audio_config: "AudioConfig",
    settings: "SidebarConfig",
) -> None:
    """Render the audio expander with provider + voice selection and synthesis."""

    state = AudioPanelState(st.session_state)
    providers = audio_config.providers or {}
    provider_keys = list(providers.keys())
    secret_elevenlabs = ui_helpers.get_secret("ELEVENLABS_API_KEY")

    with st.expander("🔊 Audio (optional)", expanded=state.is_expanded()):
        state.expand()
        nonce = state.nonce()

        if not provider_keys:
            st.warning("No TTS providers configured in settings.")
            _render_hide_button(state)
            return

        (
            selected_provider,
            provider_data,
            provider_type,
            provider_changed,
        ) = _render_provider_selector(
            state=state,
            providers=providers,
            provider_keys=provider_keys,
            default_provider=audio_config.default_provider,
            nonce=nonce,
        )

        if not selected_provider:
            _render_hide_button(state)
            return

        if provider_type == "elevenlabs":
            # Respect a feature flag: only load dynamic voices when enabled.
            if bool(provider_data.get("dynamic_voices")):
                eleven_api_key, key_changed = _render_elevenlabs_credentials(state, secret_elevenlabs)
                if key_changed:
                    nonce = state.nonce()
                catalog, refreshed = _resolve_elevenlabs_catalog(state, provider_data, eleven_api_key)
                if refreshed:
                    nonce = state.nonce()
                dynamic_voices = catalog.voices
                dynamic_error = catalog.error
                catalog_updated_at = catalog.updated_at
            else:
                eleven_api_key = state.get_elevenlabs_key()
                dynamic_voices = []
                dynamic_error = None
                catalog_updated_at = None
                st.caption("Using preset ElevenLabs voices (online catalogue disabled).")
        else:
            eleven_api_key = state.get_elevenlabs_key()
            dynamic_voices = []
            dynamic_error = None
            catalog_updated_at = None

        voices_list, used_static_fallback = _merge_voice_lists(
            provider_type=provider_type,
            provider_data=provider_data,
            dynamic_voices=dynamic_voices,
        )

        if provider_type == "elevenlabs" and catalog_updated_at:
            formatted = time.strftime("%H:%M:%S", time.localtime(catalog_updated_at))
            if dynamic_error:
                st.caption(f"Using cached ElevenLabs voices from {formatted}; last refresh failed.")
            else:
                st.caption(f"ElevenLabs voices cached at {formatted}.")

        selected_voice = _render_voice_selector(
            state=state,
            provider_key=selected_provider,
            provider_data=provider_data,
            provider_type=provider_type,
            provider_changed=provider_changed,
            voices=voices_list,
            catalog_error=dynamic_error,
            used_static_fallback=used_static_fallback,
            nonce=state.nonce(),
        )

        max_workers = _render_worker_slider(state, provider_type)
        include_word, include_sentence = _render_include_toggles(state)
        (
            sentence_choice,
            word_choice,
            sentence_styles_map,
            word_styles_map,
        ) = _render_style_controls(provider_data, state)

        cards = state.results()
        unique_info = _compute_unique_texts(cards, include_word, include_sentence)
        _render_estimate(unique_info)

        audio_summary = state.audio_summary()
        _render_summary(audio_summary, providers, selected_provider)

        instruction_payloads = {
            "sentence": sentence_styles_map.get(sentence_choice, {}).get("payload") if sentence_choice else None,
            "word": word_styles_map.get(word_choice, {}).get("payload") if word_choice else None,
        }
        instruction_keys = {"sentence": sentence_choice, "word": word_choice}

        _render_generate_button(
            state=state,
            settings=settings,
            provider_type=provider_type,
            provider_data=provider_data,
            selected_voice=selected_voice,
            include_word=include_word,
            include_sentence=include_sentence,
            unique_info=unique_info,
            instruction_payloads=instruction_payloads,
            instruction_keys=instruction_keys,
            sentence_styles_map=sentence_styles_map,
            word_styles_map=word_styles_map,
            eleven_api_key=eleven_api_key,
            max_workers=max_workers,
            providers=providers,
        )

        _render_hide_button(state)


def _render_provider_selector(
    *,
    state: AudioPanelState,
    providers: Dict[str, Dict[str, Any]],
    provider_keys: Sequence[str],
    default_provider: Optional[str],
    nonce: int,
) -> tuple[str, Dict[str, Any], str, bool]:
    current_provider = state.preferred_provider(provider_keys, default_provider)
    widget_key = state.widget_key("audio_provider", nonce)
    provider_index = provider_keys.index(current_provider)
    previous_voice = state.current_voice()

    selected_provider = st.selectbox(
        "TTS provider",
        options=provider_keys,
        index=provider_index,
        format_func=lambda key: _provider_label(providers, key),
        key=widget_key,
    )

    provider_changed = state.handle_provider_change(selected_provider, previous_voice)
    if provider_changed:
        voice_widget_key = state.widget_key(state.voice_widget_base(selected_provider), nonce)
        state.clear_widget(voice_widget_key)

    state.set_provider(selected_provider)

    provider_data = providers.get(selected_provider, {}) if selected_provider else {}
    if not isinstance(provider_data, dict):
        provider_data = {}
    provider_type = str(provider_data.get("type", selected_provider or ""))

    return selected_provider, provider_data, provider_type, provider_changed


def _render_elevenlabs_credentials(
    state: AudioPanelState,
    secret: Optional[str],
) -> tuple[str, bool]:
    stored_key = state.get_elevenlabs_key()
    if stored_key:
        state.seed_api_key_snapshot(stored_key)

    nonce_changed = False

    def _apply_key(value: str) -> None:
        nonlocal stored_key, nonce_changed
        state.set_elevenlabs_key(value)
        stored_key = value
        if state.update_api_key_snapshot(value):
            state.bump_nonce()
            nonce_changed = True

    secret_loaded = False
    if not stored_key and secret:
        secret_clean = secret.strip()
        if secret_clean:
            _apply_key(secret_clean)
            secret_loaded = True

    placeholder = (
        "Paste a new ElevenLabs API key to replace the stored value."
        if stored_key
        else "Paste your ElevenLabs API key to enable this provider."
    )

    manual_entry = st.text_input(
        "ElevenLabs API Key",
        type="password",
        key=_ELEVENLABS_API_WIDGET_KEY,
        help="Stored in ELEVENLABS_API_KEY env variable or enter manually for this session.",
        placeholder=placeholder,
    )
    manual_entry = (manual_entry or "").strip()

    if manual_entry:
        _apply_key(manual_entry)
        st.caption("ElevenLabs key stored for this session.")
    elif stored_key:
        message = "ElevenLabs key stored for this session."
        if secret_loaded or (secret and stored_key == secret.strip()):
            message += " Loaded from Streamlit secrets."
        st.caption(message)

    return stored_key, nonce_changed


def _resolve_elevenlabs_catalog(
    state: AudioPanelState,
    provider_data: Dict[str, Any],
    api_key: str,
) -> tuple[audio_catalog.ElevenLabsCatalog, bool]:
    language_codes = provider_data.get("voice_language_codes") if isinstance(provider_data, dict) else None
    cache_key = audio_catalog.elevenlabs_cache_key(api_key, language_codes)
    catalog = audio_catalog.get_catalog(state.store, cache_key)
    loading = audio_catalog.is_loading(state.store)
    nonce_changed = False

    button_label = "Load ElevenLabs voices" if not catalog.voices else "Refresh ElevenLabs voices"
    load_button = st.button(
        button_label,
        key="elevenlabs_voice_catalog_trigger",
        disabled=loading or not api_key,
        help="Fetch the latest ElevenLabs voices for the provided API key.",
    )

    if load_button and not api_key:
        ui_helpers.toast(
            "Provide ELEVENLABS_API_KEY to load ElevenLabs voices.",
            icon="⚠️",
            variant="warning",
        )
    elif load_button and api_key and cache_key:
        cached_voices = bool(catalog.voices)
        if cached_voices and audio_catalog.should_throttle(state.store, cache_key):
            ui_helpers.toast(
                "Reusing cached ElevenLabs voices — recent refresh already completed.",
                icon="ℹ️",
                variant="info",
            )
        else:
            audio_catalog.record_attempt(state.store, cache_key)
            audio_catalog.set_loading(state.store, True)
            try:
                with st.spinner("Loading ElevenLabs voices…"):
                    catalog = audio_catalog.refresh_catalog(
                        state.store,
                        cache_key=cache_key,
                        api_key=api_key,
                        language_codes=language_codes,
                    )
            finally:
                audio_catalog.set_loading(state.store, False)
            state.bump_nonce()
            nonce_changed = True

    catalog = audio_catalog.get_catalog(state.store, cache_key)
    loading = audio_catalog.is_loading(state.store)

    if not catalog.voices and api_key and not loading and catalog.error is None:
        st.caption('Click "Load ElevenLabs voices" to fetch the voice catalogue.')

    return catalog, nonce_changed


def _merge_voice_lists(
    *,
    provider_type: str,
    provider_data: Dict[str, Any],
    dynamic_voices: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], bool]:
    static_voices = provider_data.get("voices") if isinstance(provider_data, dict) else []
    static_list = static_voices if isinstance(static_voices, list) else []

    if provider_type == "elevenlabs":
        if dynamic_voices:
            return list(dynamic_voices), False
        if static_list:
            return list(static_list), True
        return [], False

    return list(static_list), False


def _render_voice_selector(
    *,
    state: AudioPanelState,
    provider_key: str,
    provider_data: Dict[str, Any],
    provider_type: str,
    provider_changed: bool,
    voices: List[Dict[str, Any]],
    catalog_error: Optional[str],
    used_static_fallback: bool,
    nonce: int,
) -> str:
    voice_ids = [voice.get("id") for voice in voices if isinstance(voice, dict) and voice.get("id")]

    if provider_changed or (state.current_voice() and state.current_voice() not in voice_ids):
        state.apply_provider_defaults(provider_key, provider_data, voice_ids)

    widget_base = state.voice_widget_base(provider_key)
    widget_key = state.widget_key(widget_base, nonce)

    if voice_ids:
        current_voice = state.current_voice()
        if current_voice not in voice_ids:
            current_voice = voice_ids[0]
            state.set_current_voice(current_voice)

        if widget_key not in st.session_state or st.session_state.get(widget_key) not in voice_ids:
            st.session_state[widget_key] = current_voice

        st.selectbox(
            "Voice",
            options=voice_ids,
            format_func=lambda vid: _voice_label(voices, vid),
            key=widget_key,
        )
        selected_voice = st.session_state.get(widget_key, voice_ids[0])
        state.set_current_voice(selected_voice)
        state.update_voice_map(provider_key, selected_voice)
    else:
        state.clear_widget(widget_key)
        state.clear_current_voice(provider_key)
        if provider_type == "elevenlabs":
            st.warning(
                'No ElevenLabs voices available. Check your API key or plan, then press "Load ElevenLabs voices".'
            )
        elif provider_key:
            st.warning("No TTS voices available for this provider. Configure voices and refresh to continue.")

    if provider_type == "elevenlabs" and catalog_error:
        st.warning(f"Failed to load ElevenLabs voices: {catalog_error}")
    elif provider_type == "elevenlabs" and used_static_fallback:
        st.info("Showing default ElevenLabs voices because none matched the requested language.")

    return state.current_voice()


def _render_worker_slider(state: AudioPanelState, provider_type: str) -> int:
    default_workers = int(state.store.get("audio_workers", 3))
    max_workers = st.slider(
        "Parallel audio workers",
        min_value=1,
        max_value=6,
        value=default_workers,
        step=1,
        help="How many TTS requests to run in parallel.",
        key="audio_workers",
    )
    if provider_type == "elevenlabs" and max_workers > 2:
        st.caption("ElevenLabs rate limits are strict — backend caps workers at 2 to avoid 429 errors.")
    return max_workers


def _render_include_toggles(state: AudioPanelState) -> tuple[bool, bool]:
    include_word = st.checkbox(
        "Include word audio",
        value=bool(state.store.get("audio_include_word", True)),
        key="audio_include_word",
    )
    include_sentence = st.checkbox(
        "Include sentence audio",
        value=bool(state.store.get("audio_include_sentence", True)),
        key="audio_include_sentence",
    )
    return include_word, include_sentence


def _render_style_controls(
    provider_data: Dict[str, Any],
    state: AudioPanelState,
) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    raw_sentence_styles = provider_data.get("sentence_styles") if isinstance(provider_data, dict) else None
    raw_word_styles = provider_data.get("word_styles") if isinstance(provider_data, dict) else None

    sentence_styles_map = raw_sentence_styles if isinstance(raw_sentence_styles, dict) else {}
    word_styles_map = raw_word_styles if isinstance(raw_word_styles, dict) else {}

    sentence_empty_message = "No sentence styles configured for this provider." if not sentence_styles_map else ""
    word_empty_message = "No word styles configured for this provider." if not word_styles_map else ""

    sentence_choice = _render_style_select(
        label="Sentence style",
        styles_map=sentence_styles_map,
        state=state,
        state_key="audio_sentence_instruction",
        default_key=provider_data.get("sentence_default") if isinstance(provider_data, dict) else "",
        empty_message=sentence_empty_message,
    )
    word_choice = _render_style_select(
        label="Word style",
        styles_map=word_styles_map,
        state=state,
        state_key="audio_word_instruction",
        default_key=provider_data.get("word_default") if isinstance(provider_data, dict) else "",
        empty_message=word_empty_message,
    )
    return sentence_choice, word_choice, sentence_styles_map or {}, word_styles_map or {}


def _render_style_select(
    *,
    label: str,
    styles_map: Optional[Dict[str, Any]],
    state: AudioPanelState,
    state_key: str,
    default_key: Any,
    empty_message: str,
) -> str:
    if not isinstance(styles_map, dict) or not styles_map:
        if empty_message:
            st.info(empty_message)
        return ""

    options = list(styles_map.keys())
    current_value = state.store.get(state_key)
    if current_value not in options:
        if isinstance(default_key, str) and default_key in options:
            current_value = default_key
        else:
            current_value = options[0]
        state.store[state_key] = current_value

    choice = st.selectbox(
        label,
        options=options,
        format_func=lambda key: styles_map.get(key, {}).get("label", key),
        key=state_key,
    )
    st.caption(str(styles_map.get(choice, {}).get("description", "")) or " ")
    return choice


def _render_estimate(unique_info: Dict[str, Any]) -> None:
    st.caption(
        "Estimated requests: "
        f"{unique_info['requests']} (unique words — {len(unique_info['unique_words'])}, "
        f"sentences — {len(unique_info['unique_sentences'])})."
    )


def _render_summary(
    audio_summary: Optional[Dict[str, Any]],
    providers: Dict[str, Dict[str, Any]],
    provider_key: str,
) -> None:
    if not isinstance(audio_summary, dict):
        return

    cache_hits = audio_summary.get("cache_hits", 0)
    fallback_hits = audio_summary.get("fallback_switches", 0)
    message = (
        f"Done: words — {audio_summary.get('word_success', 0)}, "
        f"sentences — {audio_summary.get('sentence_success', 0)}."
    )
    if cache_hits:
        message += f" Cache hits: {cache_hits}."
    if fallback_hits:
        message += f" Fallback used: {fallback_hits}×."
    st.success(message)
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

    summary_provider_key = audio_summary.get("provider") or provider_key
    summary_provider = providers.get(summary_provider_key, {}) if isinstance(providers, dict) else {}
    if not isinstance(summary_provider, dict):
        summary_provider = {}

    sentence_styles = summary_provider.get("sentence_styles", {}) if isinstance(summary_provider, dict) else {}
    word_styles = summary_provider.get("word_styles", {}) if isinstance(summary_provider, dict) else {}

    styles: List[str] = []
    sent_key = audio_summary.get("sentence_instruction_key") or ""
    word_key = audio_summary.get("word_instruction_key") or ""
    if sent_key:
        styles.append(
            f"sentence: {sentence_styles.get(sent_key, {}).get('label', sent_key)}"
        )
    if word_key:
        styles.append(
            f"word: {word_styles.get(word_key, {}).get('label', word_key)}"
        )
    if styles:
        st.caption("Styles → " + "; ".join(styles))

    provider_label = _provider_label(providers, summary_provider_key) if summary_provider_key else ""
    if not provider_label and provider_key:
        provider_label = _provider_label(providers, provider_key)
    if provider_label:
        st.caption(f"Provider → {provider_label}")
    if audio_summary.get("voice"):
        st.caption(f"Voice → {audio_summary['voice']}")


def _render_generate_button(
    *,
    state: AudioPanelState,
    settings: "SidebarConfig",
    provider_type: str,
    provider_data: Dict[str, Any],
    selected_voice: str,
    include_word: bool,
    include_sentence: bool,
    unique_info: Dict[str, Any],
    instruction_payloads: Dict[str, Any],
    instruction_keys: Dict[str, Any],
    sentence_styles_map: Dict[str, Any],
    word_styles_map: Dict[str, Any],
    eleven_api_key: str,
    max_workers: int,
    providers: Dict[str, Dict[str, Any]],
) -> None:
    button_disabled = unique_info["requests"] == 0 or not selected_voice or not provider_type
    generate_audio = st.button(
        "🔊 Generate audio",
        type="primary",
        disabled=button_disabled,
        key="generate_audio_button",
    )
    if not generate_audio:
        return

    if button_disabled:
        st.info("No text to synthesize — enable word or sentence above.")
        return

    if provider_type == "openai" and not settings.api_key:
        st.error("OPENAI_API_KEY is required for OpenAI TTS synthesis.")
        return
    if provider_type == "elevenlabs" and not eleven_api_key:
        st.error("Provide ELEVENLABS_API_KEY (environment, secrets, or field above) for ElevenLabs TTS.")
        return

    openai_client = OpenAI(api_key=settings.api_key) if provider_type == "openai" else None
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0.0)

    def _progress(done: int, total_requested: int) -> None:
        pct = (done / total_requested) if total_requested > 0 else 0.0
        progress_bar.progress(min(1.0, pct))
        status_placeholder.text(
            f"Audio progress: {done}/{total_requested} ({pct * 100:.0f}%)"
            if total_requested > 0
            else "Audio progress: 0/0 (0%)"
        )

    try:
        results = state.results()
        media_map, summary_obj = ensure_audio_for_cards(
            results,
            provider=provider_type,
            voice=selected_voice,
            include_word=include_word,
            include_sentence=include_sentence,
            cache=state.audio_cache(),
            progress_cb=_progress,
            instruction_payloads=instruction_payloads,
            instruction_keys=instruction_keys,
            max_workers=max_workers,
            openai_client=openai_client,
            openai_model=str(provider_data.get("model")) if provider_type == "openai" and provider_data.get("model") else None,
            openai_fallback_model=str(provider_data.get("fallback_model")) if provider_type == "openai" and provider_data.get("fallback_model") else None,
            eleven_api_key=eleven_api_key if provider_type == "elevenlabs" else None,
            eleven_model=str(provider_data.get("model")) if provider_type == "elevenlabs" and provider_data.get("model") else None,
        )
        state.set_audio_media(media_map)
        state.set_audio_summary(asdict(summary_obj))

        success_msg = (
            f"Done: words — {summary_obj.word_success}, "
            f"sentences — {summary_obj.sentence_success}."
        )
        if summary_obj.cache_hits:
            success_msg += f" Cache hits: {summary_obj.cache_hits}."
        if summary_obj.fallback_switches:
            success_msg += f" Fallback used: {summary_obj.fallback_switches}×."

        styles_now: List[str] = []
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
        state.set_results(results)
        status_placeholder.text(
            f"Audio progress: {summary_obj.total_requests}/{summary_obj.total_requests} (100%)"
            if summary_obj.total_requests
            else "Audio progress: 0/0 (0%)"
        )
        # Ensure the Export section above re-renders with fresh media map.
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()  # Streamlit <1.29 fallback
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover
        st.error(f"Audio synthesis failed: {exc}")
        status_placeholder.text("Audio generation failed.")
    finally:
        progress_placeholder.empty()


def _render_hide_button(state: AudioPanelState) -> None:
    if st.button("Hide audio options", key="hide_audio_panel"):
        state.collapse()
