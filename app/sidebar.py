"""Sidebar rendering for the Streamlit app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import streamlit as st

from core.llm_clients import create_client, responses_accepts_param

from . import ui_helpers


PRESET_CUSTOM_LABEL = "Custom"
SIDEBAR_PRESETS: Dict[str, Dict[str, object]] = {
    "Starter (B1 • RU • gpt-4.1-mini)": {
        "model": "gpt-4.1-mini",
        "level": "B1",
        "profile": "strict",
        "L1": "RU",
        "limit_tokens": True,
        "force_flagged": False,
    },
    "Fast (B1 • RU • gpt-4o)": {
        "model": "gpt-4o",
        "level": "B1",
        "profile": "balanced",
        "L1": "RU",
        "limit_tokens": True,
        "force_flagged": False,
    },
    "Quality (B2 • RU • gpt-4.1)": {
        "model": "gpt-4.1",
        "level": "B2",
        "profile": "balanced",
        "L1": "RU",
        "limit_tokens": False,
        "force_flagged": True,
    },
}


@dataclass
class SidebarConfig:
    api_key: Optional[str]
    model: str
    profile: str
    level: str
    L1_code: str
    L1_meta: Dict[str, str]
    temperature: float
    limit_tokens: bool
    display_raw_response: bool
    csv_with_header: bool
    csv_anki_header: bool
    force_flagged: bool


def render_sidebar(
    *,
    default_models: Sequence[str],
    preferred_order: Dict[str, int],
    block_substrings: Sequence[str],
    allowed_prefixes: Sequence[str],
    prompt_profiles: Dict[str, str],
    l1_langs: Dict[str, Dict[str, str]],
    temperature_min: float,
    temperature_max: float,
    temperature_default: float,
    temperature_step: float,
) -> SidebarConfig:
    """Render sidebar controls and return collected configuration."""

    state = st.session_state
    state.setdefault("sidebar_model_current", "gpt-4.1-mini")
    state.setdefault("sidebar_profile_current", "strict")
    state.setdefault("sidebar_level_current", "B1")
    state.setdefault("sidebar_limit_tokens_current", True)
    state.setdefault("sidebar_preset_choice", PRESET_CUSTOM_LABEL)
    state.setdefault("sidebar_preset_last_applied", PRESET_CUSTOM_LABEL)
    state.setdefault("force_flagged", False)

    l1_keys = list(l1_langs.keys()) if l1_langs else []
    if l1_keys:
        state.setdefault("sidebar_L1_current", l1_keys[0])
    else:
        state.setdefault("sidebar_L1_current", "")

    preset_options = [PRESET_CUSTOM_LABEL] + list(SIDEBAR_PRESETS.keys())
    current_choice = state.get("sidebar_preset_choice", PRESET_CUSTOM_LABEL)
    if current_choice not in preset_options:
        current_choice = PRESET_CUSTOM_LABEL
        state.sidebar_preset_choice = current_choice
    preset_index = preset_options.index(current_choice)
    selected_preset = st.sidebar.selectbox(
        "🎛️ Preset",
        preset_options,
        index=preset_index,
        help="Quick-start configuration for model, CEFR, L1 and flags. Choose 'Custom' to tweak manually.",
    )
    state.sidebar_preset_choice = selected_preset

    st.sidebar.header("🔐 API Settings")

    api_key = ui_helpers.get_secret("OPENAI_API_KEY") or st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
    )

    try:
        import openai as _openai  # type: ignore

        st.sidebar.caption(f"OpenAI SDK: v{_openai.__version__}")
    except Exception:
        pass

    model_options = ui_helpers.get_model_options(
        api_key,
        preferred_order=preferred_order,
        allowed_prefixes=allowed_prefixes,
        block_substrings=block_substrings,
        default_models=default_models,
    )

    selected_preset = state.get("sidebar_preset_choice", PRESET_CUSTOM_LABEL)
    last_applied = state.get("sidebar_preset_last_applied", PRESET_CUSTOM_LABEL)
    if (
        selected_preset != PRESET_CUSTOM_LABEL
        and selected_preset in SIDEBAR_PRESETS
        and selected_preset != last_applied
    ):
        preset_cfg = SIDEBAR_PRESETS[selected_preset]
        preset_model = str(preset_cfg.get("model", "") or "")
        if preset_model in model_options:
            state.sidebar_model_current = preset_model
        elif model_options:
            state.sidebar_model_current = model_options[0]
        preset_profile = str(preset_cfg.get("profile", ""))
        if preset_profile:
            state.sidebar_profile_current = preset_profile
        preset_level = str(preset_cfg.get("level", ""))
        if preset_level:
            state.sidebar_level_current = preset_level
        preset_L1 = str(preset_cfg.get("L1", ""))
        if preset_L1 and preset_L1 in l1_langs:
            state.sidebar_L1_current = preset_L1
        state.sidebar_limit_tokens_current = bool(preset_cfg.get("limit_tokens", state.sidebar_limit_tokens_current))
        state.force_flagged = bool(preset_cfg.get("force_flagged", state.get("force_flagged", False)))
        state.sidebar_preset_last_applied = selected_preset
    elif selected_preset == PRESET_CUSTOM_LABEL:
        state.sidebar_preset_last_applied = PRESET_CUSTOM_LABEL

    preferred_default = state.get("sidebar_model_current", "gpt-4.1-mini")
    if preferred_default not in model_options and model_options:
        preferred_default = model_options[0]
        state.sidebar_model_current = preferred_default
    try:
        default_index = model_options.index(preferred_default)
    except ValueError:
        default_index = 0

    model = st.sidebar.selectbox(
        "Model",
        model_options,
        index=default_index,
        help="Best quality — gpt-5 (if available); balanced — gpt-4.1; faster/cheaper — gpt-4o / gpt-5-mini.",
    )
    state.sidebar_model_current = model

    profile_keys = list(prompt_profiles.keys())
    profile_default = state.get("sidebar_profile_current", "strict")
    if profile_default not in profile_keys and profile_keys:
        profile_default = profile_keys[0]
        state.sidebar_profile_current = profile_default
    profile_index = profile_keys.index(profile_default) if profile_default in profile_keys else 0
    profile = st.sidebar.selectbox("Prompt profile", profile_keys, index=profile_index)
    state.sidebar_profile_current = profile

    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    level_default = state.get("sidebar_level_current", "B1")
    if level_default not in levels:
        level_default = "B1"
        state.sidebar_level_current = level_default
    level_index = levels.index(level_default)
    level = st.sidebar.selectbox("CEFR", levels, index=level_index)
    state.sidebar_level_current = level

    L1_keys = list(l1_langs.keys())
    L1_default = state.get("sidebar_L1_current", L1_keys[0] if L1_keys else "")
    if L1_default not in L1_keys and L1_keys:
        L1_default = L1_keys[0]
        state.sidebar_L1_current = L1_default
    L1_index = L1_keys.index(L1_default) if L1_default in L1_keys else 0
    L1_code = st.sidebar.selectbox("Your language (L1)", L1_keys, index=L1_index)
    state.sidebar_L1_current = L1_code
    L1_meta = l1_langs[L1_code]

    temperature = st.sidebar.slider(
        "Temperature",
        temperature_min,
        temperature_max,
        temperature_default,
        temperature_step,
    )

    limit_tokens_default = bool(state.get("sidebar_limit_tokens_current", True))
    limit_tokens = st.sidebar.checkbox(
        "Limit output tokens",
        value=limit_tokens_default,
        help="Check to limit the number of output tokens. Uncheck to allow unlimited tokens.",
    )
    state.sidebar_limit_tokens_current = limit_tokens

    display_raw_response = st.sidebar.checkbox(
        "Display raw responses",
        value=False,
        help="Check to display raw responses from the OpenAI API.",
    )

    csv_with_header = st.sidebar.checkbox(
        "CSV: include header row",
        value=True,
        help="Uncheck if your Anki import treats the first row as a record.",
    )
    csv_anki_header = st.sidebar.checkbox(
        "CSV: use Anki field names",
        value=True,
        help="Header row will be: L2_word|L2_cloze|L1_sentence|L2_collocations|L2_definition|L1_gloss|L1_hint",
    )

    force_flagged = st.sidebar.checkbox(
        "Force generate for flagged entries",
        value=state.get("force_flagged", False),
        help="If off, rows flagged as suspicious by a quick heuristic will be skipped from generation.",
    )

    st.session_state["csv_with_header"] = csv_with_header
    st.session_state["force_flagged"] = force_flagged

    guid_label = st.sidebar.selectbox(
        "Anki GUID policy",
        ["stable (update/skip existing)", "unique per export (import as new)"],
        index=0,
        help=(
            "stable: the same notes are recognized as existing/updatable\n"
            "unique: each export has new GUIDs — Anki treats them as new notes."
        ),
    )
    st.session_state["anki_guid_policy"] = "unique" if guid_label.startswith("unique") else "stable"
    st.session_state["prompt_profile"] = profile
    st.session_state["level"] = level
    st.session_state["L1_code"] = L1_code

    st.sidebar.subheader("Batch processing")
    st.sidebar.number_input(
        "Batch size",
        min_value=1,
        max_value=50,
        value=st.session_state.get("batch_size", 5),
        step=1,
        help="How many rows to process per batch.",
        key="batch_size",
    )
    st.sidebar.checkbox(
        "Auto-advance batches",
        value=st.session_state.get("auto_advance", True),
        help="Continue to the next batch automatically until finished.",
        key="auto_advance",
    )
    st.sidebar.slider(
        "Max workers per batch",
        min_value=1,
        max_value=8,
        value=st.session_state.get("max_workers", 3),
        step=1,
        help="Parallel requests inside a batch. Keep modest (3–4) to avoid rate limits.",
        key="max_workers",
    )

    with st.sidebar.expander("Advanced (Responses schema)"):
        st.checkbox(
            "Force JSON schema (ignore cache)",
            value=False,
            help=(
                "Attempt to send response_format=json_schema even if the model was previously marked as unsupported. "
                "If the SDK/model rejects it, we will fall back automatically."
            ),
            key="force_schema_checkbox",
        )
        try:
            client = create_client(api_key)
            if client is not None and not responses_accepts_param(client, "text"):
                st.caption("SDK check: Responses.create has no 'text' parameter — schema (text.format) will be disabled.")
        except Exception:
            pass
        if st.button("Reset schema support cache"):
            st.session_state.no_response_format_models = set()
            st.session_state.no_response_format_notified = set()
            st.success("Schema support cache has been reset for this session.")
        if st.button("Re-probe schema support for selected model"):
            ui_helpers.init_response_format_cache()
            client = create_client(api_key)
            if client is None:
                st.warning("OpenAI SDK not available; cannot probe.")
            else:
                ui_helpers.probe_response_format_support(client, model)
                st.info("Probe completed. Check debug panel or try generation.")

    return SidebarConfig(
        api_key=api_key,
        model=model,
        profile=profile,
        level=level,
        L1_code=L1_code,
        L1_meta=L1_meta,
        temperature=temperature,
        limit_tokens=limit_tokens,
        display_raw_response=display_raw_response,
        csv_with_header=csv_with_header,
        csv_anki_header=csv_anki_header,
        force_flagged=force_flagged,
    )
