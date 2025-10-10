"""Streamlit UI helper utilities for the Anki CSV Builder app."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from core.llm_clients import (
    create_client,
    responses_accepts_param,
    send_responses_request,
)


def get_secret(name: str) -> Optional[str]:
    """Return Streamlit secret or env var by name."""

    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def toast(message: str, *, icon: Optional[str] = None, variant: str = "info") -> None:
    """Show toast with optional fallback for older Streamlit versions."""

    toast_fn = getattr(st, "toast", None)
    if callable(toast_fn):
        kwargs: Dict[str, str] = {}
        if icon:
            kwargs["icon"] = icon
        toast_fn(message, **kwargs)
        return

    fallback_msg = f"{icon} {message}" if icon else message
    if variant == "success":
        st.success(fallback_msg)
    elif variant == "warning":
        st.warning(fallback_msg)
    else:
        st.info(fallback_msg)


def ensure_session_defaults(
    *,
    providers: Dict[str, Dict[str, object]],
    default_provider: str,
    elevenlabs_default_key: str = "",
) -> None:
    """Populate session_state with defaults expected by the UI."""

    state = st.session_state
    state.setdefault("input_data", [])
    state.setdefault("results", [])
    state.setdefault("manual_rows", [{"woord": "", "def_nl": "", "translation": ""}])
    state.setdefault("audio_cache", {})
    state.setdefault("audio_media", {})
    state.setdefault("audio_summary", None)
    state.setdefault("audio_voice_map", {})

    provider_keys = list(providers.keys())
    provider_key = default_provider if default_provider in providers else (provider_keys[0] if provider_keys else "")
    state.setdefault("audio_provider", provider_key)

    current = providers.get(state.get("audio_provider", provider_key), {})

    voices = current.get("voices") if isinstance(current, dict) else None
    default_voice = current.get("voice_default") if isinstance(current, dict) else None
    if not default_voice and isinstance(voices, list) and voices:
        voice_entry = voices[0]
        if isinstance(voice_entry, dict):
            default_voice = voice_entry.get("id", "")

    state.setdefault("audio_voice", default_voice or "")
    state.setdefault("audio_include_word", bool(current.get("include_word_default", True)))
    state.setdefault("audio_include_sentence", bool(current.get("include_sentence_default", True)))
    sentence_default = str(current.get("sentence_default", "")) if isinstance(current, dict) else ""
    word_default = str(current.get("word_default", "")) if isinstance(current, dict) else ""
    state.setdefault("audio_sentence_instruction", sentence_default)
    state.setdefault("audio_word_instruction", word_default)
    state.setdefault("audio_panel_expanded", bool(state.get("results")))

    if "elevenlabs_api_key" not in state or not state.get("elevenlabs_api_key"):
        secret = get_secret("ELEVENLABS_API_KEY")
        if secret:
            state.elevenlabs_api_key = secret
        elif elevenlabs_default_key:
            state.elevenlabs_api_key = elevenlabs_default_key


def init_signalword_state() -> None:
    """Ensure signal word usage counters exist in session."""

    state = st.session_state
    state.setdefault("sig_usage", {})
    state.setdefault("sig_last", None)


def init_response_format_cache() -> None:
    """Ensure schema support caches exist in session."""

    state = st.session_state
    state.setdefault("no_response_format_models", set())
    state.setdefault("no_response_format_notified", set())


def probe_response_format_support(client: object, model: str) -> None:
    """Run a minimal schema probe and cache unsupported models."""

    if not model:
        return
    state = st.session_state
    cache: set[str] = set(state.get("no_response_format_models", set()))
    if model in cache:
        return
    if not responses_accepts_param(client, "text"):
        cache.add(model)
        state["no_response_format_models"] = cache
        return
    try:
        probe_schema = {
            "name": "Probe",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
        }
        instructions = "Return strictly: {\"ok\": true}"
        _, meta = send_responses_request(
            client=client,
            model=model,
            instructions=instructions,
            input_text="probe",
            response_format=probe_schema,
            max_output_tokens=64,
            temperature=None,
            retries=0,
            warn=False,
        )
        if meta.get("response_format_removed"):
            cache.add(model)
            state["no_response_format_models"] = cache
            notified = set(state.get("no_response_format_notified", set()))
            if model not in notified:
                notified.add(model)
                state["no_response_format_notified"] = notified
                detail = meta.get("response_format_error")
                message = (
                    f"Model {model} ignored response_format; falling back to text parsing for this session."
                )
                if detail:
                    message += f"\nReason: {detail}"
                st.info(message, icon="ℹ️")
    except Exception:
        pass


def clean_manual_rows(rows: List[Dict]) -> List[Dict[str, str]]:
    """Trim manual editor rows and drop empty entries."""

    cleaned: List[Dict[str, str]] = []
    for raw in rows:
        woord = str(raw.get("woord", "") or "").strip()
        def_nl = str(raw.get("def_nl", "") or "").strip()
        translation = str(raw.get("translation", "") or "").strip()
        if not (woord or def_nl or translation):
            continue
        if not woord:
            continue
        cleaned.append({
            "woord": woord,
            "def_nl": def_nl,
            "translation": translation,
        })
    return cleaned


def _sort_key(model_id: str, preferred_order: Dict[str, int]) -> Tuple[int, str]:
    for prefix, rank in preferred_order.items():
        if model_id.startswith(prefix):
            return (rank, model_id)
    return (999, model_id)


def get_model_options(
    api_key: Optional[str],
    *,
    preferred_order: Dict[str, int],
    allowed_prefixes: Sequence[str],
    block_substrings: Sequence[str],
    default_models: Sequence[str],
) -> List[str]:
    """Return filtered list of available models, fallback to defaults on error."""

    if not api_key:
        return list(default_models)
    try:
        client = create_client(api_key)
        if client is None:
            st.error(
                "OpenAI SDK not available (core.llm_clients.create_client returned None). Please install the OpenAI SDK."
            )
            return list(default_models)
        models = client.models.list()
        ids: List[str] = []
        for model in getattr(models, "data", []) or []:
            mid = getattr(model, "id", "")
            if any(mid.startswith(p) for p in allowed_prefixes) and not any(b in mid for b in block_substrings):
                ids.append(mid)
        if not ids:
            return list(default_models)
        unique_ids = sorted({mid for mid in ids}, key=lambda mid: _sort_key(mid, preferred_order))
        return unique_ids
    except Exception:
        return list(default_models)


def recommend_batch_params(total: int) -> Tuple[int, int]:
    """Heuristic batch size and worker count for dataset length."""

    if total <= 0:
        return (5, 3)
    if total <= 10:
        return (total, min(10, total))
    target_batches = max(2, min(8, math.ceil(total / 20)))
    batch_size = max(1, min(total, math.ceil(total / target_batches)))
    workers = min(10, max(2, min(batch_size, 10)))
    return (batch_size, workers)


def compute_list_token(rows: Sequence[Dict]) -> str:
    """Compute fingerprint token for dataset to avoid reapplying recommendations."""

    try:
        first = (rows[0].get("woord", "") if rows else "").strip()
        last = (rows[-1].get("woord", "") if rows else "").strip()
        return f"len={len(rows)}|first={first}|last={last}"
    except Exception:
        return f"len={len(rows)}"


def apply_recommended_batch_params(total: int, *, token: Optional[str] = None) -> None:
    """Store recommended batch params and trigger rerun if needed."""

    state = st.session_state
    if token and state.get("recommend_applied_token") == token:
        return
    batch_size, workers = recommend_batch_params(total)
    state["batch_size_pending"] = int(batch_size)
    state["max_workers_pending"] = int(workers)
    if total > 1:
        state["auto_advance_pending"] = True
    if token:
        state["recommend_applied_token"] = token
    toast(f"Recommended batch: size {batch_size}, workers {workers}", icon="⚙️")
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def should_pass_temperature(model_id: str) -> bool:
    """Some models do not accept temperature parameter."""

    no_temp = st.session_state.get("no_temp_models", set())
    if model_id in no_temp:
        return False
    if model_id.startswith(("gpt-5", "o3")):
        return False
    return True
