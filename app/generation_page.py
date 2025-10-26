"""Main generation, preview, audio, and export UI for the app."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import streamlit as st

from .batch_runner import BatchRunner
from .debug_panel import render_debug_panel
from .export_panel import render_export_section
from .preview_panel import render_preview_section
from .run_controls import RunController, render_run_controls
from .run_report import render_run_report_section
from .run_status import ensure_run_stats, update_overall_progress, update_run_summary
from .sidebar import SidebarConfig
from .tts_panel import render_audio_panel
from .ui_models import AudioConfig, ExportConfig


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

    simple_trigger = bool(state.pop("simple_start_requested", False))
    simple_msg = state.pop("simple_flow_message", None)

    total = len(state.input_data)
    processed = len(state.get("results", []))
    run_stats = ensure_run_stats(state)

    st.markdown("### ① Generate")
    if simple_trigger and simple_msg:
        st.info(simple_msg)

    summary = st.empty()
    valid_now = sum(1 for card in state.get("results", []) if not card.get("error"))
    update_run_summary(summary, run_stats, processed=processed, total=total, valid=valid_now)

    overall_caption = st.empty()
    overall = st.progress(0)
    update_overall_progress(overall, overall_caption, processed=processed, total=total)

    with st.expander("Advanced run controls", expanded=False):
        actions = render_run_controls()

    if simple_trigger:
        actions["start"] = True

    batch_runner = BatchRunner(
        settings=settings,
        state=state,
        summary=summary,
        overall_progress=overall,
        overall_caption=overall_caption,
        api_delay=api_delay,
        signalword_groups=signalword_groups,
        signalwords_b1=signalwords_b1,
        signalwords_b2_plus=signalwords_b2_plus,
    )

    RunController(
        settings=settings,
        state=state,
        process_batch=batch_runner.run_next_batch,
        rerun_errored=batch_runner.rerun_errors,
    ).handle(actions)

    st.divider()

    st.markdown("### ② Preview & fix")
    render_preview_section(state)

    st.divider()

    # Render audio controls before export so freshly generated media
    # is immediately visible when exporting.
    st.markdown("### ③ Audio (optional)")
    render_audio_panel(audio_config=audio_config, settings=settings)

    st.divider()

    st.markdown("### ④ Export deck")
    render_export_section(state, settings, export_config)
    st.divider()

    st.markdown("### ⑤ Run report")
    render_run_report_section(state)

    render_debug_panel(state)
