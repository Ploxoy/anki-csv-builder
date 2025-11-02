"""Generation section UI helpers for batch orchestration."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from .batch_runner import BatchRunner
from .run_controls import RunController, render_run_controls
from .run_status import ensure_run_stats, update_overall_progress, update_run_summary
from .sidebar import SidebarConfig


def render_generation_section(
    settings: SidebarConfig,
    *,
    state: Any | None = None,
    signalword_groups: Optional[Dict],
    signalwords_b1: List[str],
    signalwords_b2_plus: List[str],
    api_delay: float,
) -> None:
    """Render generation run controls, progress, and orchestration."""

    state = state or st.session_state

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

    summary_placeholder = st.empty()
    valid_now = sum(1 for card in state.get("results", []) if not card.get("error"))
    update_run_summary(
        summary_placeholder,
        run_stats,
        processed=processed,
        total=total,
        valid=valid_now,
    )

    overall_caption = st.empty()
    overall_progress = st.progress(0)
    update_overall_progress(
        overall_progress,
        overall_caption,
        processed=processed,
        total=total,
    )

    with st.expander("Advanced run controls", expanded=False):
        actions = render_run_controls()

    if simple_trigger:
        actions["start"] = True

    batch_runner = BatchRunner(
        settings=settings,
        state=state,
        summary=summary_placeholder,
        overall_progress=overall_progress,
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
