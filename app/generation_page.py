"""Main generation, preview, audio, and export UI for the app."""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from .debug_panel import render_debug_panel
from .export_panel import render_export_section
from .generation_section import render_generation_section
from .preview_panel import render_preview_section
from .run_report import render_run_report_section
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
    render_generation_section(
        settings,
        state=state,
        signalword_groups=signalword_groups,
        signalwords_b1=signalwords_b1,
        signalwords_b2_plus=signalwords_b2_plus,
        api_delay=api_delay,
    )

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
