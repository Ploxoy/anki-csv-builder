"""Debug panel showing last model request metadata."""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st


def render_debug_panel(state: Any) -> None:
    """Render expander with last request information."""

    last_meta: Optional[dict] = None
    if state.results:
        last_meta = (state.results or [{}])[-1].get("meta")
        if not isinstance(last_meta, dict):
            last_meta = None

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
