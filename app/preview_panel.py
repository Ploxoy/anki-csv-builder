"""Preview table rendering for generated cards."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_preview_section(state: Any) -> None:
    """Render the results preview table with filters and navigation."""

    with st.container():
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
