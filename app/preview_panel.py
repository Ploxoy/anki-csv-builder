"""Preview table rendering for generated cards."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

EDITABLE_FIELDS = [
    "L2_word",
    "L2_cloze",
    "L1_sentence",
    "L2_collocations",
    "L2_definition",
    "L1_gloss",
    "L1_hint",
]


def _as_text(value: Any) -> str:
    """Normalize dataframe/editor values to plain strings."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _build_preview_df(cards: List[Dict[str, Any]], indices: List[int]) -> pd.DataFrame:
    """Build editable preview rows for selected cards."""

    rows: List[Dict[str, Any]] = []
    for idx in indices:
        card = cards[idx]
        row: Dict[str, Any] = {
            "_result_index": idx,
            "row": idx + 1,
            "status": "error" if card.get("error") else "ok",
            "error": _as_text(card.get("error")),
        }
        for field in EDITABLE_FIELDS:
            row[field] = _as_text(card.get(field))
        rows.append(row)
    return pd.DataFrame(rows)


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
        elif next_err_click:
            st.info("No errors found in current results.")

        if state.results:
            selected_indices = list(range(len(state.results)))
            if state.get("preview_only_errors"):
                selected_indices = [idx for idx, card in enumerate(state.results) if card.get("error")]

            if not selected_indices:
                st.info("No rows match the selected filter.")
                return

            st.caption(
                "Editable fields: L2_word, L2_cloze, L1_sentence, L2_collocations, "
                "L2_definition, L1_gloss, L1_hint."
            )
            preview_df = _build_preview_df(state.results, selected_indices)
            editor_key = "preview_editor_errors" if state.get("preview_only_errors") else "preview_editor_all"
            edited_df = st.data_editor(
                preview_df,
                hide_index=True,
                use_container_width=True,
                disabled=["_result_index", "row", "status", "error"],
                key=editor_key,
            )

            if st.button("Apply preview edits", key=f"{editor_key}_apply"):
                changed_fields = 0
                changed_rows = set()
                for rec in edited_df.to_dict(orient="records"):
                    try:
                        result_idx = int(rec.get("_result_index"))
                    except Exception:
                        continue
                    if not (0 <= result_idx < len(state.results)):
                        continue
                    card = state.results[result_idx]
                    for field in EDITABLE_FIELDS:
                        new_value = _as_text(rec.get(field))
                        old_value = _as_text(card.get(field))
                        if new_value != old_value:
                            card[field] = new_value
                            changed_fields += 1
                            changed_rows.add(result_idx)
                if changed_fields:
                    st.success(
                        f"Saved {changed_fields} field edits across {len(changed_rows)} row(s)."
                    )
                else:
                    st.info("No field changes detected.")
        else:
            st.info("No results yet — run a batch to populate preview.")
