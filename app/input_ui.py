"""Input handling (demo, upload, manual editor) for the Streamlit app."""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from core.parsing import parse_input
from core.sanitize_validate import is_probably_dutch_word

from . import ui_helpers


def render_input_section(demo_words: List[dict]) -> None:
    """Render upload/manual tabs and populate session input data."""

    quick_cols = st.columns([2, 1])
    with quick_cols[0]:
        st.markdown("### 🚀 Quick flow: Generate → Preview → Export")
        st.caption(
            "1. Load or enter words • 2. Press the button below • 3. Review the preview and download exports."
        )
    with quick_cols[1]:
        st.caption("Tip: the button enables auto-advance for all batches.")

    quick_start = st.button("Generate → Preview → Export", type="primary", key="quick_pipeline_start")
    if quick_start:
        if st.session_state.get("input_data"):
            st.session_state.simple_start_requested = True
            st.session_state.auto_advance_pending = True
            st.session_state.simple_flow_message = "Starting generation…"
        else:
            st.warning("Load input via Upload or Manual editor first, then run the quick flow again.")

    col_demo, col_clear = st.columns([1, 1])
    with col_demo:
        if st.button("Try demo", type="secondary"):
            st.session_state.input_data = [dict(row) for row in demo_words]
            st.session_state.manual_rows = [
                {
                    "woord": row.get("woord", ""),
                    "def_nl": row.get("def_nl", ""),
                    "translation": row.get("translation", "") or "",
                }
                for row in demo_words
            ]
            st.session_state.upload_token = None
            ui_helpers.apply_recommended_batch_params(
                len(st.session_state.input_data),
                token=ui_helpers.compute_list_token(st.session_state.input_data),
            )
            ui_helpers.toast("Demo set (6 words) loaded", icon="✅", variant="success")

    with col_clear:
        if st.button("Clear", type="secondary"):
            st.session_state.input_data = []
            st.session_state.results = []
            st.session_state.manual_rows = [{"woord": "", "def_nl": "", "translation": ""}]
            st.session_state.audio_media = {}
            st.session_state.audio_summary = None
            st.session_state.manual_override_token = None

    maybe_mode = st.session_state.get("input_mode_override") or st.session_state.get("input_mode")
    if maybe_mode not in {"upload", "manual"}:
        maybe_mode = "upload"

    tab_choice = st.radio(
        "Input mode",
        options=("upload", "manual"),
        format_func=lambda val: "📄 Upload" if val == "upload" else "✍️ Manual editor",
        key="input_mode",
        horizontal=True,
        index=(0 if maybe_mode == "upload" else 1),
    )

    st.session_state.input_mode_override = tab_choice

    if tab_choice == "upload":
        uploaded_file = st.file_uploader(
            "Upload .txt / .md / .tsv / .csv",
            type=["txt", "md", "tsv", "csv"],
            accept_multiple_files=False,
            key="file_uploader",
        )

        if uploaded_file is not None:
            upload_token = f"{getattr(uploaded_file, 'name', '')}|{getattr(uploaded_file, 'size', '')}"
            if st.session_state.get("upload_token") != upload_token or st.session_state.get("manual_override_token") not in (None, upload_token):
                try:
                    file_text = uploaded_file.read().decode("utf-8")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    file_text = uploaded_file.read().decode("utf-16")
                st.session_state.input_data = parse_input(file_text)
                st.session_state.manual_rows = [
                    {
                        "woord": row.get("woord", ""),
                        "def_nl": row.get("def_nl", ""),
                        "translation": row.get("translation", "") or "",
                    }
                    for row in st.session_state.input_data
                ]
                st.session_state.results = []
                st.session_state.upload_token = upload_token
                st.session_state.manual_override_token = None
                st.session_state.input_mode_override = "upload"
                ui_helpers.apply_recommended_batch_params(
                    len(st.session_state.input_data),
                    token=ui_helpers.compute_list_token(st.session_state.input_data),
                )
                ui_helpers.toast("Input replaced with uploaded file", icon="📄")
    else:
        manual_cols = st.columns([1, 1, 1])
        with manual_cols[0]:
            if st.button("Reset list", key="manual_reset"):
                st.session_state.manual_rows = [{"woord": "", "def_nl": "", "translation": ""}]
                st.session_state.manual_override_token = None
                st.session_state.input_mode_override = "manual"
        with manual_cols[1]:
            if st.button("Load demo", key="manual_seed_demo"):
                st.session_state.manual_rows = [
                    {
                        "woord": row.get("woord", ""),
                        "def_nl": row.get("def_nl", ""),
                        "translation": row.get("translation", "") or "",
                    }
                    for row in demo_words
                ]
                st.session_state.manual_override_token = None
        with manual_cols[2]:
            st.caption(
                "Add words, definitions, translations — mark rows as Delete to remove them."
            )

        def _normalize_row(row: dict) -> dict:
            return {
                "woord": str(row.get("woord", "") or ""),
                "def_nl": str(row.get("def_nl", "") or ""),
                "translation": str(row.get("translation", "") or ""),
            }

        manual_rows = [_normalize_row(row) for row in (st.session_state.manual_rows or [])]

        def _is_empty(row: dict) -> bool:
            return not (
                (row.get("woord") or "").strip()
                or (row.get("def_nl") or "").strip()
                or (row.get("translation") or "").strip()
            )

        def _ensure_trailing_blank(rows: list[dict]) -> list[dict]:
            rows_clean = rows[:]
            if not rows_clean or not _is_empty(rows_clean[-1]):
                rows_clean.append({"woord": "", "def_nl": "", "translation": ""})
            return rows_clean

        base_rows = _ensure_trailing_blank(manual_rows)

        editor_rows = []
        initial_state: list[bool] = st.session_state.get("_manual_delete_state", [])
        for idx, row in enumerate(base_rows):
            row_copy = dict(row)
            row_copy["_delete"] = bool(initial_state[idx]) if idx < len(initial_state) else False
            editor_rows.append(row_copy)

        manual_df = (
            pd.DataFrame(editor_rows)
            .fillna({"woord": "", "def_nl": "", "translation": "", "_delete": False})
            .reindex(columns=["woord", "def_nl", "translation", "_delete"])
        )
        delete_col_help = (
            "Mark rows you no longer need and press the delete button below to remove them."
        )
        edited_df = st.data_editor(
            manual_df,
            key="manual_editor",
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "woord": st.column_config.TextColumn("woord", help="Target Dutch lemma (required)"),
                "def_nl": st.column_config.TextColumn("def_nl", help="Optional: Dutch definition or context"),
                "translation": st.column_config.TextColumn("translation", help="Optional: translation in your L1"),
                "_delete": st.column_config.CheckboxColumn("Delete", help=delete_col_help),
            },
        )

        edited_records = (
            edited_df.fillna({"woord": "", "def_nl": "", "translation": "", "_delete": False}).to_dict("records")
        )
        st.session_state["_manual_delete_state"] = [bool(rec.get("_delete")) for rec in edited_records]

        if st.button("🗑️ Remove marked rows", key="manual_remove_marked"):
            edited_records = [row for row in edited_records if not row.get("_delete")]
            st.session_state["_manual_delete_state"] = []

        kept_records = []
        for record in edited_records:
            if record.get("_delete"):
                continue
            kept_records.append(
                {
                    "woord": str(record.get("woord", "") or ""),
                    "def_nl": str(record.get("def_nl", "") or ""),
                    "translation": str(record.get("translation", "") or ""),
                }
            )

        st.session_state.manual_rows = [row for row in kept_records if not _is_empty(row)]
        manual_clean = ui_helpers.clean_manual_rows(st.session_state.manual_rows)

        apply_col, info_col = st.columns([1, 1])
        with apply_col:
            if st.button("Use manual list", type="primary", key="manual_apply"):
                if manual_clean:
                    current_upload_token = st.session_state.get("upload_token")
                    if current_upload_token:
                        st.session_state.manual_override_token = current_upload_token
                        st.session_state.input_mode_override = "manual"
                    st.session_state.manual_rows = manual_clean
                    st.session_state.input_data = manual_clean
                    st.session_state.results = []
                    # Keep upload_token untouched so the uploaded file is not reloaded immediately.
                    if current_upload_token:
                        st.session_state.upload_token = current_upload_token
                    else:
                        st.session_state.upload_token = None
                    ui_helpers.apply_recommended_batch_params(
                        len(manual_clean),
                        token=ui_helpers.compute_list_token(manual_clean),
                    )
                    ui_helpers.toast(
                        f"Loaded {len(manual_clean)} manual rows",
                        icon="✍️",
                        variant="success",
                    )
                else:
                    st.warning("Please fill at least one row with the `woord` column.")
        with info_col:
            st.caption(f"Active rows: {len(manual_clean)}")

    if st.session_state.input_data:
        for row in st.session_state.input_data:
            ok, reason = is_probably_dutch_word(row.get("woord", ""))
            row["_flag_ok"] = bool(ok)
            row["_flag_reason"] = reason or ""

        flagged = [r for r in st.session_state.input_data if not r.get("_flag_ok", True)]
        if flagged:
            st.warning(
                f"{len(flagged)} rows flagged as suspicious. Check reasons in the preview table and sidebar tips."
            )

        st.subheader("🔍 Parsed rows")
        preview_in = pd.DataFrame(st.session_state.input_data)
        cols = [c for c in ["woord", "def_nl", "translation", "_flag_ok", "_flag_reason"] if c in preview_in.columns]

        def _reason_hint(reason: str) -> str:
            if not reason:
                return ""
            if reason == "contains digit":
                return "Example: 12345"
            if reason == "all-caps token":
                return "Example: HELLO"
            if reason.startswith("token '") and reason.endswith(" suspicious"):
                return "Example: hello (looks English)"
            return ""

        styled_df = preview_in[cols].style.format({
            "_flag_ok": lambda v: "✅" if v else "⚠️",
            "_flag_reason": lambda v: f"{v} ({_reason_hint(v)})" if v else "",
        })
        st.dataframe(styled_df, width="stretch")
    else:
        st.info("Upload a file or click **Try demo**")
