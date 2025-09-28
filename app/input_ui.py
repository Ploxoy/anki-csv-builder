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

    tab_upload, tab_manual = st.tabs(["📄 Upload", "✍️ Manual editor"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload .txt / .md / .tsv / .csv",
            type=["txt", "md", "tsv", "csv"],
            accept_multiple_files=False,
            key="file_uploader",
        )

        if uploaded_file is not None:
            upload_token = f"{getattr(uploaded_file, 'name', '')}|{getattr(uploaded_file, 'size', '')}"
            if st.session_state.get("upload_token") != upload_token:
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
                ui_helpers.apply_recommended_batch_params(
                    len(st.session_state.input_data),
                    token=ui_helpers.compute_list_token(st.session_state.input_data),
                )
                ui_helpers.toast("Input replaced with uploaded file", icon="📄")

    with tab_manual:
        manual_cols = st.columns([1, 1, 1])
        with manual_cols[0]:
            if st.button("Reset list", key="manual_reset"):
                st.session_state.manual_rows = [{"woord": "", "def_nl": "", "translation": ""}]
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
        with manual_cols[2]:
            st.caption(
                "Add words, definitions, and translations if needed. You can edit or delete rows."
            )

        manual_rows = list(st.session_state.manual_rows or [])

        def _is_empty(row: dict) -> bool:
            return not (
                (row.get("woord") or "").strip()
                or (row.get("def_nl") or "").strip()
                or (row.get("translation") or "").strip()
            )

        if not manual_rows or not _is_empty(manual_rows[-1]):
            manual_rows.append({"woord": "", "def_nl": "", "translation": ""})

        manual_df = pd.DataFrame(manual_rows).reindex(columns=["woord", "def_nl", "translation"])
        edited_df = st.data_editor(
            manual_df,
            key="manual_editor",
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "woord": st.column_config.TextColumn("woord", help="Целевое нидерландское слово"),
                "def_nl": st.column_config.TextColumn("def_nl", help="Опционально: определение или контекст"),
                "translation": st.column_config.TextColumn("translation", help="Опционально: перевод на L1"),
            },
        )

        st.session_state.manual_rows = edited_df.fillna("").to_dict("records")
        manual_clean = ui_helpers.clean_manual_rows(st.session_state.manual_rows)

        apply_col, info_col = st.columns([1, 1])
        with apply_col:
            if st.button("Use manual list", type="primary", key="manual_apply"):
                if manual_clean:
                    st.session_state.input_data = manual_clean
                    st.session_state.results = []
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
                    st.warning("Нужно заполнить хотя бы одну строку с полем 'woord'.")
        with info_col:
            st.caption(f"Активные строки: {len(manual_clean)}")

    if st.session_state.input_data:
        for row in st.session_state.input_data:
            ok, reason = is_probably_dutch_word(row.get("woord", ""))
            row["_flag_ok"] = bool(ok)
            row["_flag_reason"] = reason or ""

        flagged = [r for r in st.session_state.input_data if not r.get("_flag_ok", True)]
        if flagged:
            st.warning(
                f"{len(flagged)} rows flagged as suspicious by a quick heuristic. "
                "Use 'Force generate for flagged entries' in the sidebar to ignore flags."
            )

        st.subheader("🔍 Parsed rows")
        preview_in = pd.DataFrame(st.session_state.input_data)
        cols = [c for c in ["woord", "def_nl", "ru_short", "_flag_ok", "_flag_reason"] if c in preview_in.columns]
        st.dataframe(preview_in[cols], width="stretch")
    else:
        st.info("Upload a file or click **Try demo**")
