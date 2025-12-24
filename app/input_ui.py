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
            ui_helpers.toast("Demo set (6 items) loaded", icon="✅", variant="success")

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

        def _rows_to_text(rows: list[dict]) -> str:
            lines: list[str] = []
            for row in rows:
                woord = str(row.get("woord", "") or "").strip()
                if not woord:
                    continue
                def_nl = str(row.get("def_nl", "") or "").strip()
                translation = str(
                    row.get("translation")
                    or row.get("ru_short")
                    or row.get("L1_gloss")
                    or ""
                ).strip()
                parts = [woord]
                if def_nl:
                    parts.append(def_nl)
                if translation:
                    parts.append(translation)
                lines.append(" — ".join(parts))
            return "\n".join(lines)

        def _normalize_manual_row(row: dict) -> dict:
            return {
                "woord": str(row.get("woord", "") or "").strip(),
                "def_nl": str(row.get("def_nl", "") or "").strip(),
                "translation": str(
                    row.get("translation")
                    or row.get("ru_short")
                    or row.get("L1_gloss")
                    or ""
                ).strip(),
            }

        existing_rows = [
            _normalize_manual_row(row) for row in st.session_state.get("manual_rows", [])
        ]
        existing_text = _rows_to_text(existing_rows)
        if "manual_text_buffer" not in st.session_state:
            st.session_state.manual_text_buffer = existing_text

        with manual_cols[0]:
            if st.button("Reset text", key="manual_reset"):
                st.session_state.manual_text_buffer = ""
                st.session_state.manual_override_token = None
                st.session_state.input_mode_override = "manual"

        with manual_cols[1]:
            if st.button("Load demo text", key="manual_seed_demo"):
                st.session_state.manual_text_buffer = _rows_to_text(demo_words)
                st.session_state.manual_override_token = None

        with manual_cols[2]:
            st.caption(
                "Paste text in formats like `woord — definitie — vertaling`, Markdown table rows, or TSV."
            )

        st.markdown(
            "Enter one entry per line. Supported examples:\n"
            "- `woord`\n"
            "- `woord — definitie`\n"
            "- `woord — definitie — vertaling`\n"
            "- `woord ;; definitie ;; vertaling` (double semicolon)\n"
            "- `woord<TAB>definitie` (paste real tabs or type `\\t`)\n"
            "- `| woord | definitie | vertaling |`"
        )

        st.text_area(
            "Manual text input",
            key="manual_text_buffer",
            height=220,
            placeholder="bijvoorbeeld — ter illustratie — например",
        )

        action_cols = st.columns([1, 1])
        with action_cols[0]:
            append_mode = st.checkbox(
                "Append to existing parsed rows",
                key="manual_append_mode",
                help="If enabled, new rows from the text area are appended instead of replacing the current parsed list.",
            )
        with action_cols[1]:
            if st.button("Clear parsed rows", key="manual_clear_rows"):
                st.session_state.manual_rows = []
                st.session_state.input_data = []
                st.session_state.results = []
                existing_rows = []
                ui_helpers.toast("Manual rows cleared", icon="🧹")

        def _parse_manual_text(raw_text: str) -> list[dict]:
            normalized_text = (raw_text or "").replace("\\t", "\t")
            rows = parse_input(normalized_text)
            normalized = []
            for row in rows:
                norm = _normalize_manual_row(row)
                if norm["woord"]:
                    normalized.append(norm)
            return normalized

        if st.button("Parse & load text", type="primary", key="manual_apply_text"):
            parsed_rows = _parse_manual_text(st.session_state.get("manual_text_buffer", ""))
            if not parsed_rows:
                st.warning("No valid rows detected. Provide at least one word.")
            else:
                current_upload_token = st.session_state.get("upload_token")
                if current_upload_token:
                    st.session_state.manual_override_token = current_upload_token
                    st.session_state.input_mode_override = "manual"
                combined_rows = (existing_rows + parsed_rows) if append_mode else parsed_rows
                st.session_state.manual_rows = combined_rows
                st.session_state.input_data = combined_rows
                st.session_state.results = []
                ui_helpers.apply_recommended_batch_params(
                    len(combined_rows),
                    token=ui_helpers.compute_list_token(combined_rows),
                )
                ui_helpers.toast(
                    f"Loaded {len(parsed_rows)} new rows • total now {len(combined_rows)}",
                    icon="✍️",
                    variant="success",
                )
                existing_rows = combined_rows
        active_rows = [row for row in existing_rows if row.get("woord")]
        st.caption(f"Current manual rows ready to run: {len(active_rows)}")

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
