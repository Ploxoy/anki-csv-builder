"""Export widgets for CSV and APKG outputs."""
from __future__ import annotations

import time
from typing import Any

import streamlit as st

from core.export_anki import HAS_GENANKI, build_anki_package
from core.export_csv import generate_csv

from .sidebar import SidebarConfig
from .ui_models import ExportConfig


def render_export_section(state: Any, settings: SidebarConfig, export_config: ExportConfig) -> None:
    """Render CSV and APKG export controls."""

    csv_extras = {
        "level": state.get("level", settings.level),
        "profile": state.get("prompt_profile", settings.profile),
        "model": state.get("model_id", settings.model),
        "L1": state.get("L1_code", settings.L1_code),
    }
    include_errored = st.sidebar.checkbox("Include errored cards in exports", value=False)
    export_cards = state.results if include_errored else [card for card in state.results if not card.get("error")]
    exportable_count = len(export_cards)
    saved_count = len(state.results)
    st.caption(
        f"Exportable cards: {exportable_count} / saved results: {saved_count} / total input: {len(state.input_data)}"
    )

    csv_data = generate_csv(
        export_cards,
        settings.L1_meta,
        delimiter=export_config.csv_delimiter,
        line_terminator=export_config.csv_lineterminator,
        include_header=state.get("csv_with_header", True),
        include_extras=True,
        anki_field_header=settings.csv_anki_header,
        extras_meta=csv_extras,
    )

    state.last_csv_data = csv_data
    st.download_button(
        label="📥 Download anki_cards.csv",
        data=csv_data,
        file_name="anki_cards.csv",
        mime="text/csv",
        key="download_csv",
    )

    if HAS_GENANKI:
        try:
            front_template_raw = export_config.front_template_path.read_text(encoding="utf-8")
            front_html = front_template_raw.replace("{L1_LABEL}", settings.L1_meta["label"])
            back_html = export_config.back_template_path.read_text(encoding="utf-8")
            css_content = export_config.css_path.read_text(encoding="utf-8")
            tags_meta = {
                "level": state.get("level", settings.level),
                "profile": state.get("prompt_profile", settings.profile),
                "model": state.get("model_id", settings.model),
                "L1": state.get("L1_code", settings.L1_code),
            }

            st.caption("Also include additional decks in the package:")
            col_a, col_b = st.columns(2)
            include_basic_rev = col_a.checkbox(
                "Basic (and reversed card) — Front: L2_word, Back: L1_gloss",
                value=True,
                key="export_include_basic_reversed",
            )
            include_basic_typein = col_b.checkbox(
                "Basic (type in the answer) — Front: L1_gloss, Back: L2_word",
                value=True,
                key="export_include_basic_typein",
            )
            basic_templates = None
            if include_basic_rev:
                basic_templates = {
                    key: path.read_text(encoding="utf-8")
                    for key, path in export_config.basic_templates.items()
                }
            typein_templates = None
            if include_basic_typein:
                typein_templates = {
                    key: path.read_text(encoding="utf-8")
                    for key, path in export_config.typein_templates.items()
                }
            anki_bytes = build_anki_package(
                export_cards,
                l1_label=settings.L1_meta["label"],
                guid_policy=state.get("anki_guid_policy", "stable"),
                run_id=state.get("anki_run_id", str(int(time.time()))),
                model_id=export_config.anki_model_id,
                model_name=export_config.anki_model_name,
                deck_id=export_config.anki_deck_id,
                deck_name=export_config.anki_deck_name,
                front_template=front_html,
                back_template=back_html,
                css=css_content,
                tags_meta=tags_meta,
                media_files=state.get("audio_media"),
                include_basic_reversed=include_basic_rev,
                include_basic_typein=include_basic_typein,
                basic_templates=basic_templates,
                typein_templates=typein_templates,
            )
            state.last_anki_package = anki_bytes
            st.download_button(
                label="🧩 Download Anki deck (.apkg)",
                data=anki_bytes,
                file_name="dutch_cloze.apkg",
                mime="application/octet-stream",
                key="download_apkg",
            )
        except Exception as exc:
            st.error(f"Failed to build .apkg: {exc}")
    else:
        st.info("To enable .apkg export, add 'genanki' to requirements.txt and restart the app.")
