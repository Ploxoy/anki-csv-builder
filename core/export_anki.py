"""Экспорт карточек в Anki .apkg (Streamlit-агностично)."""
from __future__ import annotations

import hashlib
import io
from typing import Dict, Iterable, List

try:
    import genanki  # type: ignore

    HAS_GENANKI = True
except Exception:  # pragma: no cover
    HAS_GENANKI = False


def _compute_guid(card: Dict[str, str], policy: str, run_id: str) -> str:
    base = f"{card.get('L2_word','')}|{card.get('L2_cloze','')}"
    if policy == "unique":
        base = base + "|" + run_id
    try:
        return genanki.guid_for(base)  # type: ignore[attr-defined]
    except Exception:
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]


def build_anki_package(
    cards: Iterable[Dict[str, str]],
    *,
    l1_label: str,
    guid_policy: str,
    run_id: str,
    model_id: int,
    model_name: str,
    deck_id: int,
    deck_name: str,
    front_template: str,
    back_template: str,
    css: str,
    tags_meta: Dict[str, str] | None = None,
) -> bytes:
    """Сформировать .apkg пакет.

    tags_meta может включать level/profile/model/L1 для тегов.
    """

    if not HAS_GENANKI:
        raise RuntimeError(
            "genanki is not installed. Add 'genanki' to requirements.txt and redeploy."
        )

    tags_meta = tags_meta or {}

    model = genanki.Model(
        model_id,
        model_name,
        fields=[
            {"name": "L2_word"},
            {"name": "L2_cloze"},
            {"name": "L1_sentence"},
            {"name": "L2_collocations"},
            {"name": "L2_definition"},
            {"name": "L1_gloss"},
            {"name": "L1_hint"},
        ],
        templates=[{"name": "Cloze", "qfmt": front_template, "afmt": back_template}],
        css=css,
        model_type=genanki.Model.CLOZE,
    )

    deck = genanki.Deck(deck_id, deck_name)

    base_tags = [
        f"CEFR::{tags_meta.get('level','')}",
        f"profile::{tags_meta.get('profile','')}",
        f"model::{tags_meta.get('model','')}",
        f"L1::{tags_meta.get('L1','')}",
    ]

    for card in cards:
        note = genanki.Note(
            model=model,
            fields=[
                card.get("L2_word", ""),
                card.get("L2_cloze", ""),
                card.get("L1_sentence", ""),
                card.get("L2_collocations", ""),
                card.get("L2_definition", ""),
                card.get("L1_gloss", ""),
                card.get("L1_hint", ""),
            ],
            guid=_compute_guid(card, guid_policy, run_id),
            tags=[t for t in set(base_tags) if t and not t.endswith('::')],
        )
        deck.add_note(note)

    pkg = genanki.Package(deck)
    bio = io.BytesIO()
    pkg.write_to_file(bio)
    return bio.getvalue()
