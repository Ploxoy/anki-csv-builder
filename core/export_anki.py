"""Экспорт карточек в Anki .apkg (Streamlit-агностично)."""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    media_files: Dict[str, bytes] | None = None,
    include_basic_reversed: bool = False,
    include_basic_typein: bool = False,
    basic_templates: Dict[str, str] | None = None,
    typein_templates: Dict[str, str] | None = None,
) -> bytes:
    """Сформировать .apkg пакет.

    tags_meta может включать level/profile/model/L1 для тегов.
    basic_templates / typein_templates ожидают готовые HTML-шаблоны,
    передаваемые при включении соответствующих доп. колод.
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
            {"name": "AudioSentence"},
            {"name": "AudioWord"},
        ],
        templates=[{"name": "Cloze", "qfmt": front_template, "afmt": back_template}],
        css=css,
        model_type=genanki.Model.CLOZE,
    )

    deck_cloze = genanki.Deck(deck_id, f"{deck_name}::Cloze")

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
                card.get("AudioSentence", ""),
                card.get("AudioWord", ""),
            ],
            guid=_compute_guid(card, guid_policy, run_id),
            tags=[t for t in set(base_tags) if t and not t.endswith('::')],
        )
        deck_cloze.add_note(note)

    extra_decks: List[genanki.Deck] = [deck_cloze]

    def _derive_id(base: int, salt: int) -> int:
        # keep in 32-bit signed range to satisfy Anki expectations
        limit = 2_147_483_647
        return (int(base) + int(salt)) % limit or (int(base) ^ int(salt))

    def _require_template(templates: Dict[str, str] | None, key: str, deck_label: str) -> str:
        if not templates or key not in templates:
            raise ValueError(f"Missing template '{key}' required for {deck_label} deck.")
        return templates[key]

    if include_basic_reversed:
        deck_label = "Basic (and reversed)"
        basic_model_id = _derive_id(model_id, 101)
        basic_model = genanki.Model(
            basic_model_id,
            f"{model_name} — {deck_label}",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": _require_template(basic_templates, "card1_front", deck_label),
                    "afmt": _require_template(basic_templates, "card1_back", deck_label),
                },
                {
                    "name": "Card 2",
                    "qfmt": _require_template(basic_templates, "card2_front", deck_label),
                    "afmt": _require_template(basic_templates, "card2_back", deck_label),
                },
            ],
            css=css,
        )
        deck_basic = genanki.Deck(_derive_id(deck_id, 101), f"{deck_name}::Basic")
        for card in cards:
            front = card.get("L2_word", "")
            back = card.get("L1_gloss", "")
            audio_word = card.get("AudioWord", "")
            if audio_word and front:
                # Attach audio to the Dutch side (Front contains L2_word)
                front = f"{front} {audio_word}".strip()
            if not (front or back):
                continue
            note = genanki.Note(
                model=basic_model,
                fields=[front, back],
                guid=genanki.guid_for(f"basic|{front}|{back}|{run_id}"),
                tags=[t for t in set(base_tags) if t and not t.endswith('::')],
            )
            deck_basic.add_note(note)
        extra_decks.append(deck_basic)

    if include_basic_typein:
        deck_label = "Basic (type-in)"
        typein_model_id = _derive_id(model_id, 202)
        typein_model = genanki.Model(
            typein_model_id,
            f"{model_name} — {deck_label}",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": _require_template(typein_templates, "front", deck_label),
                    "afmt": _require_template(typein_templates, "back", deck_label),
                },
            ],
            css=css,
        )
        deck_typein = genanki.Deck(_derive_id(deck_id, 202), f"{deck_name}::Type In")
        for card in cards:
            front = card.get("L1_gloss", "")
            back = card.get("L2_word", "")
            audio_word = card.get("AudioWord", "")
            if audio_word and back:
                # Attach audio to the Dutch side (Back contains L2_word)
                back = f"{back} {audio_word}".strip()
            if not (front or back):
                continue
            note = genanki.Note(
                model=typein_model,
                fields=[front, back],
                guid=genanki.guid_for(f"typein|{front}|{back}|{run_id}"),
                tags=[t for t in set(base_tags) if t and not t.endswith('::')],
            )
            deck_typein.add_note(note)
        extra_decks.append(deck_typein)

    pkg = genanki.Package(extra_decks if len(extra_decks) > 1 else extra_decks[0])
    media_files = media_files or {}

    with tempfile.TemporaryDirectory() as tmpdir:
        media_paths: List[str] = []
        for filename, data in media_files.items():
            path = Path(tmpdir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            media_paths.append(str(path))

        if media_paths:
            pkg.media_files = media_paths

        bio = io.BytesIO()
        pkg.write_to_file(bio)
        return bio.getvalue()
