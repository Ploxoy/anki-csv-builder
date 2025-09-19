"""CSV экспорт карточек Anki (Streamlit-агностичная утилита)."""
from __future__ import annotations

import csv
import io
from typing import Dict, Iterable


def generate_csv(
    results: Iterable[Dict[str, str]],
    l1_meta: Dict[str, str],
    *,
    delimiter: str = "|",
    line_terminator: str = "\n",
    include_header: bool = True,
    include_extras: bool = False,
    anki_field_header: bool = True,
    extras_meta: Dict[str, str] | None = None,
) -> str:
    """Сформировать CSV с карточками.

    results — iterable карточек, каждая карточка dict.
    l1_meta ожидает ключи csv_translation, csv_gloss, label.
    extras_meta используется, если include_extras=True.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator=line_terminator)

    if include_header:
        if anki_field_header:
            header = [
                "L2_word",
                "L2_cloze",
                "L1_sentence",
                "L2_collocations",
                "L2_definition",
                "L1_gloss",
                "L1_hint",
            ]
        else:
            header = [
                "NL-слово",
                "Предложение NL (с cloze)",
                f"{l1_meta['csv_translation']} {l1_meta['label']}",
                "Коллокации (NL)",
                "Определение NL",
                f"{l1_meta['csv_gloss']} {l1_meta['label']}",
                "Подсказка (L1)",
            ]
        if include_extras:
            header += ["CEFR", "Profile", "Model", "L1"]
        writer.writerow(header)

    extras_meta = extras_meta or {}

    for card in results:
        row = [
            card.get("L2_word", ""),
            card.get("L2_cloze", ""),
            card.get("L1_sentence", ""),
            card.get("L2_collocations", ""),
            card.get("L2_definition", ""),
            card.get("L1_gloss", ""),
            card.get("L1_hint", ""),
        ]
        if include_extras:
            row += [
                extras_meta.get("level", ""),
                extras_meta.get("profile", ""),
                extras_meta.get("model", ""),
                extras_meta.get("L1", ""),
            ]
        writer.writerow(row)

    return buffer.getvalue()
