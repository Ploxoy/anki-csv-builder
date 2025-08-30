"""
parser.py — модуль для парсинга и нормализации входных данных
"""

import re
from typing import List, Dict

def parse_input(text: str) -> List[Dict]:
    """
    Парсит входной текст в различных форматах и возвращает список словарей с данными.

    Поддерживаемые форматы:
      1) Строки таблицы Markdown:
         | woord | definitie NL | RU |
         Игнорирует заголовки и разделители вида |---|, |:---|:---:|

      2) TSV (с табуляцией):
         woord<tab>definitie NL

      3) Строки с тире:
         'woord — definitie NL — RU'
         'woord — definitie NL'

      4) По одному слову на строку:
         woord

    Args:
        text (str): Входной текст в любом из поддерживаемых форматов

    Returns:
        List[Dict]: Список словарей, где каждый словарь содержит:
            - woord (str): Обязательное поле - голландское слово
            - def_nl (str, optional): Определение на голландском
            - ru_short (str, optional): Короткий перевод на русский
    """
    rows: List[Dict] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue

        # 1) Markdown table: skip headers/separators like |---| or |:---:|
        if line.startswith("|"):
            compact = re.sub(r"\s+", "", line)
            if re.match(r"^\|\:?-{3,}[\|\:\-]{0,}\:?$", compact):
                # alignment/separator row → skip
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 2:
                # remove markdown bold/italics from cell 0
                woord = re.sub(r"[*_`]", "", parts[0]).strip()
                def_nl = parts[1].strip()
                entry = {"woord": woord} if woord else {}
                if def_nl:
                    entry["def_nl"] = def_nl
                if len(parts) >= 3 and parts[2].strip():
                    entry["ru_short"] = parts[2].strip()
                if entry:
                    rows.append(entry)
                continue

        # 2) TSV (2 cols)
        if "\t" in line:
            tparts = [p.strip() for p in line.split("\t")]
            if len(tparts) == 2 and tparts[0]:
                rows.append({"woord": tparts[0], "def_nl": tparts[1]})
                continue

        # 3) Line with em-dash
        if " — " in line:
            parts = [p.strip() for p in line.split(" — ")]
            if len(parts) == 3:
                rows.append({"woord": parts[0], "def_nl": parts[1], "ru_short": parts[2]})
                continue
            if len(parts) == 2:
                rows.append({"woord": parts[0], "def_nl": parts[1]})
                continue

        # 4) Single word
        rows.append({"woord": line})
    return rows