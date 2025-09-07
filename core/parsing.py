# core/parsing.py
# Парсинг входного текста (.txt /.md / TSV / single words) → normalized rows

from __future__ import annotations
import re
from typing import List, Dict

__all__ = ["parse_input", "normalize_row"]

RE_MD_SEPARATOR = re.compile(r"^\|\s*:?-{3,}[\|\:\-]*\s*\:?$", re.IGNORECASE)
RE_WORD = re.compile(r"[^\t\r\n]+")

def _parse_markdown_line(line: str) -> Dict | None:
    """Parse a markdown table row like: | woord | definitie NL | RU |"""
    parts = [p.strip() for p in line.strip("|").split("|")]
    if len(parts) < 1:
        return None
    # remove simple markup from first cell
    woord = re.sub(r"[*_`]", "", parts[0]).strip()
    if not woord:
        return None
    out = {"woord": woord}
    if len(parts) >= 2 and parts[1].strip():
        out["def_nl"] = parts[1].strip()
    if len(parts) >= 3 and parts[2].strip():
        out["ru_short"] = parts[2].strip()
    return out

def _parse_tsv_line(line: str) -> Dict | None:
    """Parse a line with tab-separated two columns: woord<TAB>def_nl"""
    parts = [p.strip() for p in line.split("\t")]
    if len(parts) >= 2 and parts[0]:
        return {"woord": parts[0], "def_nl": parts[1]}
    return None

def _parse_em_dash_line(line: str) -> Dict | None:
    """Parse 'woord — definitie NL — RU' or 'woord — definitie NL'"""
    if " — " not in line:
        return None
    parts = [p.strip() for p in line.split(" — ")]
    if len(parts) == 3:
        return {"woord": parts[0], "def_nl": parts[1], "ru_short": parts[2]}
    if len(parts) == 2:
        return {"woord": parts[0], "def_nl": parts[1]}
    return None

def normalize_row(row: Dict) -> Dict:
    """Normalize keys and ensure strings; trim whitespace."""
    nr = {}
    nr["woord"] = str(row.get("woord", "")).strip()
    if row.get("def_nl") is not None:
        nr["def_nl"] = str(row.get("def_nl", "")).strip()
    if row.get("ru_short") is not None:
        nr["ru_short"] = str(row.get("ru_short", "")).strip()
    return nr

def parse_input(text: str) -> List[Dict]:
    """
    Parse input text into normalized rows.
    Supported formats:
      - Markdown table rows: | woord | definitie NL | RU |
      - TSV: woord <TAB> definitie
      - Line with em-dash: 'woord — definitie NL — RU' or 'woord — definitie NL'
      - Single word per line
    """
    rows: List[Dict] = []
    if text is None:
        return rows
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # markdown table separator skip
        compact = re.sub(r"\s+", "", line)
        if line.startswith("|") and RE_MD_SEPARATOR.match(compact):
            continue
        # markdown row
        if line.startswith("|"):
            parsed = _parse_markdown_line(line)
            if parsed:
                rows.append(normalize_row(parsed))
            continue
        # tsv
        if "\t" in line:
            parsed = _parse_tsv_line(line)
            if parsed:
                rows.append(normalize_row(parsed))
                continue
        # em-dash form
        if " — " in line:
            parsed = _parse_em_dash_line(line)
            if parsed:
                rows.append(normalize_row(parsed))
                continue
        # fallback: single token / single word
        # keep whole line as woord (may be 'woord' or 'woord - extra' — assume word)
        rows.append(normalize_row({"woord": line}))
    return rows