# core/parsing.py
# Input parsing for Doedutch — text (.txt/.md), TSV, and simple lists -> normalized rows.
# Comments and docstrings are in English.
from __future__ import annotations
import re
from typing import List, Dict

__all__ = ["parse_input", "normalize_row"]

# Regex that matches markdown table separator rows like: |---| or |:---:|
RE_MD_SEPARATOR = re.compile(r"^\|\s*:?-{3,}[\|\:\-]*\s*\:?$", re.IGNORECASE)
def _parse_markdown_line(line: str) -> Dict | None:
    """Parse a markdown table row like: | woord | definitie NL | RU |"""
    parts = [p.strip() for p in line.strip("|").split("|")]
    if len(parts) < 1:
        return None
    # remove simple inline markup from first cell (bold/italic/inline code)
    woord = re.sub(r"[*_`]", "", parts[0]).strip()
    if not woord:
        return None
    out: Dict[str, str] = {"woord": woord}
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
    """Parse 'woord — definitie NL — RU' or 'woord — definitie NL' (em-dash with spaces)."""
    if " — " not in line:
        return None
    parts = [p.strip() for p in line.split(" — ")]
    if len(parts) == 3:
        return {"woord": parts[0], "def_nl": parts[1], "ru_short": parts[2]}
    if len(parts) == 2:
        return {"woord": parts[0], "def_nl": parts[1]}
    return None

def normalize_row(row: Dict) -> Dict:
    """Normalize keys: ensure strings and strip whitespace. Keeps only known keys."""
    nr: Dict[str, str] = {}
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

    Notes:
    - Header rows in markdown tables (e.g. '| woord | definitie NL | RU |') are skipped.
    - Separator rows like '|---|---|' are skipped.
    """
    rows: List[Dict] = []
    if text is None:
        return rows

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip markdown table separator like: |---| or |:---:|
        compact = re.sub(r"\s+", "", line)
        if line.startswith("|") and RE_MD_SEPARATOR.match(compact):
            continue
        # Markdown row handling
        if line.startswith("|"):
            # Detect and skip header row (common in markdown tables)
            # Remove basic inline markup and whitespace from first cell and lowercase it.
            first_cell = ""
            try:
                first_cell = re.sub(r"[*_`\\s]", "", line.strip("|").split("|")[0]).strip().lower()
            except Exception:
                first_cell = ""
            # Known header tokens that indicate a table header (common variants)
            header_tokens = {
                "woord", "word", "definitienl", "definitie", "definition", "ru", "translation", "перевод"
            }
            if first_cell in header_tokens:
                # Likely a header row — skip
                continue
            parsed = _parse_markdown_line(line)
            if parsed:
                rows.append(normalize_row(parsed))
            continue

        # TSV detection
        if "\t" in line:
            parsed = _parse_tsv_line(line)
            if parsed:
                rows.append(normalize_row(parsed))
                continue

        # Em-dash form
        if " — " in line:
            parsed = _parse_em_dash_line(line)
            if parsed:
                rows.append(normalize_row(parsed))
                continue
        # Fallback: treat entire line as a single 'woord'
        rows.append(normalize_row({"woord": line}))

    return rows