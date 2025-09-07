"""
core/sanitize_validate.py — sanitation, cloze normalization, validation, and simple NL word heuristics.
This module must be Streamlit-agnostic and have no side effects.
"""
from __future__ import annotations

import re
import hashlib
from typing import Dict, List, Tuple

# Public API
__all__ = [
    "sanitize",
    "normalize_cloze_braces",
    "force_wrap_first_match",
    "try_separable_verb_wrap",
    "validate_card",
    "is_probably_dutch_word",
]

# -----------------
# Basic sanitation
# -----------------

def sanitize(value: str | None) -> str:
    """Replace forbidden pipe to avoid CSV breakage; strip whitespace."""
    if value is None:
        return ""
    return str(value).replace("|", "∣").strip()


# -----------------
# Cloze normalization
# -----------------

# Known separable verb particles (NL)
_SEP_PARTICLES = {
    "aan","af","achter","bij","binnen","buiten","door","heen","in","langs","mee",
    "na","nader","om","omhoog","omlaag","omver","onder","op","over","samen",
    "tegen","thuis","toe","uit","vast","voor","voort","weg","weer","wijzer","terug"
}
# Fix typo: replace Cyrillic 'п' case ("оп") by allowing only correct "op"
_SEP_PARTICLES.discard("оп")
_SEP_PARTICLES.add("op")


def normalize_cloze_braces(txt: str) -> str:
    """Normalize single to double braces without duplicating existing {{c1::…}}/{{c2::…}}.
    Also fixes common brace-length anomalies.
    """
    if not txt:
        return txt
    # Upgrade single-brace openings not already doubled
    txt = re.sub(r"(?<!\{)\{c([12])::", r"{{c\1::", txt)
    # Ensure closing '}}' for c1/c2 spans
    txt = re.sub(r"(\{\{c[12]::[^}]*)((?<!\})\})", r"\1}}", txt)
    # Collapse runs of 3+ opening braces before c1/c2 down to exactly 2
    txt = re.sub(r"\{\s*\{\s*\{+(?=c[12]::)", r"{{", txt)
    # Collapse 3+ closing braces after a cloze span down to exactly 2
    txt = re.sub(r"\}{3,}", r"}}", txt)
    return txt


def force_wrap_first_match(lemma: str, sentence: str) -> str:
    """If {{c1::...}} is missing, wrap the first plausible wordform.
    Heuristic: lemma and its rough verb stem (drop final -en).
    """
    if not lemma or not sentence or "{{c1::" in sentence:
        return sentence
    base = lemma.lower()
    candidates = {base}
    if base.endswith("en"):
        candidates.add(base[:-2])
    for m in re.finditer(r"[A-Za-zÀ-ÿ]+", sentence):
        w = m.group(0)
        wl = w.lower()
        if any(wl.startswith(c) or c.startswith(wl) for c in candidates):
            return sentence[:m.start()] + "{{c1::" + w + "}}" + sentence[m.end():]
    return sentence


def try_separable_verb_wrap(lemma: str, sentence: str) -> str:
    """Add {{c2::particle}} only if:
    - there is {{c1::...}} already,
    - NO existing {{c2::...}},
    - token matches a separable particle that is ALSO a prefix of the lemma.
    """
    if "{{c1::" not in sentence or "{{c2::" in sentence:
        return sentence
    if not lemma:
        return sentence

    lemma_lc = lemma.lower()
    allowed = [p for p in _SEP_PARTICLES if lemma_lc.startswith(p)]
    if not allowed:
        return sentence

    tokens = list(re.finditer(r"\b([A-Za-zÀ-ÿ]+)\b", sentence))
    for m in reversed(tokens):
        tok = m.group(1).lower()
        if tok in allowed:
            return sentence[:m.start()] + "{{c2::" + m.group(1) + "}}" + sentence[m.end():]
    return sentence


# -----------------
# Validation
# -----------------

_REQUIRED_KEYS = {"L2_word","L2_cloze","L1_sentence","L2_collocations","L2_definition","L1_gloss"}


def validate_card(card: Dict[str, str]) -> List[str]:
    """Field presence, cloze presence, collocations len=3, gloss len<=2, no pipes."""
    problems: List[str] = []
    for k in ["L2_word","L2_cloze","L1_sentence","L2_collocations","L2_definition","L1_gloss"]:
        v = card.get(k, "")
        if not isinstance(v, str) or not v.strip():
            problems.append(f"Field '{k}' is empty")
        if "|" in str(v):
            problems.append(f"Field '{k}' contains '|'")
    if "{{c1::" not in card.get("L2_cloze", ""):
        problems.append("Missing {{c1::…}} in L2_cloze")
    col_raw = card.get("L2_collocations", "")
    items = [s.strip() for s in re.split(r";\s*|\n+", col_raw) if s.strip()]
    if len(items) != 3:
        problems.append("L2_collocations must contain exactly 3 items")
    if len(card.get("L1_gloss", "").split()) > 2:
        problems.append("L1_gloss must be 1–2 words")
    return problems


# -----------------
# Dutch word heuristic (cheap local check)
# -----------------

# Allowed characters: latin letters + some diacritics, hyphen, apostrophe
_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]{1,39}$")


def is_probably_dutch_word(woord: str) -> Tuple[bool, str | None]:
    """Heuristic plausibility check for a Dutch lemma.
    Returns (ok, reason_if_not_ok_or_flag).
    - ok=True  → looks plausible
    - ok=False → reason contains an explanation
    """
    if not isinstance(woord, str) or not woord.strip():
        return False, "empty"
    w = woord.strip()
    if " " in w:
        return False, "contains space"
    if len(w) < 2 or len(w) > 40:
        return False, "length out of range"
    if any(ch.isdigit() for ch in w):
        return False, "contains digit"
    if not _TOKEN_RE.match(w):
        return False, "forbidden chars"
    if len(w) >= 3 and w.isupper():
        # likely acronym
        return False, "all-caps token"
    return True, None
