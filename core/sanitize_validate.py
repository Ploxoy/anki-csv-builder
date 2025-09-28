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
    # Allow up to 6 words in L1_gloss to better fit languages like Russian
    if len(card.get("L1_gloss", "").split()) > 6:
        problems.append("L1_gloss must be ≤ 6 words")
    return problems


# -----------------
# Dutch word heuristic (cheap local check)
# -----------------

# Allowed characters: latin letters + some diacritics, hyphen, apostrophe
_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]{1,39}$")


def is_probably_dutch_word(
    woord: str,
    allow_multiword: bool = True,
    allow_articles: bool = True,
    min_len: int = 2,
    max_len: int = 40,
) -> tuple[bool, str | None]:
    """
    Heuristic plausibility check for a Dutch lemma.

    Returns (ok, reason). If ok==False, `reason` explains why.

    Strategy:
    - Normalize whitespace and remove invisible chars.
    - Optionally strip leading articles ('de', 'het', 'een').
    - Tokenize by whitespace and hyphen.
    - Use wordfreq (if available) to detect strong EN/NL signals.
    - Fallback to morphological heuristics (suffixes, digraphs, long alpha tokens).
    """
    if not isinstance(woord, str) or not woord:
        return False, "empty"

    # Normalize and clean common invisible whitespace characters
    w = woord.strip()
    w = w.replace("\u00A0", " ")  # NO-BREAK SPACE
    w = w.replace("\u200B", "")   # ZERO WIDTH SPACE
    w = w.replace("\uFEFF", "")   # ZERO WIDTH NO-BREAK SPACE (BOM)
    w = re.sub(r"\s+", " ", w).strip()

    # Length checks after normalization
    if len(w) < min_len or len(w) > max_len:
        return False, "length out of range"

    # Optionally strip leading Dutch article
    if allow_articles:
        low = w.lower()
        if low.startswith("de "):
            w = w[len("de "):].strip()
        elif low.startswith("het "):
            w = w[len("het "):].strip()
        elif low.startswith("een "):
            w = w[len("een "):].strip()

    if not w:
        return False, "empty after removing article"

    # Tokenize: split on whitespace and hyphen
    tokens = [t for t in re.split(r"[\s\-]+", w) if t]
    if not tokens:
        return False, "empty tokens"

    # Quick reject if multiword disallowed
    if not allow_multiword and len(tokens) > 1:
        return False, "contains space"

    # Wordfreq config & thresholds
    try:
        from wordfreq import zipf_frequency  # type: ignore
        have_wordfreq = True
    except Exception:
        have_wordfreq = False

    Z_MIN = 2.5        # minimum zipf to consider 'common'
    DIFF_EN = 1.5      # en - nl difference to strongly mark as EN
    DIFF_NL = 1.0      # nl - en difference to strongly mark as NL
    LONG_ALPHA_MIN = 6

    # Morphological hints
    dutch_suffixes = ("en", "ing", "heid", "tje", "aat", "isch", "lijk", "baar", "schap", "kunde")
    dutch_digraphs = ("ij", "oe", "ui", "eu", "aa", "ee", "oo", "ou", "ch", "sch")

    token_results = []  # collect per-token verdicts

    for tok in tokens:
        if not tok:
            return False, "empty token"
        # Reject digits
        if any(ch.isdigit() for ch in tok):
            return False, "contains digit"
        # Allowed chars check
        if not _TOKEN_RE.match(tok):
            return False, "forbidden chars"
        # Reject all-caps acronyms (>=3)
        if len(tok) >= 3 and tok.isupper():
            return False, "all-caps token"

        tl = tok.lower()
        token_ok = False
        token_reason = None

        # wordfreq-based decision if available
        if have_wordfreq:
            try:
                z_nl = zipf_frequency(tok, "nl")
                z_en = zipf_frequency(tok, "en")
            except Exception:
                z_nl = z_en = None

            if z_nl is not None and z_en is not None:
                # Strong EN signal -> reject whole input
                if (z_en >= Z_MIN) and (z_en - (z_nl or 0.0) >= DIFF_EN):
                    return False, "likely English word"
                # Strong NL signal -> accept this token
                if (z_nl >= Z_MIN) and ((z_nl - (z_en or 0.0)) >= DIFF_NL):
                    token_ok = True
        # Morphological heuristics (fallback / supplement)
        if not token_ok:
            if any(tl.endswith(suf) for suf in dutch_suffixes) or any(dg in tl for dg in dutch_digraphs):
                token_ok = True
            elif len(tl) >= LONG_ALPHA_MIN and tl.isalpha():
                # accept long alphabetic compound (common in Dutch compounds)
                token_ok = True
            elif 2 <= len(tl) <= 3 and tl.isalpha():
                # short 2-3 letter tokens often valid (e.g., 'zijn', 'zij' etc.)
                token_ok = True

        token_results.append((tok, token_ok))
        if not token_ok:
            token_reason = f"token '{tok}' suspicious"
            # For multiword, defer final decision after checking other tokens
            if len(tokens) == 1:
                return False, token_reason

    # Multiword decision: if any token strongly EN => we already returned.
    # If majority tokens ok -> accept; else suspicious
    ok_count = sum(1 for _, ok in token_results if ok)
    if ok_count >= max(1, len(token_results) // 2):
        return True, None

    return False, "tokens suspicious"
