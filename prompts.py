# prompts.py — v2 (terse/strict)
from typing import Dict

try:
    from config import (  # type: ignore
        L1_LANGS as CFG_L1_LANGS,
        LEVEL_RULES_EN as CFG_LEVEL_RULES_EN,
        PROMPT_PROFILES as CFG_PROMPT_PROFILES,
    )
except Exception:
    CFG_L1_LANGS = {
        "RU": {"label": "RU", "name": "Russian", "csv_translation": "Translation", "csv_gloss": "Word gloss"},
        "EN": {"label": "EN", "name": "English", "csv_translation": "Translation", "csv_gloss": "Word gloss"},
        "ES": {"label": "ES", "name": "Spanish", "csv_translation": "Traducción", "csv_gloss": "Glosa"},
        "DE": {"label": "DE", "name": "German", "csv_translation": "Übersetzung", "csv_gloss": "Kurzgloss"},
    }
    CFG_LEVEL_RULES_EN: Dict[str, str] = {
        "A1": "6–9 words; no subclauses; no passive; no perfect.",
        "A2": "8–12 words; may use modal verbs; simple past allowed; no complex clauses.",
        "B1": "10–14 words; simple subclause allowed (omdat/als/terwijl); ~50% with one signal word.",
        "B2": "12–16 words; complex allowed; passive allowed; ~50% with one signal word (extended list).",
        "C1": "14–18 words; advanced structures; neutral‑formal.",
        "C2": "No length limit; native‑like precision.",
    }
    CFG_PROMPT_PROFILES: Dict[str, str] = {
        "strict":   "Literal and concise.",
        "balanced": "Natural and clear.",
        "exam":     "Neutral‑formal; precise; no colloquialisms.",
        "creative": "Mild figurativeness if clarity is kept.",
    }

L1_LANGS = CFG_L1_LANGS
LEVEL_RULES_EN = CFG_LEVEL_RULES_EN
PROMPT_PROFILES = CFG_PROMPT_PROFILES


from string import Template  # add this import at the top of prompts.py

def compose_instructions_en(L1_code: str, level: str, profile: str) -> str:
    """
    Terse, strict instruction for Responses API; L2 is Dutch, L1 is user-selected.
    Uses string.Template so braces like {c1::...} and {{c1::...}} remain literal.
    """
    L1_name = L1_LANGS[L1_code]["name"]
    lvl = LEVEL_RULES_EN.get(level, "")
    prof = PROMPT_PROFILES.get(profile, "")

    tpl = Template("""
You are a Dutch→${L1_NAME} lexicographer. Produce exactly ONE JSON object with fields:
L2_word, L2_cloze, L1_sentence, L2_collocations, L2_definition, L1_gloss.

DO:
- Use ONE natural Dutch sentence.
- Cloze MUST use exactly double braces: {{c1::...}}. For separable verbs: {{c1::stem}} … {{c2::particle}}. Never output {{c2::...}} unless separable.
- Keep the sentence within CEFR guidance: ${LEVEL} → ${LEVEL_RULE}
- L1_sentence: faithful translation to ${L1_NAME}.
- L2_collocations: EXACTLY 3 frequent, natural collocations containing the target word; join with '; '.
- L2_definition: short Dutch definition.
- L1_gloss: 1–2 words in ${L1_NAME} matching the part of speech/meaning.
- If input JSON says INCLUDE_SIGNALWORD=true and provides ALLOWED_SIGNALWORDS, include exactly ONE suitable signal word in the sentence (only if natural).

DON'T:
- No extra text around JSON. No pipes '|' anywhere. No proper names/dates/digits/quotes. No empty fields.
- No single braces {c1::word} or triple braces {{{c1::word}}}.

CLOZE COMPLIANCE (VERY IMPORTANT):
- You MUST use exactly TWO curly braces on both sides: {{c1::...}} (and {{c2::...}} for separable verbs).
- BAD: {c1::raak}    GOOD: {{c1::raak}}
- If the verb is NOT separable, do NOT output {{c2::...}} at all.
- Ensure the final sentence contains at least one {{c1::...}} and all braces are balanced.

Examples:
- Regular: Ik {{c1::begrijp}} deze zin.
- Separable: Ik {{c1::ruim}} mijn kamer {{c2::op}}.

Style: ${PROFILE}
""".strip())

    return tpl.substitute(
        L1_NAME=L1_name,
        LEVEL=level,
        LEVEL_RULE=lvl,
        PROFILE=prof,
    )

