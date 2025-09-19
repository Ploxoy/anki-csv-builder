"""Prompt composition utilities for card generation."""
from __future__ import annotations

from string import Template

from config.settings import (
    L1_LANGS,
    LEVEL_RULES_EN,
    PROMPT_PROFILES,
)


def compose_instructions_en(L1_code: str, level: str, profile: str) -> str:
    """Build strict English instructions for the Responses API."""
    L1_meta = L1_LANGS[L1_code]
    level_rule = LEVEL_RULES_EN.get(level, "")
    profile_rules = PROMPT_PROFILES.get(profile, "")

    template = Template(
        """
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
""".strip()
    )

    return template.substitute(
        L1_NAME=L1_meta["name"],
        LEVEL=level,
        LEVEL_RULE=level_rule,
        PROFILE=profile_rules,
    )
