"""Prompt composition utilities for card generation."""
from __future__ import annotations

from string import Template

from config.settings import L1_LANGS, LEVEL_RULES_EN, PROMPT_PROFILES

PROMPT_TEMPLATE = Template(
    """
You are an expert Dutch→$L1_NAME lexicographer and didactics writer.
Produce a STRICT JSON object with fields ONLY:
- L2_word           : the Dutch target lemma (single token, no POS tags)
- L2_cloze          : ONE short natural Dutch sentence containing cloze markup
- L1_sentence       : exact translation of that sentence into $L1_NAME
- L2_collocations   : EXACTLY 3 natural Dutch collocations with the target, joined by '; '
- L2_definition     : ONE short Dutch definition
- L1_gloss          : 1–6 $L1_NAME words (short gloss, correct POS/meaning)

GLOBAL HARD RULES
- Output JSON ONLY. No comments, no extra keys, no trailing text.
- No field may be empty. Do NOT use the '|' character anywhere.
- Modern Dutch only; avoid names, digits, and quotes.
- Present tense by default unless the sentence truly requires otherwise.

CLOZE RULES (STRICT, ANKI-COMPATIBLE)
- Use ONLY the c1 family. Do NOT use c2/c3/etc. at all.
- Each cloze must have EXACTLY two curly braces: '{{' and '}}' (not single, not triple, not quadruple).
- For regular (non-separable) targets: mark the whole target surface form as {{c1::...}}.
- For separable verbs: use MULTI-SPAN c1 — stem and particle are BOTH {{c1::...}}, e.g.:
  GOOD  : Ik {{c1::ruim}} mijn kamer {{c1::op}}.
  BAD   : Ik {{c1::ruim}} mijn kamer {{c2::op}}.        (never use c2)
  BAD   : Ik {{{{c1::ruim}}}} mijn kamer {{c1::op}}.     (never 4 braces)
- Never cloze only a suffix or ending (e.g., '-t', '-en'), or only a prefix ('ge-', 'te-').
- Never cloze function words: articles (de/het/een), pronouns (ik, je, hij, ze, we), clitics ('n), particles like 'te', 'er', 'ge-'.
- Never cloze signal words (conjunctions/connectors).
- Never wrap the WHOLE sentence; only the target form(s).

TRANSLATION & OTHER FIELDS
- L1_sentence must contain ZERO braces — absolutely no {{ }} there.
- L2_collocations must contain ZERO braces — plain text, joined by '; ' (exactly 3 items).
- L2_definition and L1_gloss must contain ZERO braces and no pipes.

SIGNAL WORDS (B1+ only, about 50% of the time)
- If CEFR ≥ B1, you MAY include exactly ONE appropriate Dutch signal word in L2_cloze.
- Do NOT start the sentence with it; place it naturally mid-clause.
- NEVER cloze the signal word.

CEFR & STYLE
- Sentence length/grammar must follow CEFR constraints for this request: $LEVEL_RULE
- Style guidance: $PROFILE_RULE

GOOD / BAD EXAMPLES
- Regular verb:
  GOOD : Ik {{c1::begrijp}} deze zin.      → “I understand this sentence.”
  BAD  : Ik {c1::begrijp} deze zin.        (single braces)
  BAD  : Ik {{{c1::begrijp}}} deze zin.    (triple)
  BAD  : Ik {{c1::begrij}}p deze zin.      (suffix only)
- Separable verb:
  GOOD : We {{c1::nemen}} het afval {{c1::mee}}.
  BAD  : We {{c1::nemen}} het afval {{c2::mee}}.  (no c2)
  BAD  : We {{{{c1::nemen}}}} het afval {{c1::mee}}. (no 4 braces)
- Articles / signal words:
  BAD  : {{c1::De}} man leest.             (no articles)
  BAD  : Ik ga, {{c1::hoewel}}, naar huis. (no cloze on signal words)

OUTPUT VALIDATION YOU MUST PASS
- L2_cloze contains at least one {{c1::...}} and NO other cloze families.
- L1_sentence, L2_collocations, L2_definition, L1_gloss contain ZERO braces.
- L2_collocations has exactly 3 items separated by '; ' (semicolon + space).
- No '|' anywhere. No names/digits/quotes.
""".strip()
)


def compose_instructions_en(L1_code: str, level: str, profile: str) -> str:
    """Return strict English instructions for the Responses API."""
    l1_meta = L1_LANGS[L1_code]
    level_rule = LEVEL_RULES_EN.get(level, "")
    profile_rule = PROMPT_PROFILES.get(profile, "")
    return PROMPT_TEMPLATE.substitute(
        L1_NAME=l1_meta["name"],
        LEVEL_RULE=level_rule,
        PROFILE_RULE=profile_rule,
    )
