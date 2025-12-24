"""Prompt composition utilities for card generation."""
from __future__ import annotations

from string import Template

from config.settings import L1_LANGS, LEVEL_RULES_EN, PROMPT_PROFILES

PROMPT_TEMPLATE = Template(
    """
You are an expert Dutch→$L1_NAME lexicographer and didactics writer.
Produce a STRICT JSON object with fields ONLY:
- L2_word           : the Dutch target lemma OR multi-word expression/collocation to learn (no POS tags)
- L2_cloze          : ONE short natural Dutch sentence containing cloze markup
- L1_sentence       : a simple learner-friendly translation of that sentence into $L1_NAME (not necessarily word-for-word)
- L2_collocations   : EXACTLY 3 natural Dutch collocations with the target, joined by '; '
- L2_definition     : ONE short Dutch definition
- L1_gloss          : 1–6 $L1_NAME words (short gloss, correct POS/meaning)

GLOBAL HARD RULES
- Output JSON ONLY. No comments, no extra keys, no trailing text.
- No field may be empty. Do NOT use the '|' character anywhere.
- Modern Dutch only; avoid names, digits, and quotes. Punctuation (commas, ellipses, etc.) is allowed when natural.
- Present tense by default unless the sentence truly requires otherwise.

INPUT FIELDS (FROM DATA_JSON)
- The input JSON contains:
  - "L2_word"              : the Dutch target (word or multi-word expression) exactly as given by the user.
  - "given_L2_definition"  : EITHER (a) a dictionary-style Dutch definition, OR (b) a full Dutch example sentence where the target appears in context.
  - "preferred_L1_gloss"   : an optional hint for L1_gloss.
- If "given_L2_definition" is a dictionary-style definition, keep L2_definition short and clear, fully compatible with that sense.
- If "given_L2_definition" is a Dutch sentence, treat it as the main context: infer the sense from that sentence and base BOTH L2_definition and L1_sentence on its meaning. L1_sentence may be slightly didactic/paraphrastic, but must preserve all key information from the Dutch sentence.

CLOZE RULES (STRICT, ANKI-COMPATIBLE)
- Use ONLY the c1 family. Do NOT use c2/c3/etc. at all.
- Each cloze must have EXACTLY two curly braces: '{{' and '}}' (not single, not triple, not quadruple).
- For regular (non-separable) single-word targets: mark the whole target surface form as {{c1::...}}.
- For separable verbs: use MULTI-SPAN c1 — stem and particle are BOTH {{c1::...}}, e.g.:
  GOOD  : Ik {{c1::ruim}} mijn kamer {{c1::op}}.
  BAD   : Ik {{c1::ruim}} mijn kamer {{c2::op}}.        (never use c2)
  BAD   : Ik {{{{c1::ruim}}}} mijn kamer {{c1::op}}.     (never 4 braces)
- Never cloze only a suffix or ending (e.g., '-t', '-en'), or only a prefix ('ge-', 'te-').
- Never cloze function words: articles (de/het/een), pronouns (ik, je, hij, ze, we), clitics ('n), particles like 'te', 'er', 'ge-'.
- Never cloze signal words (conjunctions/connectors).
- Never wrap the WHOLE sentence; only the target form(s).

MULTI-WORD TARGETS IN CLOZE
- If L2_word is a multi-word expression that contains a finite verb plus a fixed complement (e.g. "naar bed gaan"), you MUST treat them together as the target and use MULTI-SPAN c1 on the verb AND the complement in L2_cloze, e.g.:
  GOOD  : Ik {{c1::ga}} altijd op tijd {{c1::naar bed}} als ik vroeg moet opstaan.
  BAD   : Ik ga altijd op tijd {{c1::naar bed}} als ik vroeg moet opstaan.   (only the complement is clozed)
  BAD   : Ik {{c1::ga}} altijd op tijd naar bed als ik vroeg moet opstaan.   (only the verb is clozed)

TRANSLATION & OTHER FIELDS
- L1_sentence must contain ZERO braces — absolutely no {{ }} there.
- L2_collocations must contain ZERO braces — plain text, joined by '; ' (exactly 3 items).
- L2_definition and L1_gloss must contain ZERO braces and no pipes.

SIGNAL WORDS (B1+ only, about 50% of the time)
- If CEFR ≥ B1, you MAY include exactly ONE appropriate Dutch signal word in L2_cloze.
- Do NOT start the sentence with it; place it naturally mid-clause.
- NEVER cloze the signal word.

PLACEMENT OF SEPARABLE VERBS (CEFR ≥ B1)
- If the target is a separable verb and grammar allows, prefer a main-clause word order where the stem and the particle are separated (particle near the clause end), e.g., "Ik {{c1::ruim}} de kamer {{c1::op}}." rather than keeping the infinitive "opruimen" together.
- Do not force separation in contexts where it is ungrammatical or unnatural (e.g., te + infinitive, past participles like "opgeruimd", subclauses with different word order). Keep the sentence natural and idiomatic.
- Always cloze BOTH stem and particle with c1 (never use c2).
- If the input L2_word is given as a finite form + particle (e.g. "kwam erachter"), normalize L2_word to the infinitive lemma with particle (e.g. "achter komen") but in L2_cloze still cloze the finite verb + particle as they appear in the sentence, e.g.:
  INPUT  L2_word: "kwam erachter"
  GOOD   L2_word: "achter komen"
  L2_cloze: "Hij {{c1::kwam}} {{c1::erachter}} dat hij een fout had gemaakt."

TARGET FORM (LEMMA VS PHRASE)
- L2_word in the OUTPUT must always be a teachable target: either a single Dutch word (lemma) OR the multi-word expression/collocation from the input L2_word.
- If the input L2_word is a SINGLE countable noun (even if given in plural or without an article, e.g. "boeken"), write L2_word as "de <lemma>" or "het <lemma>" with the CORRECT article in singular (e.g. "het boek"). In L2_cloze you may still use any natural form (singular/plural/with other determiners).
- If the input L2_word is a SINGLE verb in a finite or past form (e.g. "werkte"), write L2_word as the infinitive lemma (e.g. "werken"). In L2_cloze you may still use the finite/past form that fits the sentence.
- If the input L2_word is a multi-word expression or collocation (e.g. "naar bed gaan", "het is me wat..."), keep the full expression as L2_word in the output. Do NOT reduce it to a single lemma and do NOT prepend an extra article.
- Punctuation (commas, ellipses, etc.) is allowed inside L2_word when it is a multi-word expression, as long as the phrase is natural Dutch.

GOOD / BAD TARGET FORM EXAMPLES
- Noun (input plural, no article):
  INPUT  L2_word: "boeken"
  GOOD  L2_word: "het boek"
  BAD   L2_word: "boeken"
- Verb (input past tense):
  INPUT  L2_word: "werkte"
  GOOD  L2_word: "werken"
  BAD   L2_word: "werkte"

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
- Modal + infinitive (no separation):
  GOOD : Je mag het schilderij niet {{c1::aanraken}}.
  BAD  : Je mag het schilderij niet {{c1::aan}} {{c1::raken}}.
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
