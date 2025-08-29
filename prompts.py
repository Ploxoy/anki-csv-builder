"""
Prompt engineering module for CEFR-level specific instruction generation.
Handles dynamic L1 language support and signal word inclusion logic.
"""

import random
from typing import Dict
from string import Template  # add this import at the top of prompts.py

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

def get_cefr_specific_instructions(cefr_level):
    """
    Get CEFR level-specific instruction modifications.
    
    Args:
        cefr_level (str): CEFR level
    
    Returns:
        str: Additional instructions specific to the CEFR level
    """
    
    cefr_modifications = {
        "A1": """
        - Use simple present tense and basic vocabulary
        - Keep sentences short and straightforward
        - Focus on everyday topics and common situations
        """,
        
        "A2": """
        - Include simple past and future tenses
        - Use familiar topics and routine activities
        - Introduce basic connecting words
        """,
        
        "B1": """
        - Use variety of tenses including perfect tenses
        - Include more complex sentence structures
        - Cover familiar and some unfamiliar topics
        - May include basic signaalwoorden
        """,
        
        "B2": """
        - Use sophisticated grammar structures
        - Include abstract concepts and complex ideas
        - Use varied vocabulary and expressions
        - Include signaalwoorden when appropriate
        """,
        
        "C1": """
        - Use advanced grammar and complex structures
        - Include nuanced meanings and subtle distinctions
        - Cover wide range of topics including specialized areas
        - Use signaalwoorden effectively
        """,
        
        "C2": """
        - Use near-native level complexity
        - Include idiomatic expressions and cultural references
        - Master all grammatical structures
        - Use signaalwoorden naturally and effectively
        """
    }
    
    return cefr_modifications.get(cefr_level, cefr_modifications["B1"])

def get_separable_verb_instruction():
    """
    Get specific instructions for handling separable verbs in cloze format.
    
    Returns:
        str: Instructions for proper separable verb cloze formatting
    """
    
    return """
    For separable verbs (like opruimen, aanraken):
    - Use format: {{c1::stem}} ... {{c2::particle}} 
    - Example: Ik ga mijn kamer {{c1::op}}ruimen -> Ik {{c1::ruim}} mijn kamer {{c2::op}}
    - Ensure both parts are marked as separate cloze deletions
    """

def get_language_specific_notes(L1_language):
    """
    Get language-specific notes for translation and definition quality.
    
    Args:
        L1_language (str): Target L1 language code
    
    Returns:
        str: Language-specific instruction notes
    """
    
    language_notes = {
        "ru": "Provide natural Russian translations avoiding literal word-for-word translation.",
        "en": "Use clear, concise English definitions and translations.",
        "de": "Provide German translations that respect grammatical differences from Dutch.",
        "fr": "Use appropriate French equivalents, considering false friends between Dutch and French.",
        "es": "Provide Spanish translations that are natural and contextually appropriate.",
    }
    
    return language_notes.get(L1_language, "Provide natural translations in the target language.")

