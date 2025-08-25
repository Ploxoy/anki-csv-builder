from typing import Dict

try:
    from config import L1_LANGS
except Exception:
    # Minimal fallback if config unavailable
    L1_LANGS = {
        "RU": {"label": "RU", "name": "Russian", "csv_translation": "Translation", "csv_gloss": "Word gloss"}
    }

# ==========================
# CEFR rules/lengths and profile styles
# ==========================

CEFR_LENGTHS: Dict[str, tuple[int, int] | None] = {
    "A1": (6, 9),
    "A2": (8, 12),
    "B1": (10, 14),
    "B2": (12, 16),
    "C1": (14, 18),
    "C2": None,  # no limit
}

LEVEL_RULES_EN: Dict[str, str] = {
    "A1": "Use only very basic surrounding vocabulary. 6–9 words. No subordinate clauses, no passive, no perfect tenses.",
    "A2": "Basic vocabulary. 8–12 words. May use modal verbs (kunnen, moeten) and simple past (was/had); still no complex clauses.",
    "B1": "10–14 words. You MAY use a simple subordinate clause (omdat/als/terwijl). In roughly 50% of cases include ONE suitable Dutch signal word.",
    "B2": "12–16 words. More complex structures allowed; passive allowed. Keep sentence natural. In ~50% of cases include ONE signal word from the extended list.",
    "C1": "14–18 words. Advanced structures allowed; neutral-formal style.",
    "C2": "No length limit; native-like naturalness and precision.",
}

PROMPT_PROFILES: Dict[str, str] = {
    "strict": "Be literal and concise; avoid figurative language; keep the simplest structure that satisfies CEFR.",
    "balanced": "Natural and clear; minor synonymy allowed if it improves fluency.",
    "exam": "Neutral-formal register; precise; avoid colloquialisms.",
    "creative": "Allow mild figurativeness if it keeps clarity and CEFR constraints.",
}

# ==========================
# Building instructions for model (EN)
# ==========================

def compose_instructions_en(L1_code: str, level: str, profile: str) -> str:
    L1_name = L1_LANGS[L1_code]["name"]
    level_rule = LEVEL_RULES_EN[level]
    profile_rule = PROMPT_PROFILES.get(profile, "")

    base = f"""
You are an expert Dutch→{L1_name} lexicographer and didactics writer.
Return a STRICT JSON object with fields:
- L2_word (the Dutch target word/lemma),
- L2_cloze (ONE short natural Dutch sentence with cloze),
- L1_sentence (an exact translation of that sentence into {L1_name}),
- L2_collocations (EXACTLY 3 frequent Dutch collocations that contain the target word, joined with '; '),
- L2_definition (ONE short Dutch definition),
- L1_gloss (1–2 words in {L1_name} matching the word's part of speech and meaning).

Hard requirements:
- Output JSON ONLY, no explanations. No field may be empty. Do not use the '|' character.
- For cloze format use EXACTLY double curly braces: "{{{{c1::target}}}}" (with exactly two opening and two closing braces).
- For separable verbs use "{{{{c1::stem}}}} … {{{{c2::particle}}}}" format.
- Never use single braces or triple braces.
- Example 1 (regular): "Ik {{{{c1::begrijp}}}} deze zin."
- Example 2 (separable): "Ik {{{{c1::ruim}}}} mijn kamer {{{{c2::op}}}}."

The Dutch sentence must be:
- Natural and contextually clear
- Present tense by default
- Avoid names, digits, quotes
- Modern Dutch only
- Keep within CEFR length constraints
""".strip()

    lvl = f"CEFR: {level}. {level_rule}".strip()
    prof = f"Style: {profile_rule}".strip()

    return base + "\n\n" + lvl + "\n" + prof
- L2_cloze (ONE short natural Dutch sentence with cloze),
- L1_sentence (an exact translation of that sentence into {L1_name}),
- L2_collocations (EXACTLY 3 frequent Dutch collocations that contain the target word, joined with '; '),
- L2_definition (ONE short Dutch definition),
- L1_gloss (1–2 words in {L1_name} matching the word's part of speech and meaning).

Hard requirements:
- Output JSON ONLY, no explanations. No field may be empty. Do not use the '|' character.
- For cloze format use EXACTLY double curly braces: "{{c1::target}}" (with exactly two opening and two closing braces).
- For separable verbs use "{{c1::stem}} … {{c2::particle}}" format.
- Never use single braces or triple braces.
- Example 1 (regular): "Ik {{c1::begrijp}} deze zin."
- Example 2 (separable): "Ik {{c1::ruim}} mijn kamer {{c2::op}}."

The Dutch sentence must be:
- Natural and contextually clear
- Present tense by default
- Avoid names, digits, quotes
- Modern Dutch only
- Keep within CEFR length constraints
""".strip()

    lvl = f"CEFR: {level}. {level_rule}".strip()
    prof = f"Style: {profile_rule}".strip()

    return base + "\n\n" + lvl + "\n" + prof


