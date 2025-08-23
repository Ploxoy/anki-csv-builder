from typing import Dict

try:
    from config import L1_LANGS
except Exception:
    # Минимальный фоллбэк, если config недоступен
    L1_LANGS = {
        "RU": {"label": "RU", "name": "Russian", "csv_translation": "Перевод", "csv_gloss": "Перевод слова"}
    }

# ==========================
# CEFR правила/длины и стили профилей
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
# Системный промпт (опционально)
# ==========================

PROMPT_SYSTEM: str = (
    "Ты — опытный лексикограф NL→RU и автор учебных материалов. "
    "Сгенерируй СТРОГО JSON-объект карточки Anki со структурой: "
    "{woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short}.\n"
    "ОБЩИЕ ПРАВИЛА (ОЧЕНЬ ВАЖНО):\n"
    "• Верни ТОЛЬКО JSON БЕЗ пояснений и форматирования.\n"
    "• НИ ОДНО поле не пустое. Запрещены пустые строки.\n"
    "• Символ '|' в текстах запрещён.\n"
    "• Если дано def_nl — строго следуй ему; не меняй базовое значение слова.\n"
    "• Сохраняй часть речи: ru_short должен соответствовать части речи слова "
    "(глагол→инфинитив; существительное→существительное; прилагательное→прилагательное).\n"
    "• ru_sentence — ТОЧНЫЙ перевод NL-предложения, без перефраза.\n"
    "• cloze_sentence — одно короткое естественное NL-предложение (8–14 слов, настоящее время, "
    "без имён/цифр/кавычек); целевое слово внутри {{c1::…}}.\n"
    "  Если слово — разделимый глагол: {{c1::stam}} … {{c2::partikel}}. Иначе только {{c1::…}}.\n"
    "• collocaties — РОВНО 3 частотные связки, разделитель '; ' (точка с запятой и пробел).\n"
    "  Каждая связка — 2–3 слова с целевым словом в естественной форме. Нельзя: бессмысленные пары "
    "(например, 'een grote caissière'), редкие/книжные, имена собственные.\n"
    "• Избегай редкой лексики; используй A2–B1 вокруг целевого слова.\n\n"
    "ФОРМАТ ВЫВОДА: один JSON-объект с ключами: woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short.\n\n"
)

# ==========================
# Построение инструкций для модели (EN)
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
- Cloze: wrap the target in {{c1::...}}. If the word is a separable verb, use {{c1::stem}} … {{c2::particle}}; otherwise ONLY {{c1::...}} (no {{c2::...}}).
- The Dutch sentence: natural; present tense by default; avoid names, digits, and quotes; modern Dutch; keep length within CEFR constraints.
- L1_sentence: an exact, faithful translation.
- L2_collocations: EXACTLY three frequent, natural combinations with the target word; join using '; '. Avoid odd or infrequent pairings and proper names. Signal words MAY appear here if natural, but are NOT required.
- L2_definition: short Dutch definition. L1_gloss: 1–2 words in {L1_name}; obey any provided Dutch definition.

CLOZE COMPLIANCE (VERY IMPORTANT):
- You MUST use exactly TWO curly braces on both sides: {{c1::...}} (and {{c2::...}} for separable verbs).
- Never use single braces {c1::...} or other bracket styles.
- BAD: {c1::raak}  GOOD: {{c1::raak}}
- If the verb is NOT separable, do NOT output {{c2::...}} at all.
- Ensure the final sentence contains at least one {{c1::...}} and all braces are balanced.

""".strip()

    lvl = f"CEFR: {level}. {level_rule}".strip()
    prof = f"Style: {profile_rule}".strip()

    return base + "\n\n" + lvl + "\n" + prof


