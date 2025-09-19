"""core.generation

Оркестрация генерации одной карточки Anki: построение payload, выбор
сигнальных слов, вызов LLM, локальная нормализация и валидация.

Модуль не зависит от Streamlit и может использоваться как из UI, так и
из фоновых задач. Функция `generate_card` возвращает карточку и
метаданные вместе с обновлённым состоянием сигнальных слов.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from core.llm_clients import (
    send_responses_request,
    get_response_parsed,
    get_response_text,
)
from core.signalwords import (
    pick_allowed_for_level,
    note_signalword_in_sentence,
)
from core.sanitize_validate import (
    sanitize,
    normalize_cloze_braces,
    force_wrap_first_match,
    try_separable_verb_wrap,
    validate_card,
)
from prompts import compose_instructions_en

__all__ = [
    "GenerationSettings",
    "GenerationResult",
    "generate_card",
    "extract_json_block",
    "should_include_signalword",
]

logger = logging.getLogger(__name__)

RAW_RESPONSE_MAX_LEN = 1500


@dataclass
class GenerationSettings:
    """Параметры генерации для одной карточки."""

    model: str
    L1_code: str
    L1_name: str
    level: str
    profile: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    include_signalword: Optional[bool] = None  # None => авто-решение по уровню
    signalword_count: int = 3
    signalword_seed: Optional[int] = None
    allow_response_format: bool = True


@dataclass
class GenerationResult:
    """Результат генерации карточки."""

    card: Dict[str, str]
    meta: Dict[str, Any]
    signal_usage: Dict[str, int]
    signal_last: Optional[str]


def should_include_signalword(woord: str, level: str, probability: float = 0.5) -> bool:
    """Детерминированно решаем, нужно ли просить модель добавить сигнал-слово."""

    if level not in {"B1", "B2", "C1", "C2"}:
        return False
    if not woord:
        return False
    # Хэшируем слово+уровень, чтобы выбор был детерминирован
    seed = int(hashlib.sha256(f"{woord}|{level}".encode()).hexdigest(), 16)
    threshold = int(probability * 10_000)
    return seed % 10_000 < threshold


def _fallback_signalwords(
    level: str,
    *,
    count: int,
    usage: Optional[Dict[str, int]] = None,
    last: Optional[str] = None,
    b1_list: Optional[Sequence[str]] = None,
    b2_list: Optional[Sequence[str]] = None,
) -> List[str]:
    if level not in {"B1", "B2", "C1", "C2"}:
        return []
    base: Sequence[str] = []
    if level == "B1":
        base = b1_list or []
    else:
        base = b2_list or []
    if not base:
        return []
    used = usage or {}
    ordered = sorted(base, key=lambda w: (used.get(w, 0), w))
    result: List[str] = []
    for word in ordered:
        if word == last:
            continue
        result.append(word)
        if len(result) >= count:
            break
    return result


RE_CODE_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
RE_FIRST_OBJECT = re.compile(r"\{[\s\S]*?\}")


def _try_parse_candidate(text: str) -> Dict:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _brace_scan_pick(text: str) -> Dict:
    opens: List[int] = []
    for idx, ch in enumerate(text):
        if ch == "{":
            opens.append(idx)
        elif ch == "}" and opens:
            start = opens.pop()
            candidate = text[start : idx + 1]
            obj = _try_parse_candidate(candidate)
            if obj:
                return obj
    return {}


def extract_json_block(text: str) -> Dict:
    """Извлечь JSON из произвольного текста ответа модели."""

    if not text:
        return {}

    match = RE_CODE_FENCE.search(text)
    if match:
        snippet = match.group(1).strip()
        obj = _try_parse_candidate(snippet)
        if obj:
            return obj
        obj = _brace_scan_pick(snippet)
        if obj:
            return obj

    match2 = RE_FIRST_OBJECT.search(text)
    if match2:
        obj = _try_parse_candidate(match2.group(0))
        if obj:
            return obj

    return _brace_scan_pick(text)


def _apply_cloze_fixes(card: Dict[str, str]) -> None:
    clz = normalize_cloze_braces(card.get("L2_cloze", ""))
    if "{{c1::" not in clz:
        clz = force_wrap_first_match(card.get("L2_word", ""), clz)
    clz = try_separable_verb_wrap(card.get("L2_word", ""), clz)
    card["L2_cloze"] = clz


def _trim_text(text: str, limit: int = RAW_RESPONSE_MAX_LEN) -> tuple[str, bool]:
    if not text:
        return "", False
    if len(text) <= limit:
        return text, False
    logger.debug("Trimming raw response from %s to %s chars", len(text), limit)
    return text[:limit] + "...", True


def generate_card(
    client: Any,
    row: Dict[str, Any],
    settings: GenerationSettings,
    *,
    signalword_groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
    signalwords_b1: Optional[Sequence[str]] = None,
    signalwords_b2_plus: Optional[Sequence[str]] = None,
    signal_usage: Optional[Dict[str, int]] = None,
    signal_last: Optional[str] = None,
) -> GenerationResult:
    """Сгенерировать карточку и вернуть её вместе с метаданными."""

    instructions = compose_instructions_en(settings.L1_code, settings.level, settings.profile)

    include_sig = (
        settings.include_signalword
        if settings.include_signalword is not None
        else should_include_signalword(row.get("woord", ""), settings.level)
    )

    force_balance = settings.level in {"B2", "C1", "C2"}
    allowed_signalwords: List[str] = []
    if include_sig:
        if signalword_groups:
            allowed_signalwords = pick_allowed_for_level(
                signalword_groups,
                settings.level,
                n=settings.signalword_count,
                usage=signal_usage,
                last=signal_last,
                force_balance=force_balance,
                seed=settings.signalword_seed,
            )
        else:
            allowed_signalwords = _fallback_signalwords(
                settings.level,
                count=settings.signalword_count,
                usage=signal_usage,
                last=signal_last,
                b1_list=signalwords_b1,
                b2_list=signalwords_b2_plus,
            )

    translation_hint = (
        row.get("translation")
        or row.get("ru_short")
        or row.get("L1_gloss")
        or ""
    )

    payload = {
        "L2_word": row.get("woord", "").strip(),
        "given_L2_definition": row.get("def_nl", "").strip(),
        "preferred_L1_gloss": str(translation_hint).strip(),
        "L1": settings.L1_name,
        "CEFR": settings.level,
        "INCLUDE_SIGNALWORD": bool(include_sig and allowed_signalwords),
        "ALLOWED_SIGNALWORDS": allowed_signalwords,
    }

    json_template = (
        '{'
        '"L2_word": "<Dutch lemma>", '
        '"L2_cloze": "ONE Dutch sentence with {{c1::...}} (and {{c2::...}} only if separable)", '
        f'"L1_sentence": "<exact translation into {settings.L1_name}>", '
        '"L2_collocations": "colloc1; colloc2; colloc3", '
        '"L2_definition": "<short Dutch definition>", '
        f'"L1_gloss": "<1-2 {settings.L1_name} words>"'
        '}'
    )

    json_schema = {
        "name": "AnkiClozeCard",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "L2_word",
                "L2_cloze",
                "L1_sentence",
                "L2_collocations",
                "L2_definition",
                "L1_gloss",
            ],
            "properties": {
                "L2_word": {"type": "string", "minLength": 1},
                "L2_cloze": {"type": "string", "minLength": 1},
                "L1_sentence": {"type": "string", "minLength": 1},
                "L2_collocations": {"type": "string", "minLength": 1},
                "L2_definition": {"type": "string", "minLength": 1},
                "L1_gloss": {"type": "string", "minLength": 1},
            },
        },
    }

    input_text = (
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\nReply with STRICT JSON ONLY. It must match this template exactly (same keys, one-line JSON):\n"
        + json_template
    )

    response, send_meta = send_responses_request(
        client=client,
        model=settings.model,
        instructions=instructions,
        input_text=input_text,
        response_format=json_schema if settings.allow_response_format else None,
        max_output_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
    )
    logger.debug(
        "Generation request completed",
        extra={
            "model": settings.model,
            "level": settings.level,
            "profile": settings.profile,
            "woord": row.get("woord", ""),
        },
    )

    raw_text_full = get_response_text(response)
    raw_text, raw_trimmed = _trim_text(raw_text_full)
    parsed = get_response_parsed(response) or extract_json_block(raw_text_full)

    card = {
        "L2_word": sanitize(parsed.get("L2_word", payload["L2_word"])),
        "L2_cloze": sanitize(parsed.get("L2_cloze", "")),
        "L1_sentence": sanitize(parsed.get("L1_sentence", "")),
        "L2_collocations": sanitize(parsed.get("L2_collocations", "")),
        "L2_definition": sanitize(parsed.get("L2_definition", payload.get("given_L2_definition", ""))),
        "L1_gloss": sanitize(parsed.get("L1_gloss", payload.get("preferred_L1_gloss", ""))),
        "L1_hint": "",
        "error": "",
    }

    _apply_cloze_fixes(card)
    problems_initial = validate_card(card)

    repair_attempted = False
    repair_raw = ""
    repair_trimmed = False
    allow_schema_next = settings.allow_response_format and not send_meta.get("response_format_removed", False)

    if problems_initial:
        repair_attempted = True
        repair_prompt = (
            instructions
            + "\n\nREPAIR: The previous JSON has issues: "
            + "; ".join(problems_initial)
            + ". Fix ONLY the problematic fields and return STRICT JSON again."
        )
        repair_input = "Previous JSON:\n" + json.dumps(card, ensure_ascii=False)
        repair_meta: Dict[str, Any] = {}
        try:
            repair_resp, repair_meta = send_responses_request(
                client=client,
                model=settings.model,
                instructions=repair_prompt,
                input_text=repair_input,
                response_format=json_schema if allow_schema_next else None,
                max_output_tokens=settings.max_output_tokens,
                temperature=settings.temperature,
            )
            logger.debug(
                "Repair request executed",
                extra={"woord": row.get("woord", ""), "model": settings.model},
            )
            repair_full = get_response_text(repair_resp)
            repair_raw, repair_trimmed = _trim_text(repair_full)
            parsed_repair = get_response_parsed(repair_resp) or extract_json_block(repair_full)
        except Exception:
            parsed_repair = {}

        if parsed_repair:
            for key in [
                "L2_word",
                "L2_cloze",
                "L1_sentence",
                "L2_collocations",
                "L2_definition",
                "L1_gloss",
            ]:
                if key in parsed_repair:
                    card[key] = sanitize(parsed_repair.get(key, card.get(key, "")))
            _apply_cloze_fixes(card)

    problems_final = validate_card(card)

    usage_updated, last_updated, signal_found = note_signalword_in_sentence(
        card.get("L2_cloze", ""),
        allowed_signalwords,
        usage=signal_usage,
        last=signal_last,
    )

    meta: Dict[str, Any] = {
        "allowed_signalwords": allowed_signalwords,
        "include_signalword": bool(include_sig and allowed_signalwords),
        "signalword_found": signal_found,
        "problems_initial": problems_initial,
        "problems_final": problems_final,
        "repair_attempted": repair_attempted,
        "raw_response": raw_text,
        "raw_response_truncated": raw_trimmed,
        "response_format_removed": send_meta.get("response_format_removed", False),
        "temperature_removed": send_meta.get("temperature_removed", False),
    }
    if repair_raw:
        meta["repair_response"] = repair_raw
        meta["repair_response_truncated"] = repair_trimmed
        meta["repair_response_format_removed"] = repair_meta.get("response_format_removed", False)
        meta["repair_temperature_removed"] = repair_meta.get("temperature_removed", False)

    meta_full = {
        **meta,
        "model": settings.model,
        "level": settings.level,
        "profile": settings.profile,
    }
    card["meta"] = meta_full

    return GenerationResult(
        card=card,
        meta=meta_full,
        signal_usage=usage_updated,
        signal_last=last_updated,
    )
