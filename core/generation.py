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
import random
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
from core.prompts import compose_instructions_en

__all__ = [
    "GenerationSettings",
    "GenerationResult",
    "generate_card",
    "extract_json_block",
    "should_include_signalword",
]

logger = logging.getLogger(__name__)

RAW_RESPONSE_MAX_LEN = 1500
ERROR_MESSAGE_MAX_LEN = 200


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
    seed: Optional[int] = None,
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

    rng = random.Random(seed) if seed is not None else None
    used = usage or {}
    scored = []
    for word in base:
        if word == last:
            continue
        jitter = rng.random() if rng is not None else 0.0
        scored.append((used.get(word, 0), jitter, word))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    result = [word for _, _, word in scored][:count]
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


FORBIDDEN_CLOZE_TOKENS = {"de", "het", "een", "daardoor", "daarom", "dus", "maar", "omdat", "want", "terwijl", "hoewel", "zodat", "doordat", "bovendien", "echter", "bijvoorbeeld", "tenzij", "ondanks", "desondanks", "daarentegen", "aangezien", "zodra", "voordat", "nadat"}



def _strip_cloze(text: str) -> str:
    if not text:
        return ""
    import re
    return re.sub(r'\{\{.*?\}\}', '', text)


def _sanitize_c1_spans(sentence: str, target: str) -> str:
    if not sentence:
        return ""
    import re
    target_tokens = [t.strip() for t in target.lower().split()] if target else []

    def _fix_match(match):
        inner = match.group(1).strip()
        token = inner.lower()
        if token in FORBIDDEN_CLOZE_TOKENS:
            return inner
        if len(token) <= 2 and token not in target_tokens:
            return inner
        return match.group(0)
    sentence = re.sub(r'\{\{c1::(.*?)\}\}', _fix_match, sentence)
    sentence = re.sub(r'\{\{c2::(.*?)\}\}', r'\1', sentence)
    sentence = sentence.replace('{{{{', '{{').replace('}}}}', '}}')
    return sentence


def _apply_cloze_fixes(card: Dict[str, str]) -> None:
    clz = normalize_cloze_braces(card.get("L2_cloze", ""))
    if "{{c1::" not in clz:
        clz = force_wrap_first_match(card.get("L2_word", ""), clz)
    clz = try_separable_verb_wrap(card.get("L2_word", ""), clz)
    clz = _sanitize_c1_spans(clz, card.get("L2_word", ""))
    card["L2_cloze"] = clz


def _base_card_from_row(row: Dict[str, Any]) -> Dict[str, str]:
    """Возвращает заготовку карточки, если генерация не удалась."""

    lemma = sanitize(row.get("woord", ""))
    definition = _strip_cloze(sanitize(row.get("def_nl", "")))
    gloss = _strip_cloze(
        sanitize(
            row.get("translation")
            or row.get("ru_short")
            or row.get("L1_gloss")
            or ""
        )
    )
    return {
        "L2_word": lemma,
        "L2_cloze": "",
        "L1_sentence": "",
        "L2_collocations": "",
        "L2_definition": definition,
        "L1_gloss": gloss,
        "L1_hint": "",
        "AudioSentence": "",
        "AudioWord": "",
        "error": "",
    }


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
                seed=settings.signalword_seed,
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
        f'"L1_gloss": "<1-6 {settings.L1_name} words>"'
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

    def _finalize_with_error(
        *,
        stage: str,
        error_message: str,
        send_meta: Optional[Dict[str, Any]] = None,
        raw_response: str = "",
        raw_trimmed: bool = False,
    ) -> GenerationResult:
        send_meta = send_meta or {}
        card = _base_card_from_row(row)
        clean_error = error_message.strip()
        if len(clean_error) > ERROR_MESSAGE_MAX_LEN:
            clean_error = clean_error[: ERROR_MESSAGE_MAX_LEN - 3] + "..."
        card["error"] = clean_error

        meta: Dict[str, Any] = {
            "allowed_signalwords": allowed_signalwords,
            "include_signalword": bool(include_sig and allowed_signalwords),
            "signalword_found": None,
            "signalword_seed": settings.signalword_seed,
            "problems_initial": [],
            "problems_final": [],
            "repair_attempted": False,
            "raw_response": raw_response,
            "raw_response_truncated": raw_trimmed,
            "response_format_removed": send_meta.get("response_format_removed", False),
            "response_format_error": send_meta.get("response_format_error"),
            "temperature_removed": send_meta.get("temperature_removed", False),
            "error": clean_error,
            "error_stage": stage,
        }
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
            signal_usage=dict(signal_usage or {}),
            signal_last=signal_last,
        )

    # Collect request info for debugging (trim long fields for meta)
    _instr_short, _instr_trim = _trim_text(instructions)
    _input_short, _input_trim = _trim_text(input_text)
    request_info: Dict[str, Any] = {
        "model": settings.model,
        "allow_response_format": settings.allow_response_format,
        "response_format": None,  # will be set below if used
        "response_format_used": False,
        "max_output_tokens": settings.max_output_tokens,
        "temperature": settings.temperature,
        "instructions": _instr_short,
        "instructions_truncated": _instr_trim,
        "input_text": _input_short,
        "input_text_truncated": _input_trim,
    }

    try:
        rf_param = json_schema if settings.allow_response_format else None
        # reflect actual use in request_info (before call)
        if rf_param is not None:
            request_info["response_format"] = "json_schema"
            request_info["response_format_used"] = True
        response, send_meta = send_responses_request(
            client=client,
            model=settings.model,
            instructions=instructions,
            input_text=input_text,
            response_format=rf_param,
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Generation request failed",
            extra={
                "model": settings.model,
                "level": settings.level,
                "profile": settings.profile,
                "woord": row.get("woord", ""),
                "stage": "llm_request",
            },
        )
        return _finalize_with_error(
            stage="llm_request",
            error_message=f"llm_request_failed: {exc}",
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

    try:
        raw_text_full = get_response_text(response)
        raw_text, raw_trimmed = _trim_text(raw_text_full)
        parsed = get_response_parsed(response) or extract_json_block(raw_text_full)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Response parsing failed",
            extra={
                "model": settings.model,
                "level": settings.level,
                "profile": settings.profile,
                "woord": row.get("woord", ""),
                "stage": "parse_response",
            },
        )
        return _finalize_with_error(
            stage="parse_response",
            error_message=f"response_parse_failed: {exc}",
            send_meta=send_meta,
        )

    card = {
        "L2_word": sanitize(parsed.get("L2_word", payload["L2_word"])),
        "L2_cloze": sanitize(parsed.get("L2_cloze", "")),
        "L1_sentence": _strip_cloze(sanitize(parsed.get("L1_sentence", ""))),
        "L2_collocations": _strip_cloze(sanitize(parsed.get("L2_collocations", ""))),
        "L2_definition": _strip_cloze(sanitize(parsed.get("L2_definition", payload.get("given_L2_definition", "")))),
        "L1_gloss": _strip_cloze(sanitize(parsed.get("L1_gloss", payload.get("preferred_L1_gloss", "")))),
        "L1_hint": "",
        "error": "",
    }
    # Audio placeholders filled later by TTS pipeline
    card.setdefault("AudioWord", "")
    card.setdefault("AudioSentence", "")

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
            rf_repair = json_schema if allow_schema_next else None
            repair_resp, repair_meta = send_responses_request(
                client=client,
                model=settings.model,
                instructions=repair_prompt,
                input_text=repair_input,
                response_format=rf_repair,
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
        else:
            # capture repair request info for debugging
            _rinstr, _rinstr_trim = _trim_text(repair_prompt)
            _rinput, _rinput_trim = _trim_text(repair_input)
            request_info["repair_request"] = {
                "response_format": "json_schema" if allow_schema_next else None,
                "instructions": _rinstr,
                "instructions_truncated": _rinstr_trim,
                "input_text": _rinput,
                "input_text_truncated": _rinput_trim,
            }

    problems_final = validate_card(card)

    if problems_final:
        card["error"] = "validation_failed: " + "; ".join(problems_final)
        logger.warning(
            "Validation failed after generation",
            extra={
                "woord": row.get("woord", ""),
                "model": settings.model,
                "stage": "validation",
                "problems": problems_final,
            },
        )

    usage_updated, last_updated, signal_found = note_signalword_in_sentence(
        card.get("L2_cloze", ""),
        allowed_signalwords,
        usage=signal_usage,
        last=signal_last,
    )

    # post-call: note if schema was removed by SDK
    request_info["response_format_removed"] = send_meta.get("response_format_removed", False)
    request_info["temperature_removed"] = send_meta.get("temperature_removed", False)
    request_info["retries"] = send_meta.get("retries", 0)
    if send_meta.get("response_format_error"):
        request_info["response_format_error"] = send_meta.get("response_format_error")

    meta: Dict[str, Any] = {
        "allowed_signalwords": allowed_signalwords,
        "include_signalword": bool(include_sig and allowed_signalwords),
        "signalword_found": signal_found,
        "signalword_seed": settings.signalword_seed,
        "problems_initial": problems_initial,
        "problems_final": problems_final,
        "repair_attempted": repair_attempted,
        "raw_response": raw_text,
        "raw_response_truncated": raw_trimmed,
        "response_format_removed": send_meta.get("response_format_removed", False),
        "response_format_error": send_meta.get("response_format_error"),
        "temperature_removed": send_meta.get("temperature_removed", False),
        "error": card.get("error", ""),
        "error_stage": "validation" if card.get("error") else None,
        "request": request_info,
    }
    if repair_raw:
        meta["repair_response"] = repair_raw
        meta["repair_response_truncated"] = repair_trimmed
        meta["repair_response_format_removed"] = repair_meta.get("response_format_removed", False)
        meta["repair_response_format_error"] = repair_meta.get("response_format_error")
        meta["repair_temperature_removed"] = repair_meta.get("temperature_removed", False)

    meta_full = {
        **meta,
        "model": settings.model,
        "level": settings.level,
        "profile": settings.profile,
        "error": card.get("error", ""),
    }
    card["meta"] = meta_full

    return GenerationResult(
        card=card,
        meta=meta_full,
        signal_usage=usage_updated,
        signal_last=last_updated,
    )
