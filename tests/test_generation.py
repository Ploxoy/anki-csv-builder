"""Тесты для core.generation.generate_card.

Проверяем базовые сценарии:
- успешная генерация с parsed-ответом и обновлением счётчиков сигнал-слов;
- fallback к текстовому JSON и автоматический repair-проход;
- усечение raw-response и фиксация флага в метаданных.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import core.generation as gen


class _DummyResponse:
    def __init__(self, parsed: Dict[str, str] | None = None, text: str = "") -> None:
        self.output_parsed = parsed
        self.output_text = text


@pytest.fixture
def dummy_row() -> Dict[str, str]:
    return {
        "woord": "aanraken",
        "def_nl": "iets met je hand voelen",
        "translation": "touch",
    }


@pytest.fixture
def settings() -> gen.GenerationSettings:
    return gen.GenerationSettings(
        model="gpt-4o-mini",
        L1_code="EN",
        L1_name="English",
        level="B1",
        profile="balanced",
        temperature=0.4,
        max_output_tokens=512,
        signalword_seed=123,
    )


def test_generate_card_with_parsed_response(monkeypatch, dummy_row, settings):
    """Функция должна брать parsed-ответ, обновлять usage и выставлять метаданные."""

    first_resp = _DummyResponse(
        parsed={
            "L2_word": "aanraken",
            "L2_cloze": "Hij wil {{c1::aanraken}} maar wacht nog even.",
            "L1_sentence": "He wants to touch but waits a moment.",
            "L2_collocations": "iemand aanraken; zacht aanraken; gezicht aanraken",
            "L2_definition": "iets of iemand met je hand voelen",
            "L1_gloss": "touch",
        },
        text="ignored",
    )

    def fake_send(*args, **kwargs):
        return first_resp, {
            "response_format_removed": False,
            "temperature_removed": False,
            "retries": 0,
        }

    call_log = []

    def fake_pick(groups, level, n, usage, last, force_balance, seed):
        call_log.append((level, n, seed))
        return ["omdat", "maar"]

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    monkeypatch.setattr(gen, "pick_allowed_for_level", fake_pick)

    result = gen.generate_card(
        client=object(),
        row=dummy_row,
        settings=settings,
        signalword_groups={"cause": {"B1": ["omdat", "want"]}},
        signal_usage={"omdat": 2},
        signal_last="want",
    )

    assert result.card["L2_cloze"].startswith("Hij wil {{c1::aanraken}}"), "cloze должен быть нормализован"
    assert result.signal_usage.get("maar") == 1, "обнаруженное сигнал-слово должно учитываться"
    assert result.signal_last == "maar"
    assert result.card["meta"]["model"] == settings.model
    assert not result.card["meta"]["raw_response_truncated"], "усечение не требуется"
    assert call_log == [("B1", settings.signalword_count, settings.signalword_seed)]


def test_generate_card_repair_from_text(monkeypatch, dummy_row, settings):
    """Если parsed пуст, берём JSON из текста и при необходимости запускаем repair."""

    # Первый ответ некорректный: нет collocations, поэтому валидатор должен заставить repair.
    bad_json = json.dumps(
        {
            "L2_word": "aanraken",
            "L2_cloze": "Hij {{c1::raakt}} de hond.",
            "L1_sentence": "He touches the dog.",
            "L2_collocations": "",
            "L2_definition": "iets met je hand voelen",
            "L1_gloss": "touch",
        }
    )
    good_json = json.dumps(
        {
            "L2_word": "aanraken",
            "L2_cloze": "Hij wil {{c1::aanraken}} zijn hond voorzichtig.",
            "L1_sentence": "He wants to touch his dog gently.",
            "L2_collocations": "iemand aanraken; zacht aanraken; voorzichtig aanraken",
            "L2_definition": "iets of iemand met je hand voelen",
            "L1_gloss": "touch",
        }
    )

    responses = [
        (
            _DummyResponse(parsed=None, text=f"Ответ:```json\n{bad_json}\n```"),
            {"response_format_removed": True, "temperature_removed": False, "retries": 0},
        ),
        (
            _DummyResponse(parsed=None, text=good_json),
            {"response_format_removed": False, "temperature_removed": False, "retries": 0},
        ),
    ]

    def fake_send(*_, **__):
        return responses.pop(0)

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    monkeypatch.setattr(gen, "pick_allowed_for_level", lambda *a, **k: ["omdat"])

    result = gen.generate_card(
        client=object(),
        row=dummy_row,
        settings=settings,
        signalword_groups=None,
        signalwords_b1=["omdat"],
        signal_usage={},
        signal_last=None,
    )

    meta = result.card["meta"]
    assert result.card["L2_collocations"].count(";") == 2, "repair должен заполнить три коллокации"
    assert meta["repair_attempted"], "ожидали запуск repair"
    assert not meta["problems_final"], "после repair проблем быть не должно"
    # Резервное fallback, когда нет групп, должен брать список B1.
    assert meta["allowed_signalwords"] == ["omdat"]
    assert meta["response_format_removed"], "ожидали флаг удаления schema"


def test_raw_response_truncation(monkeypatch, dummy_row, settings):
    """Длинный raw-response должен обрезаться и получать флаг truncated."""

    long_text = "{" + "a" * 4000 + "}"
    resp = _DummyResponse(parsed=None, text=long_text)

    def fake_send(*args, **kwargs):
        return resp, {
            "response_format_removed": False,
            "temperature_removed": False,
            "retries": 0,
        }

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    monkeypatch.setattr(gen, "pick_allowed_for_level", lambda *a, **k: [])

    result = gen.generate_card(
        client=object(),
        row=dummy_row,
        settings=settings,
        signalword_groups=None,
    )

    meta = result.card["meta"]
    assert meta["raw_response_truncated"], "должен быть установлен флаг усечения"
    assert len(meta["raw_response"]) <= gen.RAW_RESPONSE_MAX_LEN + 3, "усечённое значение не должно превышать лимит"
