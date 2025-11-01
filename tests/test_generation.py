"""Тесты для core.generation.generate_card.

Проверяем базовые сценарии:
- успешная генерация с parsed-ответом и обновлением счётчиков сигнал-слов;
- fallback к текстовому JSON и автоматический repair-проход;
- усечение raw-response и фиксация флага в метаданных.
"""
from __future__ import annotations

import json
import logging
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
            "cached_tokens": 900,
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
    assert result.card["error"] == ""
    assert not result.card["meta"]["raw_response_truncated"], "усечение не требуется"
    assert call_log == [("B1", settings.signalword_count, settings.signalword_seed)]
    req_meta = result.card["meta"]["request"]
    assert req_meta["cached_tokens"] == 900
    assert req_meta["prompt_tokens"] == 0
    assert req_meta["completion_tokens"] == 0


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
    assert meta["error"] == ""


def test_raw_response_truncation(monkeypatch, dummy_row, settings):
    """Длинный raw-response должен обрезаться и получать флаг truncated."""

    valid_json = json.dumps(
        {
            "L2_word": "aanraken",
            "L2_cloze": "Hij wil {{c1::aanraken}} de hond.",
            "L1_sentence": "He wants to touch the dog.",
            "L2_collocations": "iemand aanraken; zacht aanraken; hond aanraken",
            "L2_definition": "iets met je hand voelen",
            "L1_gloss": "touch",
        }
    )
    long_text = valid_json + "\n" + ("a" * 4000)
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
    assert meta["error"] == ""


def test_generate_card_without_signalword_when_disabled(monkeypatch, dummy_row, settings):
    """При include_signalword=False подбор слов не выполняется, мета пустая."""

    settings = gen.GenerationSettings(
        model=settings.model,
        L1_code=settings.L1_code,
        L1_name=settings.L1_name,
        level="A2",
        profile=settings.profile,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        include_signalword=False,
        signalword_seed=settings.signalword_seed,
    )

    resp = _DummyResponse(
        parsed={
            "L2_word": "aanraken",
            "L2_cloze": "Ik wil {{c1::aanraken}} de hond voorzichtig.",
            "L1_sentence": "I want to touch the dog carefully.",
            "L2_collocations": "iemand aanraken; voorzichtig aanraken; zacht aanraken",
            "L2_definition": "iets of iemand met je hand voelen",
            "L1_gloss": "touch",
        },
        text="",
    )

    def fake_send(*args, **kwargs):
        assert kwargs.get("response_format") is not None
        return resp, {
            "response_format_removed": False,
            "temperature_removed": False,
            "retries": 0,
        }

    def boom(*_args, **_kwargs):  # should never be called when include_signalword=False
        raise AssertionError("signalword selector should not be invoked")

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    monkeypatch.setattr(gen, "pick_allowed_for_level", boom)
    monkeypatch.setattr(gen, "_fallback_signalwords", boom)

    result = gen.generate_card(
        client=object(),
        row=dummy_row,
        settings=settings,
        signalword_groups={"contrast": {"B1": ["maar"]}},
        signal_usage=None,
        signal_last=None,
    )

    meta = result.card["meta"]
    assert meta["allowed_signalwords"] == []
    assert not meta["include_signalword"], "при выключенном флаге не должны просить сигнал-слово"
    assert meta["signalword_found"] is None
    assert result.signal_usage == {}
    assert result.signal_last is None
    assert result.card["error"] == ""


def test_generate_card_fallback_signalwords_for_b2(monkeypatch, dummy_row, settings):
    """Фолбэк списков B2 выбирает наименее использованные и учитывает last."""

    settings = gen.GenerationSettings(
        model=settings.model,
        L1_code=settings.L1_code,
        L1_name=settings.L1_name,
        level="B2",
        profile=settings.profile,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        include_signalword=True,
        signalword_seed=None,
    )

    parsed = {
        "L2_word": "aanraken",
        "L2_cloze": "Hij wil {{c1::aanraken}} de hond, bovendien leert hij geduld.",
        "L1_sentence": "He wants to touch the dog, moreover he learns patience.",
        "L2_collocations": "iemand aanraken; zacht aanraken; dier aanraken",
        "L2_definition": "iets of iemand met je hand voelen",
        "L1_gloss": "touch",
    }

    calls = []

    def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return _DummyResponse(parsed=parsed), {
            "response_format_removed": False,
            "temperature_removed": False,
            "retries": 0,
        }

    def forbid_group_path(*_args, **_kwargs):
        raise AssertionError("group path should not be used")

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    # ensure groups path not used so fallback executes
    monkeypatch.setattr(gen, "pick_allowed_for_level", forbid_group_path)

    usage = {"echter": 5, "daarentegen": 1}
    result = gen.generate_card(
        client=object(),
        row=dummy_row,
        settings=settings,
        signalword_groups=None,
        signalwords_b1=["maar"],
        signalwords_b2_plus=["echter", "bovendien", "daarentegen"],
        signal_usage=usage,
        signal_last="echter",
    )

    meta = result.card["meta"]
    assert meta["allowed_signalwords"] == ["bovendien", "daarentegen"], "ожидали выбор без последнего и с учётом usage"
    assert meta["signalword_found"] == "bovendien"
    assert result.signal_usage["bovendien"] == 1
    assert result.signal_last == "bovendien"
    assert calls[0]["response_format"] is not None
    assert result.card["error"] == ""


def test_generate_card_handles_llm_failure(monkeypatch, dummy_row, settings, caplog):
    """Ошибки клиента должны возвращать пустую карточку с понятным error."""

    settings.include_signalword = False

    def fake_send(*args, **kwargs):
        raise RuntimeError("boom")

    signal_usage = {"omdat": 2}

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    monkeypatch.setattr(gen, "pick_allowed_for_level", lambda *a, **k: [])

    with caplog.at_level(logging.ERROR):
        result = gen.generate_card(
            client=object(),
            row=dummy_row,
            settings=settings,
            signalword_groups=None,
            signal_usage=signal_usage,
            signal_last="omdat",
        )

    assert result.card["L2_word"] == "aanraken"
    assert result.card["error"].startswith("llm_request_failed"), "ожидали пометку об ошибке"
    assert result.card["meta"]["error_stage"] == "llm_request"
    assert result.signal_usage == signal_usage, "usage возвращается без изменений"
    assert result.signal_usage is not signal_usage, "должна возвращаться копия"
    assert any("Generation request failed" in rec.message for rec in caplog.records)


def test_generate_card_marks_validation_error(monkeypatch, dummy_row, settings, caplog):
    """Если после repair карточка всё ещё невалидна, error должен выставляться."""

    bad_card = {
        "L2_word": "aanraken",
        "L2_cloze": "Hij {{c1::raakt}} de hond.",
        "L1_sentence": "He touches the dog.",
        "L2_collocations": "aanraken",
        "L2_definition": "iets met je hand voelen",
        "L1_gloss": "touch",
    }

    responses = [
        (
            _DummyResponse(parsed=bad_card, text=json.dumps(bad_card)),
            {"response_format_removed": False, "temperature_removed": False, "retries": 0},
        ),
        (
            _DummyResponse(parsed=bad_card, text=json.dumps(bad_card)),
            {"response_format_removed": False, "temperature_removed": False, "retries": 0},
        ),
    ]

    def fake_send(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(gen, "send_responses_request", fake_send)
    monkeypatch.setattr(gen, "pick_allowed_for_level", lambda *a, **k: [])

    with caplog.at_level(logging.WARNING):
        result = gen.generate_card(
            client=object(),
            row=dummy_row,
            settings=settings,
            signalword_groups=None,
        )

    assert "validation_failed" in result.card["error"]
    assert result.card["meta"]["error_stage"] == "validation"
    assert result.card["meta"]["problems_final"], "должны сохранить список проблем"
    assert result.card["meta"]["repair_attempted"], "ожидали попытку repair"
    assert any("Validation failed after generation" in rec.message for rec in caplog.records)
