from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import batch_runner
from app.batch_runner import BatchRunner
from app.sidebar import SidebarConfig


class DummyState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class DummyProgress:
    def __init__(self):
        self.values: list[float] = []

    def progress(self, value: float) -> None:
        self.values.append(value)


class DummyStatus:
    def __init__(self):
        self.messages: list[str] = []

    def caption(self, text: str) -> None:
        self.messages.append(text)


class DummySummary:
    def __init__(self):
        self.messages: list[str] = []

    def caption(self, text: str) -> None:
        self.messages.append(text)


def _make_settings() -> SidebarConfig:
    return SidebarConfig(
        api_key="sk-test",
        model="gpt-5",
        profile="strict",
        level="B1",
        L1_code="EN",
        L1_meta={"name": "English"},
        temperature=0.4,
        limit_tokens=False,
        display_raw_response=False,
        csv_with_header=True,
        csv_anki_header=True,
        force_flagged=False,
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch, progress: DummyProgress, status: DummyStatus) -> None:
    monkeypatch.setattr(batch_runner.ui_helpers, "init_signalword_state", lambda: None)
    monkeypatch.setattr(batch_runner.ui_helpers, "init_response_format_cache", lambda: None)
    monkeypatch.setattr(batch_runner, "build_run_report", lambda state: None)
    monkeypatch.setattr(batch_runner, "update_overall_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "update_run_summary", lambda *args, **kwargs: None)

    def fake_render_batch_header(**_: object):
        return None, progress, status

    monkeypatch.setattr(batch_runner, "render_batch_header", fake_render_batch_header)
    monkeypatch.setattr(batch_runner.st, "info", lambda *args, **kwargs: None)

    context = SimpleNamespace(
        client=object(),
        max_tokens=None,
        temperature=None,
        allow_response_format=True,
        force_flagged=False,
    )
    monkeypatch.setattr(batch_runner, "_prepare_generation_run", lambda state, settings: context)


def test_run_next_batch_accumulates_results(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = DummyProgress()
    status = DummyStatus()
    _patch_common(monkeypatch, progress, status)

    rows = [
        {"woord": "eerste", "_flag_ok": True},
        {"woord": "tweede", "_flag_ok": True},
    ]

    state = DummyState(
        input_data=rows,
        results=[],
        current_index=0,
        batch_size=2,
        max_workers=3,
    )

    def fake_generate_card(*_, **kwargs):
        row = kwargs["row"]
        card = {
            "L2_word": row["woord"],
            "L2_cloze": f"Ik {{c1::{row['woord']}}}.",
            "L1_sentence": "Sentence",
            "L2_collocations": "a; b; c",
            "L2_definition": "def",
            "L1_gloss": "gloss",
            "error": "",
            "meta": {"response_format_removed": False, "error": ""},
        }
        return SimpleNamespace(card=card, meta=card["meta"], signal_usage={}, signal_last=None)

    monkeypatch.setattr(batch_runner, "generate_card", fake_generate_card)

    runner = BatchRunner(
        settings=_make_settings(),
        state=state,
        summary=DummySummary(),
        overall_progress=None,
        overall_caption=None,
        api_delay=0.0,
        signalword_groups=None,
        signalwords_b1=[],
        signalwords_b2_plus=[],
    )

    runner.run_next_batch()

    assert len(state.results) == 2
    assert all(card["error"] == "" for card in state.results)
    assert state.current_index == 2
    assert state.get("max_workers_runtime") == 3
    assert progress.values[-1] == 1.0


def test_run_next_batch_reduces_workers_on_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = DummyProgress()
    status = DummyStatus()
    info_messages: list[str] = []

    _patch_common(monkeypatch, progress, status)
    monkeypatch.setattr(batch_runner.st, "info", lambda message, **_: info_messages.append(message))

    rows = [
        {"woord": "een", "_flag_ok": True},
        {"woord": "twee", "_flag_ok": True},
        {"woord": "drie", "_flag_ok": True},
    ]

    state = DummyState(
        input_data=rows,
        results=[],
        current_index=0,
        batch_size=3,
        max_workers=4,
    )

    def fake_generate_card(*_, **kwargs):
        row = kwargs["row"]
        error_text = "HTTP 429 transient"
        card = {
            "L2_word": row["woord"],
            "L2_cloze": f"Wij {{c1::{row['woord']}}} veel.",
            "L1_sentence": "Sentence",
            "L2_collocations": "a; b; c",
            "L2_definition": "def",
            "L1_gloss": "gloss",
            "error": error_text,
            "meta": {"response_format_removed": False, "error": error_text},
        }
        return SimpleNamespace(card=card, meta=card["meta"], signal_usage={}, signal_last=None)

    monkeypatch.setattr(batch_runner, "generate_card", fake_generate_card)

    runner = BatchRunner(
        settings=_make_settings(),
        state=state,
        summary=DummySummary(),
        overall_progress=None,
        overall_caption=None,
        api_delay=0.0,
        signalword_groups=None,
        signalwords_b1=[],
        signalwords_b2_plus=[],
    )

    runner.run_next_batch()

    assert len(state.results) == 3
    assert all("429" in card["error"] for card in state.results)
    assert state.get("max_workers_pending") == 3  # reduced from 4 to 3
    assert state.get("max_workers_runtime") == 3
    assert any("reducing max workers" in msg for msg in info_messages)
