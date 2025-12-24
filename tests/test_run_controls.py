from __future__ import annotations

from types import SimpleNamespace

from app.run_controls import RunController
from app.sidebar import SidebarConfig


class DummyState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _settings() -> SidebarConfig:
    return SidebarConfig(
        api_key="sk",
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


def test_schedule_auto_continue_triggers_rerun(monkeypatch):
    state = DummyState(
        input_data=[1, 2, 3],
        current_index=0,
        auto_advance=True,
        auto_continue=False,
        run_active=True,
    )

    rerun_called = {}
    monkeypatch.setattr("app.run_controls.st.rerun", lambda: rerun_called.setdefault("called", True))

    controller = RunController(
        settings=_settings(),
        state=state,
        process_batch=lambda: None,
        rerun_errored=lambda: None,
    )

    controller._schedule_auto_continue()

    assert rerun_called.get("called") is True
    assert state.auto_continue is True
