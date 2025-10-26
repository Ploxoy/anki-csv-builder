"""Run control widgets and controller logic for batch generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import streamlit as st

from .sidebar import SidebarConfig
from .run_status import reset_run_stats


ControlActions = Dict[str, bool]


def render_run_controls() -> ControlActions:
    """Render run control buttons and return their click states."""

    col_start, col_next, col_stop, col_rerun = st.columns([1, 1, 1, 1])
    return {
        "start": col_start.button("Start run", type="primary"),
        "next": col_next.button("Next batch"),
        "stop": col_stop.button("Stop run"),
        "rerun": col_rerun.button("Re-run errored only"),
    }


@dataclass
class RunController:
    """State machine that reacts to run control actions."""

    settings: SidebarConfig
    state: Any
    process_batch: Callable[[], None]
    rerun_errored: Callable[[], None]

    def handle(self, actions: ControlActions) -> None:
        """Apply run control state transitions and batch handling."""

        if actions.get("start"):
            self._start_run()
        elif actions.get("next"):
            self._next_batch()
        elif actions.get("stop"):
            self.state.run_active = False
            self.state.auto_continue = False
            st.info("Run paused. Use Next batch or Start run to continue.")
        elif actions.get("rerun"):
            if not self.settings.api_key:
                st.error("Provide OPENAI_API_KEY before retrying errors.")
            else:
                self.rerun_errored()

        self._auto_advance_if_needed()

    def _start_run(self) -> None:
        if not self.settings.api_key:
            st.error("Provide OPENAI_API_KEY via Secrets, environment variable, or the input field.")
            return

        self.state.results = []
        self.state.audio_media = {}
        self.state.audio_summary = None
        self.state.current_index = 0
        reset_run_stats(self.state)
        self.state.run_active = True
        client = self._ensure_client()
        if client is not None:
            from . import ui_helpers

            ui_helpers.probe_response_format_support(client, self.settings.model)
        self.process_batch()
        self._schedule_auto_continue()

    def _next_batch(self) -> None:
        if not self.state.get("run_active"):
            self.state.run_active = True
        if not self.settings.api_key:
            st.error("Provide OPENAI_API_KEY before running batches.")
            return

        self.process_batch()
        self._schedule_auto_continue()

    def _auto_advance_if_needed(self) -> None:
        total = self._total_items()
        if not (
            self.state.get("auto_continue")
            and self.state.get("run_active")
            and self.state.current_index < total
        ):
            return

        if not self.state.get("auto_advance"):
            self.state.auto_continue = False
            return

        if not self.settings.api_key:
            self.state.auto_continue = False
            self.state.run_active = False
            st.error("Provide OPENAI_API_KEY before running batches.")
            return

        self.state.auto_continue = False
        self.process_batch()
        self._schedule_auto_continue()

    def _schedule_auto_continue(self) -> None:
        total = self._total_items()
        if not (self.state.get("auto_advance") and self.state.current_index < total):
            return

        self.state.auto_continue = True
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    def _total_items(self) -> int:
        input_data = getattr(self.state, "input_data", None)
        if isinstance(input_data, list):
            return len(input_data)
        return 0

    def _ensure_client(self) -> Any:
        from core.llm_clients import create_client

        try:
            return create_client(self.settings.api_key)
        except Exception:
            return None
