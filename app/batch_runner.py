"""Batch processing helpers for the generation page."""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from core.generation import GenerationSettings, generate_card
from core.llm_clients import create_client

from . import ui_helpers
from .run_status import (
    ensure_run_stats,
    record_batch_stats,
    render_batch_header,
    update_overall_progress,
    update_run_summary,
)
from .sidebar import SidebarConfig
from .ui_models import GenerationRunContext


@dataclass
class BatchRunner:
    """Orchestrates batch generation runs and updates session state."""

    settings: SidebarConfig
    state: Any
    summary: Any
    overall_progress: Any
    overall_caption: Any
    api_delay: float
    signalword_groups: Optional[Dict]
    signalwords_b1: List[str]
    signalwords_b2_plus: List[str]

    def run_next_batch(self) -> None:
        run_ctx = _prepare_generation_run(self.state, self.settings)
        if run_ctx is None:
            return

        run_stats = ensure_run_stats(self.state)
        ui_helpers.init_signalword_state()
        ui_helpers.init_response_format_cache()
        if not self.state.get("anki_run_id"):
            self.state.anki_run_id = str(int(time.time()))
        self.state.model_id = self.settings.model

        total_items = len(self.state.input_data)
        start_idx = int(self.state.current_index or 0)
        batch_size = int(self.state.get("batch_size", 5))
        end_idx = min(start_idx + batch_size, total_items)
        if start_idx >= total_items:
            return

        indices = list(range(start_idx, end_idx))
        input_snapshot = list(self.state.input_data)
        workers = int(self.state.get("max_workers", 3))

        _, batch_prog, batch_status = render_batch_header(
            start_index=start_idx,
            end_index=end_idx,
            total=total_items,
            batch_size=len(indices),
            workers=workers,
        )
        batch_start_ts = time.time()
        results_map: Dict[int, dict] = {}
        completed = 0

        def _worker(idx: int, row: dict) -> Tuple[int, dict]:
            return self._generate_entry(
                idx,
                row,
                run_ctx,
                include_flag_reason=True,
                preserve_flagged_fields=False,
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker, idx, input_snapshot[idx]): idx for idx in indices}
            for future in as_completed(futures):
                idx, card = future.result()
                results_map[idx] = card
                completed += 1
                batch_prog.progress(min(1.0, completed / max(len(indices), 1)))
                elapsed = max(0.001, time.time() - batch_start_ts)
                active = max(0, min(workers, len(indices) - completed))
                queued = max(0, len(indices) - completed - active)
                rate = completed / elapsed
                batch_status.caption(
                    f"Done {completed}/{len(indices)} • Active ~{active} • Queued ~{queued} • "
                    f"{elapsed:.1f}s • {rate:.2f}/s"
                )
                done_tasks = start_idx + completed
                update_overall_progress(
                    self.overall_progress,
                    self.overall_caption,
                    processed=done_tasks,
                    total=total_items,
                )
                if self.api_delay > 0:
                    time.sleep(self.api_delay)

        usage = dict(self.state.get("sig_usage", {}))
        last = self.state.get("sig_last")
        batch_errors = 0
        batch_transient = 0
        for idx in indices:
            card = results_map.get(idx)
            if card is None:
                continue
            self.state.results.append(card)
            meta = card.get("meta", {}) or {}
            if meta.get("response_format_removed"):
                cache = set(self.state.get("no_response_format_models", set()))
                notified = set(self.state.get("no_response_format_notified", set()))
                if self.settings.model not in cache:
                    cache.add(self.settings.model)
                    self.state.no_response_format_models = cache
                if self.settings.model not in notified:
                    notified.add(self.settings.model)
                    self.state.no_response_format_notified = notified
                    detail = meta.get("response_format_error")
                    message = (
                        f"Model {self.settings.model} ignored schema (text.format); "
                        "falling back to text parsing for this session."
                    )
                    if detail:
                        message += f"\nReason: {detail}"
                    st.info(message, icon="ℹ️")
            found = meta.get("signalword_found")
            if found:
                usage[found] = usage.get(found, 0) + 1
                last = found
            err_text = (card.get("error") or "").lower()
            if err_text:
                batch_errors += 1
                if any(code in err_text for code in ("429", "rate", "timeout", "502", "503")):
                    batch_transient += 1

        self.state.sig_usage = usage
        self.state.sig_last = last
        self.state.current_index = end_idx
        overall_count = len(self.state.results)
        update_overall_progress(
            self.overall_progress,
            self.overall_caption,
            processed=overall_count,
            total=total_items,
        )
        batch_elapsed = max(0.001, time.time() - batch_start_ts)
        batch_status.caption(
            f"Batch finished in {batch_elapsed:.1f}s • {len(indices)/batch_elapsed:.2f}/s"
        )

        record_batch_stats(
            run_stats,
            items_processed=len(indices),
            errors=batch_errors,
            transient_errors=batch_transient,
            batch_started_at=batch_start_ts,
            batch_duration=batch_elapsed,
        )
        valid_total = sum(1 for card in self.state.results if not card.get("error"))
        update_run_summary(
            self.summary,
            run_stats,
            processed=overall_count,
            total=total_items,
            valid=valid_total,
        )

        if batch_transient >= 2 and self.state.get("max_workers", 3) > 1:
            self.state.max_workers = int(self.state.get("max_workers", 3)) - 1
            st.info(
                f"Transient errors detected ({batch_transient}); reducing max workers to {self.state.max_workers} for next batch.",
                icon="⚠️",
            )

    def rerun_errors(self) -> None:
        err_indices = []
        for card in self.state.get("results", []):
            meta = card.get("meta", {}) or {}
            idx = meta.get("input_index")
            if card.get("error") and isinstance(idx, int):
                err_indices.append(idx)

        if not err_indices:
            st.info("No errored cards to re-run.")
            return

        run_ctx = _prepare_generation_run(self.state, self.settings)
        if run_ctx is None:
            return

        st.info(f"Re-running {len(err_indices)} errored items…")
        progress = st.progress(0)
        results_map: Dict[int, dict] = {}

        def _worker(idx: int, row: dict) -> Tuple[int, dict]:
            return self._generate_entry(
                idx,
                row,
                run_ctx,
                include_flag_reason=False,
                preserve_flagged_fields=True,
            )

        with ThreadPoolExecutor(max_workers=int(self.state.get("max_workers", 3))) as ex:
            futures = {ex.submit(_worker, idx, self.state.input_data[idx]): idx for idx in err_indices}
            completed = 0
            for future in as_completed(futures):
                idx, card = future.result()
                results_map[idx] = card
                completed += 1
                progress.progress(min(1.0, completed / max(len(err_indices), 1)))

        new_results: List[dict] = []
        for card in self.state.get("results", []):
            meta = card.get("meta", {}) or {}
            idx = meta.get("input_index")
            if isinstance(idx, int) and idx in results_map:
                new_results.append(results_map[idx])
            else:
                new_results.append(card)
        self.state.results = new_results

        usage: Dict[str, int] = {}
        last = None
        for card in self.state.results:
            meta = card.get("meta", {}) or {}
            found = meta.get("signalword_found")
            if found:
                usage[found] = usage.get(found, 0) + 1
                last = found
        self.state.sig_usage = usage
        self.state.sig_last = last
        st.success("Errored items re-run completed.")

    def _generate_entry(
        self,
        idx: int,
        row: dict,
        run_ctx: GenerationRunContext,
        *,
        include_flag_reason: bool,
        preserve_flagged_fields: bool,
    ) -> Tuple[int, dict]:
        if not run_ctx.force_flagged and not row.get("_flag_ok", True):
            return idx, _flagged_card(
                row,
                idx,
                include_reason=include_flag_reason,
                preserve_fields=preserve_flagged_fields,
            )

        try:
            seed = random.randint(0, 2**31 - 1)
            gen_settings = _build_generation_settings(self.settings, run_ctx, seed)
            gen_result = generate_card(
                client=run_ctx.client,
                row=row,
                settings=gen_settings,
                signalword_groups=self.signalword_groups,
                signalwords_b1=self.signalwords_b1,
                signalwords_b2_plus=self.signalwords_b2_plus,
                signal_usage=None,
                signal_last=None,
            )
            card = gen_result.card
            meta = card.get("meta", {}) or {}
            meta["input_index"] = idx
            card["meta"] = meta
            return idx, card
        except Exception as exc:  # pragma: no cover
            return idx, _exception_card(exc, row, idx)


def _prepare_generation_run(state: Any, settings: SidebarConfig) -> Optional[GenerationRunContext]:
    """Create a reusable generation context or report a missing SDK."""

    client = create_client(settings.api_key)
    if client is None:
        st.error("OpenAI SDK not available; install the openai package to continue.")
        return None

    max_tokens = 3000 if settings.limit_tokens else None
    temperature = settings.temperature if ui_helpers.should_pass_temperature(settings.model) else None
    no_rf_models = set(state.get("no_response_format_models", set()))
    force_schema = state.get("force_schema_checkbox", False)
    allow_response_format = settings.model not in no_rf_models or force_schema
    force_flagged = state.get("force_flagged", False)

    return GenerationRunContext(
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        allow_response_format=allow_response_format,
        force_flagged=force_flagged,
    )


def _build_generation_settings(
    settings: SidebarConfig, run_ctx: GenerationRunContext, seed: int
) -> GenerationSettings:
    """Construct GenerationSettings using shared runtime context."""

    return GenerationSettings(
        model=settings.model,
        L1_code=settings.L1_code,
        L1_name=settings.L1_meta["name"],
        level=settings.level,
        profile=settings.profile,
        temperature=run_ctx.temperature,
        max_output_tokens=run_ctx.max_tokens,
        allow_response_format=run_ctx.allow_response_format,
        signalword_seed=seed,
    )


def _flagged_card(
    row: Dict[str, Any],
    idx: int,
    *,
    include_reason: bool,
    preserve_fields: bool,
) -> Dict[str, Any]:
    """Return a placeholder card for flagged items."""

    card = {
        "L2_word": row.get("woord", ""),
        "L2_cloze": "",
        "L1_sentence": "",
        "L2_collocations": "",
        "L2_definition": "",
        "L1_gloss": "",
        "L1_hint": "",
        "AudioSentence": "",
        "AudioWord": "",
        "error": "flagged_precheck",
    }
    if preserve_fields:
        card["L2_definition"] = row.get("def_nl", "")
        card["L1_gloss"] = row.get("translation", "")

    meta: Dict[str, Any] = {"input_index": idx}
    if include_reason:
        meta["flag_reason"] = row.get("_flag_reason", "")
    card["meta"] = meta
    return card


def _exception_card(exc: Exception, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Produce a fallback card when generation raises an exception."""

    return {
        "L2_word": row.get("woord", ""),
        "L2_cloze": "",
        "L1_sentence": "",
        "L2_collocations": "",
        "L2_definition": row.get("def_nl", ""),
        "L1_gloss": row.get("translation", ""),
        "L1_hint": "",
        "AudioSentence": "",
        "AudioWord": "",
        "error": f"exception: {exc}",
        "meta": {"input_index": idx},
    }
