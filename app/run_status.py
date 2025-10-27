"""Run status helpers for generation progress tracking."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import streamlit as st

RunStats = Dict[str, Any]

_DEFAULT_RUN_STATS: RunStats = {
    "batches": 0,
    "items": 0,
    "elapsed": 0.0,
    "errors": 0,
    "transient": 0,
    "start_ts": None,
}


def ensure_run_stats(state: Any) -> RunStats:
    """Ensure session state has a run_stats dict with default keys."""
    stats = getattr(state, "run_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    merged: RunStats = dict(_DEFAULT_RUN_STATS)
    merged.update({k: v for k, v in stats.items() if k in _DEFAULT_RUN_STATS})
    state.run_stats = merged
    return merged


def reset_run_stats(state: Any) -> RunStats:
    """Reset run statistics to defaults."""
    stats = dict(_DEFAULT_RUN_STATS)
    state.run_stats = stats
    return stats


def record_batch_stats(
    stats: RunStats,
    *,
    items_processed: int,
    errors: int,
    transient_errors: int,
    batch_started_at: float,
    batch_duration: float,
) -> None:
    """Update run statistics after a batch finishes."""
    if not stats.get("start_ts"):
        stats["start_ts"] = batch_started_at
    stats["batches"] += 1
    stats["items"] += items_processed
    stats["elapsed"] += batch_duration
    stats["errors"] += errors
    stats["transient"] += transient_errors


def format_run_caption(
    stats: RunStats,
    *,
    processed: int,
    total: int,
    valid: Optional[int] = None,
) -> str:
    """Return a human-readable caption for the run summary."""
    valid_part = f" • valid {valid}" if valid is not None else ""
    start_ts = stats.get("start_ts")
    if start_ts:
        total_elapsed = max(0.001, time.time() - float(start_ts))
        rate = stats["items"] / total_elapsed if total_elapsed > 0 else 0.0
        return (
            f"Run: batches {stats['batches']} • processed {processed}/{total}{valid_part} • "
            f"elapsed {total_elapsed:.1f}s • {rate:.2f}/s • errors {stats['errors']} "
            f"(transient {stats['transient']})"
        )
    return f"Run: processed {processed}/{total}{valid_part}"


def update_run_summary(
    placeholder: Any,
    stats: RunStats,
    *,
    processed: int,
    total: int,
    valid: Optional[int] = None,
) -> None:
    """Write the run summary caption into the provided placeholder."""
    if placeholder is None:
        return
    placeholder.caption(format_run_caption(stats, processed=processed, total=total, valid=valid))


def update_overall_progress(
    progress: Any,
    caption: Any,
    *,
    processed: int,
    total: int,
) -> None:
    """Update the overall progress bar and its caption."""
    pct = processed / max(total, 1)
    if progress is not None:
        progress.progress(min(1.0, pct))
    if caption is not None:
        caption.caption(f"Overall: {processed}/{total} processed")


def render_batch_header(
    *,
    start_index: int,
    end_index: int,
    total: int,
    batch_size: int,
    workers: int,
) -> Tuple[Any, Any, Any]:
    """Create batch UI placeholders (header, progress, status)."""
    header = st.empty()
    header.caption(
        f"Batch {start_index+1}–{end_index} of {total} • size {batch_size} • workers {workers}"
    )
    return header, st.progress(0), st.empty()
