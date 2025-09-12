"""
core/signalwords.py

Pure, Streamlit-agnostic helpers for building and choosing signal words ("signaalwoorden").

API:
- build_signal_pool(signalword_groups, level) -> dict[group, list[word]]
- choose_signalwords(pool, n=3, usage=None, last=None, force_balance=False) -> list[word]
- note_signalword_in_sentence(sentence, allowed, usage=None, last=None) -> (new_usage, new_last, found_word_or_None)

The functions do not depend on Streamlit; callers may persist usage/last in session state.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import random


def _levels_up_to(level: str) -> List[str]:
    """Return inclusive list of levels from A1 up to the provided level."""
    ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
    try:
        idx = ORDER.index(level)
    except ValueError:
        idx = 2  # default to B1
    return ORDER[: idx + 1]


def build_signal_pool(signalword_groups: Dict[str, Dict[str, List[str]]], level: str) -> Dict[str, List[str]]:
    """Build a per-group pool including all entries up to `level` (inclusive).

    signalword_groups format expected:
      { group_name: { 'A1': [...], 'A2': [...], ... }, ... }

    Returns dict[group_name, list_of_words]. Empty dict if signalword_groups is falsy.
    """
    pool: Dict[str, List[str]] = {}
    if not signalword_groups:
        return pool
    lvls = set(_levels_up_to(level))
    for grp, by_lvl in signalword_groups.items():
        items: List[str] = []
        for lv, arr in by_lvl.items():
            if lv in lvls:
                items.extend(arr or [])
        if items:
            pool[grp] = items
    return pool


def choose_signalwords(
    pool: Dict[str, List[str]],
    n: int = 3,
    usage: Optional[Dict[str, int]] = None,
    last: Optional[str] = None,
    force_balance: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
    """Choose up to `n` candidate signal words from the pool.

    - pool: mapping group -> list[word]
    - usage: optional dict word->count (lower count preferred)
    - last: optional last-used word to avoid immediate repetition
    - force_balance: if True, prefer one word per distinct group when possible
    - seed: optional int to make selection deterministic for tests

    Returns list of words (len <= n). The result is stable given same inputs and seed.
    """
    if not pool:
        return []

    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random

    # Flatten with (word, group) pairs
    pairs: List[Tuple[str, str]] = []
    for g, arr in pool.items():
        for w in arr:
            pairs.append((w, g))

    # Sort by usage (ascending) then by group/name to stabilize
    used = usage or {}
    pairs.sort(key=lambda wg: (used.get(wg[0], 0), wg[1], wg[0]))

    result: List[str] = []
    seen_groups: set[str] = set()

    # If force_balance, attempt to pick words from distinct groups first
    for w, g in pairs:
        if len(result) >= n:
            break
        if w == last:
            continue
        if force_balance and g in seen_groups:
            continue
        # Randomize tie-break within same priority to avoid always picking lexical first
        # But keep deterministic if seed provided by shuffling equal-usage block
        result.append(w)
        seen_groups.add(g)

    # If not enough found, fill from remaining sorted pairs
    if len(result) < n:
        for w, g in pairs:
            if len(result) >= n:
                break
            if w == last or w in result:
                continue
            result.append(w)

    # If there are more than needed (edge cases), truncate
    return result[:n]


def note_signalword_in_sentence(
    sentence: str,
    allowed: List[str],
    usage: Optional[Dict[str, int]] = None,
    last: Optional[str] = None,
) -> Tuple[Dict[str, int], Optional[str], Optional[str]]:
    """Detect which allowed signalword appears in `sentence`, update usage and last.

    Returns (new_usage_dict, new_last, found_word_or_None).
    - usage dict is a shallow-copied dict (if provided) with increments applied.
    - last is updated to found word when detected.
    - If none found, returns original usage/last and None.
    """
    if not sentence or not allowed:
        return (usage or {}, last, None)
    low = sentence.lower()
    u = dict(usage or {})
    found = None
    for w in allowed:
        if w.lower() in low:
            u[w] = u.get(w, 0) + 1
            found = w
            last = w
            break
    return (u, last, found)


# Helper: simple balanced choose that accepts signalword_groups like config.SIGNALWORD_GROUPS
def pick_allowed_for_level(
    signalword_groups: Dict[str, Dict[str, List[str]]],
    level: str,
    n: int = 3,
    usage: Optional[Dict[str, int]] = None,
    last: Optional[str] = None,
    force_balance: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
    """Convenience: build the pool from groups and call choose_signalwords."""
    pool = build_signal_pool(signalword_groups, level)
    return choose_signalwords(pool, n=n, usage=usage, last=last, force_balance=force_balance, seed=seed)
