"""Persistent cache helpers for response_format support probes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


CACHE_DIR = Path("cache")
CACHE_PATH = CACHE_DIR / "response_format.json"


@dataclass
class ProbeRecord:
    supported: bool
    reason: Optional[str] = None
    updated_at: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, object]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ProbeRecord":
        return cls(
            supported=bool(data.get("supported", True)),
            reason=data.get("reason") if isinstance(data.get("reason"), str) else None,
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


def load_response_format_cache() -> Dict[str, ProbeRecord]:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    cache: Dict[str, ProbeRecord] = {}
    if isinstance(raw, dict):
        for model, payload in raw.items():
            if isinstance(model, str) and isinstance(payload, dict):
                cache[model] = ProbeRecord.from_dict(payload)
    return cache


def save_response_format_cache(cache: Dict[str, ProbeRecord]) -> None:
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {model: record.to_dict() for model, record in cache.items()}
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_probe_cache(
    cache: Dict[str, ProbeRecord],
    *,
    model: str,
    supported: bool,
    reason: Optional[str] = None,
) -> Dict[str, ProbeRecord]:
    if not model:
        return cache
    cache[model] = ProbeRecord(supported=supported, reason=reason)
    return cache
