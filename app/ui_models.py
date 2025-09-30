"""Shared dataclasses for UI configuration objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AudioConfig:
    """Settings describing available audio providers."""

    providers: Dict[str, Dict[str, Any]]
    default_provider: str


@dataclass
class ExportConfig:
    """Settings used for CSV/.apkg export panels."""

    csv_delimiter: str
    csv_lineterminator: str
    anki_model_id: int
    anki_deck_id: int
    anki_model_name: str
    anki_deck_name: str
    front_template: str
    back_template: str
    css: str


@dataclass
class GenerationRunContext:
    """Reusable LLM invocation settings calculated per run."""

    client: Any
    max_tokens: Optional[int]
    temperature: Optional[float]
    allow_response_format: bool
    force_flagged: bool
