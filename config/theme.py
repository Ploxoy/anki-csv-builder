"""Doedutch UI theme configuration derived from the UI style guide."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent
_STYLES_DIR = _HERE / "styles"
_BASE_STYLESHEET = _STYLES_DIR / "doedutch.css"

LIGHT_COLOR_ROLES: Dict[str, str] = {
    "background": "#f8f8f8",
    "background_alt": "#f1f4f8",
    "surface": "#ffffff",
    "surface_soft": "#fcfcfc",
    "sidebar": "#ffffff",
    "border": "#e0e0e0",
    "border_strong": "#d0d5dd",
    "primary": "#3178c6",
    "primary_hover": "#255a9b",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "text_primary": "#333333",
    "text_secondary": "#6c757d",
    "text_muted": "#8c98a4",
    "input_bg": "#ffffff",
}

DARK_COLOR_ROLES: Dict[str, str] = {
    "background": "#111827",
    "background_alt": "#1f2937",
    "surface": "#1b2433",
    "surface_soft": "#202b3c",
    "sidebar": "#1a2130",
    "border": "#273244",
    "border_strong": "#334055",
    "primary": "#4d9ff6",
    "primary_hover": "#3a84d1",
    "success": "#4ade80",
    "warning": "#facc15",
    "error": "#f87171",
    "text_primary": "#f8f9fb",
    "text_secondary": "#c5d0e3",
    "text_muted": "#93a2c2",
    "input_bg": "#111827",
}

TYPOGRAPHY: Dict[str, str | int] = {
    "body_font": "Inter, 'Segoe UI', sans-serif",
    "heading_font": "'Nunito Sans', 'Segoe UI', sans-serif",
    "mono_font": "'JetBrains Mono', 'Consolas', 'Courier New', monospace",
    "base_font_size_px": 16,
    "heading_scale": 1.25,
}

LAYOUT: Dict[str, str | int] = {
    "max_width": "1200px",
    "radius_sm": "6px",
    "radius_md": "8px",
    "radius_lg": "12px",
}


@lru_cache(maxsize=1)
def load_theme_css() -> str:
    """Return the Doedutch base stylesheet (light + dark) as a cached string."""

    return _BASE_STYLESHEET.read_text(encoding="utf-8")
