"""Centralized pricing tables for text and audio models (USD)."""
from __future__ import annotations

from typing import Dict

# Approximate Responses API pricing (USD per 1M tokens), aligned with
# https://platform.openai.com/docs/pricing?latest-pricing=standard#text-tokens
# Update these values when OpenAI adjusts rates.
MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5.1": {"input": 1.25, "output": 10.0},
    "gpt-5.1-chat-latest": {"input": 1.25, "output": 10.0},
    "gpt-5.1-codex-max": {"input": 1.25, "output": 10.0},
    "gpt-5.1-codex": {"input": 1.25, "output": 10.0},
    "gpt-5.1-codex-mini": {"input": 0.25, "output": 2.0},
    "gpt-5": {"input": 1.25, "output": 10.0},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.0},
    "gpt-5-codex": {"input": 1.25, "output": 10.0},
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "output": 0.4},
    "gpt-5-pro": {"input": 15.0, "output": 120.0},
    "gpt-5-search-api": {"input": 1.25, "output": 10.0},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.6},
    "gpt-4o-search-preview": {"input": 2.5, "output": 10.0},
    "gpt-realtime": {"input": 4.0, "output": 16.0},
    "gpt-realtime-mini": {"input": 0.6, "output": 2.4},
    "gpt-4o-realtime-preview": {"input": 5.0, "output": 20.0},
    "gpt-4o-mini-realtime-preview": {"input": 0.6, "output": 2.4},
    "gpt-audio": {"input": 2.5, "output": 10.0},
    "gpt-audio-mini": {"input": 0.6, "output": 2.4},
    "gpt-4o-audio-preview": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.6},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 1.1, "output": 4.4},
    "o1-pro": {"input": 150.0, "output": 600.0},
    "o3": {"input": 2.0, "output": 8.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "o3-pro": {"input": 20.0, "output": 80.0},
    "o3-deep-research": {"input": 10.0, "output": 40.0},
    "o4-mini": {"input": 1.1, "output": 4.4},
    "o4-mini-deep-research": {"input": 2.0, "output": 8.0},
    "codex-mini-latest": {"input": 1.5, "output": 6.0},
    "computer-use-preview": {"input": 3.0, "output": 12.0},
}

# Approximate TTS pricing (USD per 1M characters generated), aligned with
# https://platform.openai.com/docs/pricing?latest-pricing=standard#transcription-and-speech
AUDIO_MODEL_PRICING_USD_PER_1M_CHAR: Dict[str, float] = {
    "gpt-4o-mini-tts": 0.15,
    "gpt-4o-tts": 2.5,
    "gpt-audio": 2.5,
    "gpt-audio-mini": 0.6,
    "gpt-4o-audio-preview": 2.5,
    "gpt-4o-mini-audio-preview": 0.6,
}
