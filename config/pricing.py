"""Centralized pricing tables for text and audio models (USD)."""
from __future__ import annotations

from typing import Dict

# Approximate Responses API pricing (USD per 1K tokens), aligned with
# https://platform.openai.com/docs/pricing?latest-pricing=standard#text-tokens
# Update these values when OpenAI adjusts rates.
MODEL_PRICING_USD_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-5": {"input": 0.03, "output": 0.09},
    "gpt-5-mini": {"input": 0.01, "output": 0.03},
    "gpt-5-nano": {"input": 0.005, "output": 0.015},
    "gpt-4.1": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "o3": {"input": 0.002, "output": 0.006},
    "o3-mini": {"input": 0.0008, "output": 0.0024},
}

# Approximate TTS pricing (USD per 1K characters generated), aligned with
# https://platform.openai.com/docs/pricing?latest-pricing=standard#transcription-and-speech
AUDIO_MODEL_PRICING_USD_PER_1K_CHAR: Dict[str, float] = {
    "gpt-4o-mini-tts": 0.015,  # $15 per 1M characters
    "gpt-4o-tts": 0.03,        # $30 per 1M characters
}
