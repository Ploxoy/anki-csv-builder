"""
core/llm_clients.py

Small wrapper helpers around the OpenAI Responses API.
Responsibilities:
- create a lightweight client wrapper
- send a single request with limited retries + exponential backoff
- handle common SDK differences and the "Unsupported parameter: 'temperature'" error
- provide helpers to extract textual output and parsed structured output from the response

This module is Streamlit-agnostic and contains no UI code.
"""
from __future__ import annotations

import time
import json
from typing import Any, Dict, Optional, Tuple

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - openai may not be installed in test env
    OpenAI = None  # type: ignore


def create_client(api_key: Optional[str]) -> Any:
    """Create and return an OpenAI client instance or None if SDK unavailable.

    Keep this thin so callers can still mock or create client differently in tests.
    """
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _sleep_backoff(attempt: int, base: float = 0.5) -> None:
    """Simple exponential backoff sleep (in seconds)."""
    time.sleep(base * (2 ** attempt))


def send_responses_request(
    client: Any,
    model: str,
    instructions: str,
    input_text: str,
    response_format: Optional[Dict] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    retries: int = 2,
    backoff_base: float = 0.5,
) -> Tuple[Any, Dict[str, Any]]:
    """Call Responses API with retries.

    Returns tuple (response, metadata). Metadata содержит флаги, был ли
    удалён response_format/temperature и сколько попыток потребовалось.
    """

    if client is None:
        raise RuntimeError("OpenAI client is not available. Ensure openai SDK is installed.")

    attempt = 0
    last_exc: Optional[Exception] = None
    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    metadata: Dict[str, Any] = {
        "response_format_removed": False,
        "temperature_removed": False,
        "retries": 0,
    }

    while attempt <= retries:
        try:
            resp = client.responses.create(**kwargs)
            metadata["retries"] = attempt
            return resp, metadata
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()

            # If server/SDK complains about unsupported parameter or unexpected kwarg,
            # remove the offending parameter(s) and retry immediately.
            # This handles messages like:
            # - "Unsupported parameter: 'temperature'"
            # - "Responses.create() got an unexpected keyword argument 'response_format'"
            handled = False
            if "temperature" in msg and "unsupported parameter" in msg or "temperature" in msg and "unexpected keyword" in msg:
                if "temperature" in kwargs:
                    kwargs.pop("temperature", None)
                    metadata["temperature_removed"] = True
                    handled = True
            if "response_format" in msg or "unexpected keyword argument 'response_format'" in msg:
                if "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    metadata["response_format_removed"] = True
                    handled = True
            if handled:
                try:
                    resp = client.responses.create(**kwargs)
                    metadata["retries"] = attempt
                    return resp, metadata
                except Exception as exc2:
                    last_exc = exc2
                    # fall through to retry/backoff if transient

            # Transient error handling (naive): retry on common transient keywords or 429/5xx
            transient = False
            try:
                if any(t in msg for t in ("429", "rate", "too many", "timeout", "502", "503", "500")):
                    transient = True
            except Exception:
                transient = False

            if not transient or attempt == retries:
                raise last_exc

            _sleep_backoff(attempt, base=backoff_base)
            attempt += 1
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected error in send_responses_request")
# --- response extraction helpers ---

def get_response_text(resp: Any) -> str:
    """Extract a readable textual body from various SDK response shapes.

    The Responses API SDK may provide: resp.output_text or nested resp.output[*].content[*].text.value etc.
    This helper attempts a best-effort extraction and falls back to str(resp).
    """
    if resp is None:
        return ""
    # direct attribute
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Newer structured responses: resp.output -> list of objects with content
    try:
        parts: list[str] = []
        for out in getattr(resp, "output", []) or []:
            for item in getattr(out, "content", []) or []:
                # sdk objects may have .text.value
                txt_obj = getattr(item, "text", None)
                if txt_obj and hasattr(txt_obj, "value"):
                    parts.append(txt_obj.value)
                elif isinstance(item, dict):
                    # dict-like fallback
                    if "text" in item and isinstance(item["text"], dict) and "value" in item["text"]:
                        parts.append(item["text"]["value"])
                    elif "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(item.text)
        if parts:
            return "".join(parts)
    except Exception:
        pass
    # Fallback: try stringifying resp
    try:
        return json.dumps(resp, default=str, ensure_ascii=False)
    except Exception:
        return str(resp)


def get_response_parsed(resp: Any) -> Dict:
    """Try to extract structured parsed output (if model returned structured JSON).

    Preferred sources (in order):
    - resp.output_parsed (SDK convenience)
    - resp.output[*].content[*].parsed
    - None -> {}
    """
    if resp is None:
        return {}
    try:
        p = getattr(resp, "output_parsed", None)
        if isinstance(p, dict) and p:
            return p
    except Exception:
        pass
    try:
        for out in (getattr(resp, "output", None) or []):
            for item in (getattr(out, "content", None) or []):
                if hasattr(item, "parsed"):
                    p = getattr(item, "parsed")
                    if isinstance(p, dict) and p:
                        return p
                if isinstance(item, dict) and isinstance(item.get("parsed"), dict) and item.get("parsed"):
                    return item.get("parsed")
    except Exception:
        pass
    return {}
