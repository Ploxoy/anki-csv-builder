"""Tests for core.llm_clients behaviours (parameter removal, retries)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core import llm_clients as llm


class DummyResponses:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self.sequence[len(self.calls) - 1]
        if isinstance(action, Exception):
            raise action
        return action


def _client_with(sequence):
    return SimpleNamespace(responses=DummyResponses(sequence))


def test_send_request_drops_temperature_on_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with([
        RuntimeError("Unsupported parameter: 'temperature'"),
        SimpleNamespace(ok=True, usage=SimpleNamespace(cached_tokens=512, prompt_tokens=123, completion_tokens=45, total_tokens=168)),
    ])

    response, meta = llm.send_responses_request(
        client=client,
        model="gpt-test",
        instructions="instructions",
        input_text="payload",
        temperature=0.8,
    )

    assert response.ok is True
    assert meta["temperature_removed"] is True
    assert meta["cached_tokens"] == 512
    assert meta["prompt_tokens"] == 123
    assert meta["completion_tokens"] == 45
    assert meta["total_tokens"] == 168
    assert meta["retries"] == 0
    # First call had temperature; second call should not.
    assert len(client.responses.calls) == 2
    assert "temperature" in client.responses.calls[0]
    assert "temperature" not in client.responses.calls[1]


def test_send_request_drops_response_format_when_text_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with([
        RuntimeError("unexpected keyword argument 'text'"),
        SimpleNamespace(ok=True, usage={"cached_tokens": 128, "prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}),
    ])

    response, meta = llm.send_responses_request(
        client=client,
        model="gpt-test",
        instructions="instr",
        input_text="payload",
        response_format={"json_schema": {"title": "Card"}},
    )

    assert response.ok is True
    assert meta["response_format_removed"] is True
    assert "response_format_error" in meta
    assert meta["retries"] == 0
    assert meta["cached_tokens"] == 128
    assert meta["prompt_tokens"] == 50
    assert meta["completion_tokens"] == 20
    assert meta["total_tokens"] == 70
    assert "text" in client.responses.calls[0]
    assert "text" not in client.responses.calls[1]


def test_send_request_reads_cached_tokens_from_input_tokens_details() -> None:
    client = _client_with([
        SimpleNamespace(
            ok=True,
            usage={
                "input_tokens": 400,
                "output_tokens": 20,
                "total_tokens": 420,
                "input_tokens_details": {"cached_tokens": 300},
            },
        ),
    ])

    response, meta = llm.send_responses_request(
        client=client,
        model="gpt-test",
        instructions="instr",
        input_text="payload",
    )

    assert response.ok is True
    assert meta["cached_tokens"] == 300
    assert meta["prompt_tokens"] == 400
    assert meta["completion_tokens"] == 20
    assert meta["total_tokens"] == 420


def test_send_request_retries_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with([
        RuntimeError("rate limit 429"),
        RuntimeError("rate limit 429"),
        SimpleNamespace(ok=True),
    ])

    monkeypatch.setattr(llm, "_sleep_backoff", lambda attempt, base=0.5: None)

    response, meta = llm.send_responses_request(
        client=client,
        model="gpt-test",
        instructions="instr",
        input_text="payload",
        retries=3,
    )

    assert response.ok is True
    assert meta["retries"] == 2
    assert len(client.responses.calls) == 3


def test_send_request_drops_prompt_cache_params_when_rejected() -> None:
    client = _client_with([
        RuntimeError("Unsupported parameter: 'prompt_cache_key'"),
        RuntimeError("Unsupported parameter: 'prompt_cache_retention'"),
        SimpleNamespace(ok=True, usage={"prompt_tokens_details": {"cached_tokens": 256}, "input_tokens": 800, "output_tokens": 50}),
    ])

    response, meta = llm.send_responses_request(
        client=client,
        model="gpt-test",
        instructions="instr",
        input_text="payload",
        prompt_cache_key="doedutch:card:test",
        prompt_cache_retention="24h",
    )

    assert response.ok is True
    assert meta["prompt_cache_key_removed"] is True
    assert meta["prompt_cache_retention_removed"] is True
    assert meta["cached_tokens"] == 256
    assert len(client.responses.calls) == 3
    assert "prompt_cache_key" in client.responses.calls[0]
    assert "prompt_cache_key" not in client.responses.calls[1]
    assert "prompt_cache_retention" in client.responses.calls[1]
    assert "prompt_cache_retention" not in client.responses.calls[2]
