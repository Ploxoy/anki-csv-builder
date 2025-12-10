from __future__ import annotations

from app import response_cache


def test_response_format_cache_roundtrip(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "rf"
    cache_path = cache_dir / "response_format.json"

    # Monkeypatch module-level paths
    monkeypatch.setattr(response_cache, "CACHE_DIR", cache_dir, raising=False)
    monkeypatch.setattr(response_cache, "CACHE_PATH", cache_path, raising=False)

    cache: dict[str, response_cache.ProbeRecord] = {}
    cache = response_cache.update_probe_cache(cache, model="gpt-5", supported=True)
    cache = response_cache.update_probe_cache(
        cache,
        model="gpt-4o-mini",
        supported=False,
        reason="response_format_removed",
    )
    response_cache.save_response_format_cache(cache)

    loaded = response_cache.load_response_format_cache()
    assert "gpt-5" in loaded and loaded["gpt-5"].supported is True
    assert loaded["gpt-4o-mini"].supported is False
    assert loaded["gpt-4o-mini"].reason == "response_format_removed"
