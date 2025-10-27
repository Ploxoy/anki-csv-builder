"""Helper utilities for managing Streamlit session state of the audio panel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, MutableMapping, Optional, Sequence


def _ensure_dict(store: MutableMapping[str, Any], key: str) -> Dict[str, Any]:
    value = store.get(key)
    if not isinstance(value, dict):
        value = {}
        store[key] = value
    return value


@dataclass
class AudioPanelState:
    """Typed façade over ``st.session_state`` for the audio panel."""

    store: MutableMapping[str, Any]

    def is_expanded(self) -> bool:
        return bool(self.store.get("audio_panel_expanded", False))

    def expand(self) -> None:
        self.store["audio_panel_expanded"] = True

    def collapse(self) -> None:
        self.store["audio_panel_expanded"] = False

    def preferred_provider(self, provider_keys: Sequence[str], default_provider: Optional[str]) -> str:
        if not provider_keys:
            return ""
        current = self.store.get("audio_provider")
        if isinstance(current, str) and current in provider_keys:
            return current
        if default_provider and default_provider in provider_keys:
            return default_provider
        return provider_keys[0]

    def set_provider(self, provider_key: str) -> None:
        self.store["audio_provider"] = provider_key

    def nonce(self) -> int:
        value = self.store.get("audio_ui_nonce")
        if isinstance(value, int):
            return value
        self.store["audio_ui_nonce"] = 0
        return 0

    def bump_nonce(self) -> int:
        new_value = self.nonce() + 1
        self.store["audio_ui_nonce"] = new_value
        return new_value

    def widget_key(self, base: str, nonce: int) -> str:
        return f"{base}__{nonce}"

    def handle_provider_change(self, new_key: str, current_voice: str) -> bool:
        previous = self.store.get("_audio_provider_last")
        changed = bool(new_key) and new_key != previous
        if changed:
            if isinstance(previous, str) and previous and current_voice:
                voice_map = self.voice_map()
                voice_map[previous] = current_voice
            self.store["_audio_provider_last"] = new_key
        return changed

    def voice_widget_base(self, provider_key: str) -> str:
        return f"audio_voice__{provider_key}" if provider_key else "audio_voice__default"

    def clear_widget(self, widget_key: str) -> None:
        self.store.pop(widget_key, None)

    def audio_cache(self) -> Dict[str, Any]:
        return _ensure_dict(self.store, "audio_cache")

    def audio_summary(self) -> Optional[Dict[str, Any]]:
        summary = self.store.get("audio_summary")
        return summary if isinstance(summary, dict) else None

    def set_audio_summary(self, summary: Dict[str, Any]) -> None:
        self.store["audio_summary"] = summary

    def set_audio_media(self, media_map: Dict[str, Any]) -> None:
        self.store["audio_media"] = media_map

    def results(self) -> list[Dict[str, Any]]:
        results = self.store.get("results")
        return results if isinstance(results, list) else []

    def set_results(self, results: Sequence[Dict[str, Any]]) -> None:
        self.store["results"] = [dict(card) for card in results]

    def current_voice(self) -> str:
        value = self.store.get("audio_voice")
        return value if isinstance(value, str) else ""

    def set_current_voice(self, voice: str) -> None:
        self.store["audio_voice"] = voice

    def clear_current_voice(self, provider_key: str) -> None:
        self.store["audio_voice"] = ""
        if provider_key:
            voice_map = self.voice_map()
            voice_map.pop(provider_key, None)

    def voice_map(self) -> Dict[str, str]:
        return _ensure_dict(self.store, "audio_voice_map")

    def update_voice_map(self, provider_key: str, voice: str) -> None:
        if not provider_key:
            return
        voice_map = self.voice_map()
        if voice:
            voice_map[provider_key] = voice
        else:
            voice_map.pop(provider_key, None)

    def apply_provider_defaults(
        self,
        provider_key: str,
        provider_data: Dict[str, Any],
        voice_ids: Sequence[str],
    ) -> str:
        voice_map = self.voice_map()
        stored_voice = voice_map.get(provider_key)
        default_voice = ""

        configured_default = provider_data.get("voice_default") if isinstance(provider_data, dict) else ""
        if isinstance(configured_default, str) and configured_default:
            default_voice = configured_default
        if stored_voice and stored_voice in voice_ids:
            default_voice = stored_voice
        if default_voice and default_voice not in voice_ids:
            default_voice = ""
        if not default_voice and voice_ids:
            default_voice = voice_ids[0]

        if default_voice:
            self.set_current_voice(default_voice)
            if provider_key:
                voice_map[provider_key] = default_voice
        else:
            self.clear_current_voice(provider_key)

        include_word_default = True
        include_sentence_default = True
        if isinstance(provider_data, dict):
            include_word_default = provider_data.get("include_word_default", True)
            include_sentence_default = provider_data.get("include_sentence_default", True)
            sentence_default = provider_data.get("sentence_default")
            word_default = provider_data.get("word_default")
            if isinstance(sentence_default, str) and sentence_default:
                self.store["audio_sentence_instruction"] = sentence_default
            if isinstance(word_default, str) and word_default:
                self.store["audio_word_instruction"] = word_default

        self.store["audio_include_word"] = bool(include_word_default)
        self.store["audio_include_sentence"] = bool(include_sentence_default)

        return self.current_voice()

    def get_elevenlabs_key(self) -> str:
        value = self.store.get("elevenlabs_api_key")
        return value.strip() if isinstance(value, str) else ""

    def set_elevenlabs_key(self, key: str) -> None:
        self.store["elevenlabs_api_key"] = key

    def clear_elevenlabs_key(self) -> None:
        self.store.pop("elevenlabs_api_key", None)
        self.store.pop("_elevenlabs_api_key_snapshot", None)

    def api_key_snapshot(self) -> str:
        value = self.store.get("_elevenlabs_api_key_snapshot")
        return value if isinstance(value, str) else ""

    def update_api_key_snapshot(self, value: str) -> bool:
        previous = self.api_key_snapshot()
        if value != previous:
            self.store["_elevenlabs_api_key_snapshot"] = value
            return True
        return False

    def seed_api_key_snapshot(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            return
        if not isinstance(self.store.get("_elevenlabs_api_key_snapshot"), str):
            self.store["_elevenlabs_api_key_snapshot"] = value

    def is_replacing_elevenlabs_key(self) -> bool:
        return bool(self.store.get("_elevenlabs_replacing_key", False))

    def begin_elevenlabs_key_replace(self) -> None:
        self.store["_elevenlabs_replacing_key"] = True

    def end_elevenlabs_key_replace(self) -> None:
        self.store["_elevenlabs_replacing_key"] = False
