"""
Draft Pydantic schemas for the public API (generate / tts).

These are not wired yet; intended for Phase 0 to stabilize request/response
contracts while the UI is still Streamlit-based.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ---------- Shared ----------

class UsageEvent(BaseModel):
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    audio_chars: Optional[int] = None
    audio_tokens: Optional[int] = None
    seconds: Optional[float] = None
    raw_cost_usd: Optional[float] = None
    raw_cost_eur: Optional[float] = None
    charged_cost_eur: Optional[float] = None
    markup_tier: Optional[str] = None
    markup_multiplier: Optional[float] = None
    request_id: Optional[str] = None
    elapsed_ms: Optional[int] = None


class ErrorEnvelope(BaseModel):
    error: Dict[str, Any]

class UserSettings(BaseModel):
    """Persisted (non-secret) per-user defaults.

    Intentionally excludes any API keys. Provider keys must live on the server.
    """

    prompt_version: str = "p0"
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    cefr: str = "B1"
    profile: str = "balanced"
    l1: str = "EN"
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    audio_provider: str = "openai"
    audio_model: Optional[str] = None
    audio_voice: Optional[str] = None
    force_generate_flagged: bool = False
    generate_audio: bool = False
    include_audio_word: bool = True
    include_audio_sentence: bool = True
    reuse_text_cache: bool = False
    include_basic_reversed: bool = False
    include_basic_typein: bool = False
    default_deck_name: str = "Dutch"


class UserSettingsUpsertRequest(BaseModel):
    settings: UserSettings


class UserSettingsResponse(BaseModel):
    user_id: str
    settings: UserSettings
    updated_at: Optional[str] = None


class UsageEventRecord(BaseModel):
    created_at: str
    run_id: Optional[str] = None
    kind: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    audio_chars: Optional[int] = None
    audio_tokens: Optional[int] = None
    seconds: Optional[float] = None
    raw_cost_usd: Optional[float] = None
    raw_cost_eur: Optional[float] = None
    charged_cost_eur: Optional[float] = None
    markup_tier: Optional[str] = None
    markup_multiplier: Optional[float] = None
    request_id: Optional[str] = None


class UsageSummary(BaseModel):
    events: int
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    audio_chars: int = 0
    raw_cost_usd: Optional[float] = None


class UsageListResponse(BaseModel):
    user_id: str
    items: List[UsageEventRecord]
    summary: UsageSummary


# ---------- Beta auth (Phase 0.5) ----------

class InviteCreateRequest(BaseModel):
    label: Optional[str] = None


class InviteCreateResponse(BaseModel):
    user_id: str
    token: str


class WhoAmIResponse(BaseModel):
    user_id: str


class UserRecord(BaseModel):
    id: str
    label: Optional[str] = None
    status: str
    created_at: str
    last_used_at: Optional[str] = None


class UserListResponse(BaseModel):
    items: List[UserRecord]


class UserStatusRequest(BaseModel):
    status: Literal["active", "blocked"]


class UserRotateResponse(BaseModel):
    user_id: str
    token: str


# ---------- Generate ----------

class GenerateItem(BaseModel):
    id: str
    woord: str
    def_nl: Optional[str] = ""
    translation: Optional[str] = ""


class GenerateFlags(BaseModel):
    force_schema: bool = True
    allow_repair: bool = True
    reuse_text_cache: bool = False


class GenerateRequest(BaseModel):
    run_id: Optional[str] = None
    prompt_version: str
    provider: str
    model: str
    cefr: str
    profile: str
    l1: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    guid_policy: Literal["stable", "unique"] = "stable"
    flags: GenerateFlags = Field(default_factory=GenerateFlags)
    items: List[GenerateItem]


class Card(BaseModel):
    L2_word: str
    L2_cloze: str
    L1_sentence: str
    L2_collocations: str
    L2_definition: str
    L1_gloss: str


class GenerateItemResult(BaseModel):
    id: str
    status: Literal["ok", "repaired", "failed", "flagged"]
    card: Optional[Card] = None
    error: Optional[str] = None
    usage: Optional[UsageEvent] = None


class GenerateResponse(BaseModel):
    run_id: str
    prompt_version: str
    provider: str
    model: str
    items: List[GenerateItemResult]
    run_report: Dict[str, Any]
    timing: Dict[str, Any]


class GenerateJobCreateResponse(BaseModel):
    job_id: str
    run_id: str
    status: Literal["queued", "running", "done", "failed"]


class GenerateJobStatusResponse(BaseModel):
    job_id: str
    run_id: str
    status: Literal["queued", "running", "done", "failed"]
    processed_items: int = 0
    total_items: int = 0
    error: Optional[str] = None
    result: Optional[GenerateResponse] = None
    updated_at: Optional[str] = None


class GenerateJobWorkerRequest(BaseModel):
    job_id: Optional[str] = None
    max_items: Optional[int] = None


class GenerateJobWorkerResponse(BaseModel):
    processed: bool
    job_id: Optional[str] = None
    status: Optional[Literal["queued", "running", "done", "failed"]] = None
    processed_items: int = 0
    total_items: int = 0
    message: Optional[str] = None


# ---------- Export ----------

class ExportCard(BaseModel):
    L2_word: str
    L2_cloze: str
    L1_sentence: str
    L2_collocations: str
    L2_definition: str
    L1_gloss: str
    L1_hint: str = ""
    AudioSentence: str = ""
    AudioWord: str = ""


class ExportDeckRequest(BaseModel):
    run_id: Optional[str] = None
    l1: str
    cefr: str
    profile: str
    model: str
    deck_name: str = "Dutch"
    guid_policy: Literal["stable", "unique"] = "stable"
    include_basic_reversed: bool = False
    include_basic_typein: bool = False
    use_persisted_media: bool = False
    media_map: Optional[Dict[str, str]] = None
    cards: List[ExportCard]


class ExportFileResponse(BaseModel):
    file_name: str
    mime_type: str
    content_b64: str
    card_count: int


# ---------- Audio assets ----------

class AudioAssetCheckRequest(BaseModel):
    filenames: List[str] = Field(default_factory=list)


class AudioAssetCheckResponse(BaseModel):
    found: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ---------- TTS ----------

class TTSItem(BaseModel):
    card_id: str
    type: Literal["word", "sentence"]
    text: str


class TTSRequest(BaseModel):
    run_id: Optional[str] = None
    provider: str
    model: Optional[str] = None
    voice: Optional[str] = None
    items: List[TTSItem]


class TTSOption(BaseModel):
    id: str
    label: str


class TTSProviderOptions(BaseModel):
    models: List[str] = Field(default_factory=list)
    voices: List[TTSOption] = Field(default_factory=list)
    default_model: Optional[str] = None
    default_voice: Optional[str] = None


class TTSOptionsResponse(BaseModel):
    text_models: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)
    by_provider: Dict[str, TTSProviderOptions] = Field(default_factory=dict)


class TTSVoiceCheckRequest(BaseModel):
    provider: str
    voice_id: str


class TTSVoiceCheckResponse(BaseModel):
    provider: str
    id: str
    label: str
    valid: bool = True
    source: str = "manual"


class TTSAudio(BaseModel):
    card_id: str
    type: Literal["word", "sentence"]
    status: Literal["ok", "failed", "cached"]
    filename: Optional[str] = None
    audio_b64: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[UsageEvent] = None


class TTSSummary(BaseModel):
    ok: int
    failed: int
    cached: int
    errors: List[str] = Field(default_factory=list)
    usage: Dict[str, Any]
    cost: Dict[str, Any]


class TTSStorageInfo(BaseModel):
    persisted: bool = False
    stored_clips: int = 0
    error: Optional[str] = None


class TTSResponse(BaseModel):
    run_id: str
    provider: str
    model: str
    audios: List[TTSAudio]
    summary: TTSSummary
    storage: Optional[TTSStorageInfo] = None
    timing: Dict[str, Any] = Field(default_factory=dict)
