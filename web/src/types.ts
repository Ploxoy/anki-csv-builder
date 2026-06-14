export type GenerateItem = {
  id: string;
  woord: string;
  def_nl?: string;
  translation?: string;
};

export type GenerateRequest = {
  run_id?: string;
  prompt_version: string;
  provider: string;
  model: string;
  cefr: string;
  profile: string;
  l1: string;
  temperature?: number | null;
  max_output_tokens?: number | null;
  flags?: {
    force_schema?: boolean;
    allow_repair?: boolean;
    reuse_text_cache?: boolean;
  };
  items: GenerateItem[];
};

export type GenerateItemResultStatus = "ok" | "repaired" | "failed" | "flagged";

export type Card = {
  L2_word: string;
  L2_cloze: string;
  L1_sentence: string;
  L2_collocations: string;
  L2_definition: string;
  L1_gloss: string;
  L1_hint?: string;
  AudioSentence?: string;
  AudioWord?: string;
};

export type UsageEvent = {
  provider: string;
  model: string;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cached_tokens?: number | null;
  audio_chars?: number | null;
  audio_tokens?: number | null;
  seconds?: number | null;
  elapsed_ms?: number | null;
};

export type GenerateItemResult = {
  id: string;
  status: GenerateItemResultStatus;
  card?: Card | null;
  error?: string | null;
  usage?: UsageEvent | null;
};

export type GenerateResponse = {
  run_id: string;
  prompt_version: string;
  provider: string;
  model: string;
  items: GenerateItemResult[];
  run_report: Record<string, unknown>;
  timing: {
    elapsed_ms: number;
    text_cache_hits?: number;
    text_assets_stored?: number;
    text_cache_errors?: number;
  };
};

export type GenerateJobCreateResponse = {
  job_id: string;
  run_id: string;
  status: "queued" | "running" | "done" | "failed";
};

export type GenerateJobStatusResponse = {
  job_id: string;
  run_id: string;
  status: "queued" | "running" | "done" | "failed";
  processed_items: number;
  total_items: number;
  error?: string | null;
  result?: GenerateResponse | null;
  updated_at?: string | null;
};

export type ExportDeckRequest = {
  run_id?: string;
  l1: string;
  cefr: string;
  profile: string;
  model: string;
  deck_name: string;
  guid_policy: "stable" | "unique";
  include_basic_reversed: boolean;
  include_basic_typein: boolean;
  use_persisted_media?: boolean;
  media_map?: Record<string, string>;
  cards: Card[];
};

export type ExportFileResponse = {
  file_name: string;
  mime_type: string;
  content_b64: string;
  card_count: number;
};

export type AudioAssetCheckRequest = {
  filenames: string[];
};

export type AudioAssetCheckResponse = {
  found: string[];
  missing: string[];
  error?: string | null;
};

export type TTSItemType = "word" | "sentence";

export type TTSItem = {
  card_id: string;
  type: TTSItemType;
  text: string;
};

export type TTSRequest = {
  run_id?: string;
  provider: string;
  model?: string;
  voice?: string;
  items: TTSItem[];
};

export type TTSOption = {
  id: string;
  label: string;
};

export type TTSProviderOptions = {
  models: string[];
  voices: TTSOption[];
  default_model?: string | null;
  default_voice?: string | null;
};

export type TTSOptionsResponse = {
  text_models: string[];
  providers: string[];
  by_provider: Record<string, TTSProviderOptions>;
};

export type TTSVoiceCheckRequest = {
  provider: string;
  voice_id: string;
};

export type TTSVoiceCheckResponse = {
  provider: string;
  id: string;
  label: string;
  valid: boolean;
  source?: string;
};

export type TTSSharedVoiceAddRequest = {
  public_user_id?: string | null;
  voice_id: string;
  new_name?: string | null;
  bookmarked?: boolean;
};

export type TTSSharedVoiceAddResponse = {
  provider: string;
  id: string;
  label: string;
  source?: string;
};

export type TTSPreviewRequest = {
  provider: string;
  model?: string;
  voice?: string;
  text: string;
};

export type TTSPreviewResponse = {
  provider: string;
  model: string;
  voice: string;
  text: string;
  filename: string;
  audio_b64: string;
  summary: {
    ok: number;
    failed: number;
    cached: number;
    errors?: string[];
    usage: Record<string, unknown>;
    cost: Record<string, unknown>;
  };
  timing?: Record<string, unknown>;
};

export type TTSAudio = {
  card_id: string;
  type: TTSItemType;
  status: "ok" | "failed" | "cached";
  filename?: string | null;
  audio_b64?: string | null;
  error?: string | null;
  usage?: UsageEvent | null;
};

export type TTSStorageInfo = {
  persisted: boolean;
  stored_clips: number;
  error?: string | null;
};

export type TTSResponse = {
  run_id: string;
  provider: string;
  model: string;
  audios: TTSAudio[];
  summary: {
    ok: number;
    failed: number;
    cached: number;
    errors?: string[];
    usage: Record<string, unknown>;
    cost: Record<string, unknown>;
  };
  storage?: TTSStorageInfo | null;
  timing?: Record<string, unknown>;
};

export type UserSettings = {
  prompt_version: string;
  provider: string;
  model: string;
  cefr: string;
  profile: string;
  l1: string;
  temperature?: number | null;
  max_output_tokens?: number | null;
  audio_provider: string;
  audio_model?: string | null;
  audio_voice?: string | null;
  force_generate_flagged: boolean;
  generate_audio: boolean;
  include_audio_word: boolean;
  include_audio_sentence: boolean;
  reuse_text_cache: boolean;
  include_basic_reversed: boolean;
  include_basic_typein: boolean;
  default_deck_name: string;
};

export type UserSettingsResponse = {
  user_id: string;
  settings: UserSettings;
  updated_at?: string | null;
};

export type UserSettingsUpsertRequest = {
  settings: UserSettings;
};

export type UsageEventRecord = {
  created_at: string;
  run_id?: string | null;
  kind?: string | null;
  provider?: string | null;
  model?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cached_tokens?: number | null;
  audio_chars?: number | null;
  audio_tokens?: number | null;
  seconds?: number | null;
  raw_cost_usd?: number | null;
  raw_cost_eur?: number | null;
  charged_cost_eur?: number | null;
  markup_tier?: string | null;
  markup_multiplier?: number | null;
  request_id?: string | null;
};

export type UsageSummary = {
  events: number;
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  audio_chars: number;
  raw_cost_usd?: number | null;
};

export type UsageListResponse = {
  user_id: string;
  items: UsageEventRecord[];
  summary: UsageSummary;
};

// Admin / beta
export type UserRecord = {
  id: string;
  label?: string | null;
  status: string;
  created_at: string;
  last_used_at?: string | null;
};

export type UserListResponse = {
  items: UserRecord[];
};

export type InviteCreateResponse = {
  user_id: string;
  token: string;
};

export type AdminUsageResponse = UsageListResponse;
