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
  timing: { elapsed_ms: number };
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
