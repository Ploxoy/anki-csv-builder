import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { parseItems } from "./lib/parse";
import { loadJson, saveJson } from "./lib/storage";
import {
  AudioAssetCheckRequest,
  AudioAssetCheckResponse,
  Card,
  ExportDeckRequest,
  ExportFileResponse,
  GenerateRequest,
  GenerateResponse,
  GenerateJobCreateResponse,
  GenerateJobStatusResponse,
  InviteCreateResponse,
  TTSOptionsResponse,
  TTSPreviewRequest,
  TTSPreviewResponse,
  TTSRequest,
  TTSResponse,
  TTSSharedVoiceAddRequest,
  TTSSharedVoiceAddResponse,
  TTSVoiceCheckRequest,
  TTSVoiceCheckResponse,
  UsageListResponse,
  UserListResponse,
  UserSettingsResponse,
  UserSettingsUpsertRequest,
} from "./types";
import {
  apiErrorText,
  appendUniqueError,
  AudioRunSummary,
  decodeBase64ToArrayBuffer,
  DEFAULT_SETTINGS,
  downloadBlobFile,
  ExportFormat,
  formatElapsedMs,
  formatPercent,
  GenerateProgressMeta,
  generateRunId,
  normalizeImportedText,
  normalizedDeckName,
  parseAttachmentFilename,
  parseTemperatureValue,
  SETTINGS_KEY,
  settingsFingerprint,
  Settings,
  temperatureToDisplayString,
} from "./lib/uiState";
import {
  clearAdminNotices,
  clearGenerateNotices,
  clearSettingsNotices,
  EMPTY_NOTICES,
  ScopedNotices,
  withAdminNotice,
  withGenerateNotice,
  withSettingsNotice,
} from "./lib/messages";
import { AppShell } from "./ui/AppShell";
import { GenerateTab } from "./features/generate/GenerateTab";
import { SettingsTab } from "./features/settings/SettingsTab";
import { AdminTab } from "./features/admin/AdminTab";
import { TabId } from "./lib/uiState";

function buildTtsItems(
  generated: GenerateResponse,
  opts: { includeWord: boolean; includeSentence: boolean }
): TTSRequest["items"] {
  const items: TTSRequest["items"] = [];
  for (const row of generated.items) {
    if (!row.card) continue;
    if (row.status !== "ok" && row.status !== "repaired") continue;
    if (opts.includeWord) {
      const word = (row.card.L2_word || "").trim();
      if (word && !(row.card.AudioWord || "").trim()) items.push({ card_id: row.id, type: "word", text: word });
    }
    if (opts.includeSentence) {
      const sentence = (row.card.L2_cloze || "").trim();
      if (sentence && !(row.card.AudioSentence || "").trim()) items.push({ card_id: row.id, type: "sentence", text: sentence });
    }
  }
  return items;
}

function countAttachedAudioClips(
  generated: GenerateResponse,
  opts: { includeWord: boolean; includeSentence: boolean }
): number {
  let count = 0;
  for (const row of generated.items) {
    if (!row.card) continue;
    if (row.status !== "ok" && row.status !== "repaired") continue;
    if (opts.includeWord && (row.card.AudioWord || "").trim()) count += 1;
    if (opts.includeSentence && (row.card.AudioSentence || "").trim()) count += 1;
  }
  return count;
}

type AttachedAudioField = "AudioWord" | "AudioSentence";

type AttachedAudioEntry = {
  field: AttachedAudioField;
  filename: string;
};

function soundFilenameFromField(value?: string | null): string {
  const text = (value || "").trim();
  if (!text) return "";
  const match = text.match(/^\[sound:([^\]]+)\]$/i);
  return (match?.[1] || text).trim();
}

function collectAttachedAudioEntries(
  generated: GenerateResponse,
  opts: { includeWord: boolean; includeSentence: boolean }
): AttachedAudioEntry[] {
  const entries: AttachedAudioEntry[] = [];
  for (const row of generated.items) {
    if (!row.card) continue;
    if (row.status !== "ok" && row.status !== "repaired") continue;
    if (opts.includeWord) {
      const filename = soundFilenameFromField(row.card.AudioWord);
      if (filename) entries.push({ field: "AudioWord", filename });
    }
    if (opts.includeSentence) {
      const filename = soundFilenameFromField(row.card.AudioSentence);
      if (filename) entries.push({ field: "AudioSentence", filename });
    }
  }
  return entries;
}

function stripMissingAttachedAudio(generated: GenerateResponse, missingFilenames: Set<string>): GenerateResponse {
  if (missingFilenames.size === 0) return generated;
  return {
    ...generated,
    items: generated.items.map((item) => {
      if (!item.card) return item;
      const card = { ...item.card };
      if (missingFilenames.has(soundFilenameFromField(card.AudioWord))) card.AudioWord = "";
      if (missingFilenames.has(soundFilenameFromField(card.AudioSentence))) card.AudioSentence = "";
      return { ...item, card };
    }),
  };
}

const EXPORT_REQUEST_SOFT_LIMIT_BYTES = 4_200_000;

function estimateExportRequestSizeBytes(payload: ExportDeckRequest): number {
  return new TextEncoder().encode(JSON.stringify(payload)).length;
}

function formatMb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(2);
}


function mergeTtsIntoResponse(generated: GenerateResponse, tts: TTSResponse): GenerateResponse {
  const byCard = new Map<string, { word?: string; sentence?: string }>();
  for (const audio of tts.audios) {
    if (audio.status === "failed") continue;
    if (!audio.filename) continue;
    const current = byCard.get(audio.card_id) || {};
    if (audio.type === "word") current.word = audio.filename;
    if (audio.type === "sentence") current.sentence = audio.filename;
    byCard.set(audio.card_id, current);
  }
  return {
    ...generated,
    items: generated.items.map((item) => {
      if (!item.card) return item;
      const entry = byCard.get(item.id);
      if (!entry) return item;
      const card = { ...item.card };
      if (entry.word) card.AudioWord = `[sound:${entry.word}]`;
      if (entry.sentence) card.AudioSentence = `[sound:${entry.sentence}]`;
      return { ...item, card };
    }),
  };
}

function mapServerSettings(base: Settings, payload: UserSettingsResponse): Settings {
  return {
    ...base,
    promptVersion: payload.settings.prompt_version,
    provider: payload.settings.provider,
    model: payload.settings.model,
    cefr: payload.settings.cefr,
    profile: payload.settings.profile,
    l1: payload.settings.l1,
    audioProvider: payload.settings.audio_provider,
    audioModel: payload.settings.audio_model || "",
    audioVoice: payload.settings.audio_voice || "",
    temperature: temperatureToDisplayString(payload.settings.temperature),
    includeFlagged: payload.settings.force_generate_flagged,
    generateAudio: payload.settings.generate_audio,
    includeAudioWord: payload.settings.include_audio_word,
    includeAudioSentence: payload.settings.include_audio_sentence,
    reuseTextCards: payload.settings.reuse_text_cache,
    includeBasicReversed: payload.settings.include_basic_reversed,
    includeBasicTypein: payload.settings.include_basic_typein,
    defaultDeck: payload.settings.default_deck_name,
  };
}

const STARTER_INPUT = "aanraken\tiets voelen\tto touch\nbegrijpen\tsnappen wat iets betekent\tto understand";
const TTS_OPTIONS_AUTO_REFRESH_MS = 3 * 60 * 1000;
const OPENAI_TTS_BATCH_TIMEOUT_MS = 30_000;
const ELEVENLABS_TTS_BATCH_TIMEOUT_MS = 45_000;
const CUSTOM_AUDIO_VOICE_LABELS_KEY = "doedutch.customAudioVoiceLabels.v1";

export default function App() {
  const loadedSettings = useMemo<Settings>(() => ({ ...DEFAULT_SETTINGS, ...loadJson(SETTINGS_KEY, DEFAULT_SETTINGS) }), []);

  const [settings, setSettings] = useState<Settings>(loadedSettings);
  const [initialSettingsSnapshot, setInitialSettingsSnapshot] = useState<Settings>(loadedSettings);
  const [inputText, setInputText] = useState<string>(STARTER_INPUT);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [response, setResponse] = useState<GenerateResponse | null>(null);
  const [usage, setUsage] = useState<UsageListResponse | null>(null);
  const [users, setUsers] = useState<UserListResponse | null>(null);
  const [newInvite, setNewInvite] = useState<InviteCreateResponse | null>(null);
  const [adminUsage, setAdminUsage] = useState<UsageListResponse | null>(null);
  const [adminBusy, setAdminBusy] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("generate");
  const [generateProgress, setGenerateProgress] = useState(0);
  const [generateProgressLabel, setGenerateProgressLabel] = useState("");
  const [generateProgressMeta, setGenerateProgressMeta] = useState<GenerateProgressMeta | null>(null);
  const [exportBusy, setExportBusy] = useState<ExportFormat | null>(null);
  const [audioMediaMap, setAudioMediaMap] = useState<Record<string, string>>({});
  const [audioRunSummary, setAudioRunSummary] = useState<AudioRunSummary | null>(null);
  const [ttsOptions, setTtsOptions] = useState<TTSOptionsResponse | null>(null);
  const [ttsOptionsBusy, setTtsOptionsBusy] = useState(false);
  const [ttsOptionsLoadedAt, setTtsOptionsLoadedAt] = useState<number | null>(null);
  const [customAudioVoiceLabels, setCustomAudioVoiceLabels] = useState<Record<string, string>>(() =>
    loadJson(CUSTOM_AUDIO_VOICE_LABELS_KEY, {})
  );
  const [notices, setNotices] = useState<ScopedNotices>({
    generate: { ...EMPTY_NOTICES.generate },
    settings: { ...EMPTY_NOTICES.settings },
    admin: { ...EMPTY_NOTICES.admin },
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileImportModeRef = useRef<"replace" | "append">("replace");

  const deckPreview = useMemo(() => {
    const base = settings.defaultDeck?.trim() || "Deck";
    return `${base}.csv / ${base}.apkg`;
  }, [settings.defaultDeck]);

  const parsed = useMemo(() => parseItems(inputText), [inputText]);
  const adminEnabled = settings.xApiKey.trim().length > 0;
  const canGenerateByInput = parsed.items.length > 0;
  const temperatureState = useMemo(() => parseTemperatureValue(settings.temperature), [settings.temperature]);
  const canGenerate = canGenerateByInput && !temperatureState.error;
  const isDirty = useMemo(
    () => settingsFingerprint(settings) !== settingsFingerprint(initialSettingsSnapshot),
    [settings, initialSettingsSnapshot]
  );
  const audioClipCount = useMemo(() => Object.keys(audioMediaMap).length, [audioMediaMap]);
  const audioProviderKey = (settings.audioProvider || "openai").trim().toLowerCase();

  const availableAudioProviders = useMemo(() => {
    const fromApi = ttsOptions?.providers || [];
    const fromState = settings.audioProvider ? [audioProviderKey] : [];
    const merged = Array.from(new Set([...fromApi, ...fromState].filter(Boolean)));
    return merged.length > 0 ? merged : ["openai"];
  }, [ttsOptions, settings.audioProvider, audioProviderKey]);

  const textModelOptions = useMemo(() => {
    const fromApi = ttsOptions?.text_models || [];
    const fallback = settings.model ? [settings.model] : [];
    const merged = Array.from(new Set([...fromApi, ...fallback].filter(Boolean)));
    return merged.length > 0 ? merged : ["gpt-4.1-mini"];
  }, [ttsOptions, settings.model]);

  const activeAudioProviderOptions = ttsOptions?.by_provider?.[audioProviderKey];
  const audioModelOptions = activeAudioProviderOptions?.models || [];
  const audioVoiceOptions = activeAudioProviderOptions?.voices || [];

  const availableAudioModelOptions = useMemo(() => {
    const fromCurrent = settings.audioModel ? [settings.audioModel] : [];
    return Array.from(new Set([...audioModelOptions, ...fromCurrent].filter(Boolean)));
  }, [audioModelOptions, settings.audioModel]);

  const availableAudioVoiceOptions = useMemo(() => {
    const fromApi = audioVoiceOptions.map((voice) => voice.id);
    const fromCurrent = settings.audioVoice ? [settings.audioVoice] : [];
    return Array.from(new Set([...fromApi, ...fromCurrent].filter(Boolean)));
  }, [audioVoiceOptions, settings.audioVoice]);

  const audioVoiceLabels = useMemo(() => {
    const map: Record<string, string> = {};
    for (const voice of audioVoiceOptions) {
      if (voice.id) map[voice.id] = voice.label || voice.id;
    }
    for (const [voiceId, label] of Object.entries(customAudioVoiceLabels)) {
      if (voiceId) map[voiceId] = label || voiceId;
    }
    if (settings.audioVoice && !map[settings.audioVoice]) {
      map[settings.audioVoice] = settings.audioVoice;
    }
    return map;
  }, [audioVoiceOptions, customAudioVoiceLabels, settings.audioVoice]);

  const exportCards = useMemo(() => {
    const items = response?.items || [];
    return items
      .filter((it) => !!it.card && (it.status === "ok" || it.status === "repaired"))
      .map((it) => {
        const card = it.card as Card;
        return {
          L2_word: card.L2_word || "",
          L2_cloze: card.L2_cloze || "",
          L1_sentence: card.L1_sentence || "",
          L2_collocations: card.L2_collocations || "",
          L2_definition: card.L2_definition || "",
          L1_gloss: card.L1_gloss || "",
          L1_hint: card.L1_hint || "",
          AudioSentence: card.AudioSentence || "",
          AudioWord: card.AudioWord || "",
        } as Card;
      });
  }, [response]);

  const hasAudioFailures = !!audioRunSummary && audioRunSummary.requested && audioRunSummary.failed > 0;
  const canUsePersistedAudioForApkg =
    !!response?.run_id &&
    !!audioRunSummary?.requested &&
    !!audioRunSummary?.persisted &&
    (audioRunSummary?.storedClips || 0) > 0;

  function setGenerateNotice(section: "input" | "run" | "audio" | "export", level: "info" | "success" | "warning" | "error", message: string, details?: string) {
    setNotices((prev) => withGenerateNotice(prev, section, { level, message, details }));
  }

  function clearGenerate() {
    setNotices((prev) => clearGenerateNotices(prev));
  }

  function setSettingsNotice(section: "toolbar" | "access" | "generation" | "audio" | "export", level: "info" | "success" | "warning" | "error", message: string, details?: string) {
    setNotices((prev) => withSettingsNotice(prev, section, { level, message, details }));
  }

  function clearSettings() {
    setNotices((prev) => clearSettingsNotices(prev));
  }

  function setAdminNotice(section: "toolbar" | "invite" | "users" | "usage", level: "info" | "success" | "warning" | "error", message: string, details?: string) {
    setNotices((prev) => withAdminNotice(prev, section, { level, message, details }));
  }

  function clearAdmin() {
    setNotices((prev) => clearAdminNotices(prev));
  }

  function patchSettings(patch: Partial<Settings>) {
    setSettings((current) => ({ ...current, ...patch }));
  }

  function onAudioProviderChange(providerId: string) {
    const nextOptions = ttsOptions?.by_provider?.[providerId];
    setSettings((current) => ({
      ...current,
      audioProvider: providerId,
      audioModel: nextOptions?.default_model || current.audioModel,
      audioVoice: nextOptions?.default_voice || current.audioVoice,
    }));
  }

  useEffect(() => {
    saveJson(SETTINGS_KEY, settings);
  }, [settings]);

  useEffect(() => setWarnings(parsed.warnings), [parsed.warnings]);

  useEffect(() => {
    setSettings((current) => {
      const parsedTemp = parseTemperatureValue(current.temperature);
      if (!parsedTemp.usedLegacyScale || parsedTemp.percent == null) return current;
      const normalized = formatPercent(parsedTemp.percent);
      if (normalized === current.temperature.trim()) return current;
      return { ...current, temperature: normalized };
    });
  }, []);

  useEffect(() => {
    if (activeTab === "admin" && !adminEnabled) {
      setActiveTab("settings");
    }
  }, [activeTab, adminEnabled]);

  useEffect(() => {
    if (!settings.userToken.trim()) return;
    void onLoadTtsOptions(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.userToken, settings.apiBase]);

  useEffect(() => {
    if (!settings.userToken.trim()) return;
    if (ttsOptionsBusy) return;
    const ageMs = ttsOptionsLoadedAt == null ? Number.POSITIVE_INFINITY : Date.now() - ttsOptionsLoadedAt;
    const delayMs = Math.max(5000, TTS_OPTIONS_AUTO_REFRESH_MS - ageMs);
    const timer = window.setTimeout(() => {
      void onLoadTtsOptions(true);
    }, delayMs);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.userToken, settings.apiBase, ttsOptionsBusy, ttsOptionsLoadedAt]);

  function apiHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (settings.userToken.trim()) headers.Authorization = `Bearer ${settings.userToken.trim()}`;
    if (settings.xApiKey.trim()) headers["X-API-Key"] = settings.xApiKey.trim();
    return headers;
  }

  async function verifyAttachedAudioAssets(
    generated: GenerateResponse,
    opts: { includeWord: boolean; includeSentence: boolean },
    headers: Record<string, string>
  ): Promise<{ payload: GenerateResponse; attached: number; missing: number; diagnostics: string[] }> {
    const entries = collectAttachedAudioEntries(generated, opts);
    if (entries.length === 0) {
      return { payload: generated, attached: 0, missing: 0, diagnostics: [] };
    }

    const filenames = Array.from(new Set(entries.map((entry) => entry.filename).filter(Boolean)));
    if (filenames.length === 0) {
      return { payload: generated, attached: entries.length, missing: 0, diagnostics: [] };
    }

    const body: AudioAssetCheckRequest = { filenames };
    try {
      const res = await fetch(`${settings.apiBase || ""}/api/audio/assets/check`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      const data = (await res.json().catch(() => null)) as AudioAssetCheckResponse | any;
      if (!res.ok) {
        return {
          payload: generated,
          attached: entries.length,
          missing: 0,
          diagnostics: [`Attached audio verification skipped: ${apiErrorText(data, res.status)}.`],
        };
      }
      if (data?.error) {
        return {
          payload: generated,
          attached: entries.length,
          missing: 0,
          diagnostics: [`Attached audio verification skipped: ${data.error}.`],
        };
      }

      const foundSet = new Set((data?.found || []).map((name: string) => String(name || "").trim()).filter(Boolean));
      const missingSet = new Set(
        filenames.filter((filename) => !foundSet.has(filename))
      );
      const missingAttachedCount = entries.filter((entry) => missingSet.has(entry.filename)).length;
      const diagnostics: string[] = [];
      if (missingAttachedCount > 0) {
        diagnostics.push(
          `Attached audio verification: ${entries.length - missingAttachedCount}/${entries.length} linked clip(s) exist, ` +
            `${missingAttachedCount} missing clip(s) will be synthesized.`
        );
      } else {
        diagnostics.push(`Attached audio verification: ${entries.length}/${entries.length} linked clip(s) exist.`);
      }
      if (data?.error) {
        diagnostics.push(`Attached audio verification warning: ${data.error}.`);
      }
      return {
        payload: stripMissingAttachedAudio(generated, missingSet),
        attached: entries.length,
        missing: missingAttachedCount,
        diagnostics,
      };
    } catch (err: any) {
      return {
        payload: generated,
        attached: entries.length,
        missing: 0,
        diagnostics: [`Attached audio verification skipped: ${err?.message || String(err)}.`],
      };
    }
  }

  async function onLoadSettings() {
    clearSettings();
    try {
      const url = `${settings.apiBase || ""}/api/settings`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        throw new Error(apiErrorText(data, res.status));
      }
      const payload = data as UserSettingsResponse;
      const nextSettings = mapServerSettings(settings, payload);
      setSettings(nextSettings);
      setInitialSettingsSnapshot(nextSettings);
      setSettingsNotice("toolbar", "success", `Settings loaded for ${payload.user_id}.`);
    } catch (e: any) {
      setSettingsNotice("toolbar", "error", e?.message || String(e));
    }
  }

  async function onSaveSettings() {
    clearSettings();
    const tempParsed = parseTemperatureValue(settings.temperature);
    if (tempParsed.error) {
      setSettingsNotice("generation", "error", tempParsed.error);
      return;
    }

    try {
      const url = `${settings.apiBase || ""}/api/settings`;
      const headers = apiHeaders();
      const body: UserSettingsUpsertRequest = {
        settings: {
          prompt_version: settings.promptVersion,
          provider: settings.provider,
          model: settings.model,
          cefr: settings.cefr,
          profile: settings.profile,
          l1: settings.l1,
          audio_provider: settings.audioProvider,
          audio_model: settings.audioModel || null,
          audio_voice: settings.audioVoice || null,
          temperature: tempParsed.ratio,
          max_output_tokens: null,
          force_generate_flagged: settings.includeFlagged,
          generate_audio: settings.generateAudio,
          include_audio_word: settings.includeAudioWord,
          include_audio_sentence: settings.includeAudioSentence,
          reuse_text_cache: settings.reuseTextCards,
          include_basic_reversed: settings.includeBasicReversed,
          include_basic_typein: settings.includeBasicTypein,
          default_deck_name: settings.defaultDeck,
        },
      };
      const res = await fetch(url, { method: "PUT", headers, body: JSON.stringify(body) });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        throw new Error(apiErrorText(data, res.status));
      }
      const payload = data as UserSettingsResponse;
      const nextSettings = mapServerSettings(settings, payload);
      setSettings(nextSettings);
      setInitialSettingsSnapshot(nextSettings);
      setSettingsNotice("toolbar", "success", `Settings saved for ${payload.user_id}.`);
    } catch (e: any) {
      setSettingsNotice("toolbar", "error", e?.message || String(e));
    }
  }

  function onRevertSettings() {
    setSettings(initialSettingsSnapshot);
    clearSettings();
    setSettingsNotice("toolbar", "info", "Changes reverted to last saved snapshot.");
  }

  async function onLoadUsage() {
    clearSettings();
    try {
      const url = `${settings.apiBase || ""}/api/usage?limit=50`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        throw new Error(apiErrorText(data, res.status));
      }
      const payload = data as UsageListResponse;
      setUsage(payload);
      if ((payload.summary?.events || 0) > 0) {
        setSettingsNotice("toolbar", "success", `Usage loaded (${payload.summary.events} events).`);
      } else {
        setSettingsNotice("toolbar", "info", "Usage loaded: no events yet for this token.");
      }
    } catch (e: any) {
      setSettingsNotice("toolbar", "error", e?.message || String(e));
    }
  }

  async function adminCreateInvite(label: string) {
    setAdminBusy(true);
    clearAdmin();
    try {
      const url = `${settings.apiBase || ""}/api/admin/invite`;
      const headers = apiHeaders();
      const res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify({ label }),
      });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(apiErrorText(data, res.status));
      setNewInvite(data as InviteCreateResponse);
      setAdminNotice("invite", "success", "Invite created. Token is shown once.");
    } catch (e: any) {
      setAdminNotice("toolbar", "error", e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminListUsers() {
    setAdminBusy(true);
    setNotices((prev) => withAdminNotice(prev, "users", null));
    try {
      const url = `${settings.apiBase || ""}/api/admin/users`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(apiErrorText(data, res.status));
      const payload = data as UserListResponse;
      setUsers(payload);
      setAdminNotice("users", "success", `Loaded ${payload.items.length} user(s).`);
    } catch (e: any) {
      setAdminNotice("users", "error", e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminSetStatus(userId: string, status: "active" | "blocked") {
    setAdminBusy(true);
    setNotices((prev) => withAdminNotice(prev, "toolbar", null));
    try {
      const url = `${settings.apiBase || ""}/api/admin/users/${userId}/status`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "POST", headers, body: JSON.stringify({ status }) });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(apiErrorText(data, res.status));
      setAdminNotice("toolbar", "success", `User ${userId} set to ${status}.`);
      await adminListUsers();
    } catch (e: any) {
      setAdminNotice("toolbar", "error", e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminRotate(userId: string) {
    setAdminBusy(true);
    setNotices((prev) => withAdminNotice(prev, "toolbar", null));
    try {
      const url = `${settings.apiBase || ""}/api/admin/users/${userId}/rotate`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "POST", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(apiErrorText(data, res.status));
      setNewInvite(data as InviteCreateResponse);
      setAdminNotice("invite", "success", `Token rotated for ${userId}.`);
      await adminListUsers();
    } catch (e: any) {
      setAdminNotice("toolbar", "error", e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminLoadUsage(userId: string) {
    setAdminBusy(true);
    setNotices((prev) => withAdminNotice(prev, "usage", null));
    try {
      const url = `${settings.apiBase || ""}/api/admin/usage?user_id=${encodeURIComponent(userId)}&limit=100`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(apiErrorText(data, res.status));
      setAdminUsage(data as UsageListResponse);
      setAdminNotice("usage", "success", `Loaded usage for ${userId}.`);
    } catch (e: any) {
      setAdminNotice("usage", "error", e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function onLoadTtsOptions(silent = false) {
    if (!settings.userToken.trim()) {
      if (!silent) setSettingsNotice("access", "warning", "Invite token is required to load TTS options.");
      return;
    }
    if (ttsOptionsBusy) {
      return;
    }
    if (!silent) {
      setNotices((prev) => withSettingsNotice(prev, "generation", null));
      setNotices((prev) => withSettingsNotice(prev, "audio", null));
    }

    setTtsOptionsBusy(true);
    try {
      const url = `${settings.apiBase || ""}/api/tts/options`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        throw new Error(apiErrorText(data, res.status));
      }
      const payload = data as TTSOptionsResponse;
      setTtsOptions(payload);
      setTtsOptionsLoadedAt(Date.now());

      setSettings((current) => {
        const currentProvider = (current.audioProvider || "").trim().toLowerCase();
        const providerFromApi = payload.providers[0] || "openai";
        const provider = payload.by_provider[currentProvider] ? currentProvider : providerFromApi;
        const selected = payload.by_provider[provider];
        const textModels = payload.text_models || [];
        const currentTextModel = (current.model || "").trim();
        const currentModel = (current.audioModel || "").trim();
        const currentVoice = (current.audioVoice || "").trim();
        const textModelInList = !!currentTextModel && textModels.includes(currentTextModel);
        const modelInList = !!currentModel && (selected?.models || []).includes(currentModel);
        return {
          ...current,
          model: currentTextModel && textModelInList ? currentTextModel : textModels[0] || currentTextModel || "gpt-4.1-mini",
          audioProvider: provider,
          audioModel: currentModel && modelInList ? currentModel : selected?.default_model || currentModel,
          audioVoice: currentVoice || selected?.default_voice || "",
        };
      });

      if (!silent) {
        setSettingsNotice("audio", "success", "Model and voice lists were refreshed.");
      }
    } catch (e: any) {
      if (!silent) {
        setSettingsNotice("audio", "error", e?.message || String(e));
      }
    } finally {
      setTtsOptionsBusy(false);
    }
  }

  async function onCheckElevenLabsVoiceId(voiceId: string) {
    const cleanedVoiceId = voiceId.trim();
    if (!cleanedVoiceId) {
      setSettingsNotice("audio", "warning", "Paste an ElevenLabs voice ID before checking it.");
      return;
    }
    if (!settings.userToken.trim()) {
      setSettingsNotice("access", "warning", "Invite token is required to check ElevenLabs voices.");
      return;
    }

    setTtsOptionsBusy(true);
    try {
      const payload: TTSVoiceCheckRequest = { provider: "elevenlabs", voice_id: cleanedVoiceId };
      const url = `${settings.apiBase || ""}/api/tts/voice/check`;
      const res = await fetch(url, {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify(payload),
      });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        throw new Error(apiErrorText(data, res.status));
      }

      const checked = data as TTSVoiceCheckResponse;
      const elevenOptions = ttsOptions?.by_provider?.elevenlabs;
      const elevenModels = elevenOptions?.models || [];
      setCustomAudioVoiceLabels((current) => {
        const next = { ...current, [checked.id]: checked.label || checked.id };
        saveJson(CUSTOM_AUDIO_VOICE_LABELS_KEY, next);
        return next;
      });
      setSettings((current) => ({
        ...current,
        audioProvider: "elevenlabs",
        audioModel: elevenModels.includes(current.audioModel)
          ? current.audioModel
          : elevenOptions?.default_model || elevenModels[0] || "eleven_multilingual_v2",
        audioVoice: checked.id,
      }));
      setSettingsNotice("audio", "success", `ElevenLabs voice selected: ${checked.label || checked.id}.`);
    } catch (e: any) {
      setSettingsNotice("audio", "error", e?.message || String(e));
    } finally {
      setTtsOptionsBusy(false);
    }
  }

  async function onAddElevenLabsSharedVoice(params: { publicUserId: string; voiceId: string; newName: string }): Promise<void> {
    const publicUserId = params.publicUserId.trim();
    const voiceId = params.voiceId.trim();
    const newName = params.newName.trim();
    if (!publicUserId || !voiceId || !newName) {
      setSettingsNotice("audio", "warning", "Public user ID, voice ID, and name are required to add a shared ElevenLabs voice.");
      return;
    }
    if (!settings.userToken.trim()) {
      setSettingsNotice("access", "warning", "Invite token is required to add ElevenLabs shared voices.");
      return;
    }

    setTtsOptionsBusy(true);
    try {
      const payload: TTSSharedVoiceAddRequest = {
        public_user_id: publicUserId,
        voice_id: voiceId,
        new_name: newName,
        bookmarked: true,
      };
      const res = await fetch(`${settings.apiBase || ""}/api/tts/voice/add-shared`, {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify(payload),
      });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        throw new Error(apiErrorText(data, res.status));
      }

      const added = data as TTSSharedVoiceAddResponse;
      const elevenOptions = ttsOptions?.by_provider?.elevenlabs;
      const elevenModels = elevenOptions?.models || [];
      setCustomAudioVoiceLabels((current) => {
        const next = { ...current, [added.id]: added.label || added.id };
        saveJson(CUSTOM_AUDIO_VOICE_LABELS_KEY, next);
        return next;
      });
      setSettings((current) => ({
        ...current,
        audioProvider: "elevenlabs",
        audioModel: elevenModels.includes(current.audioModel)
          ? current.audioModel
          : elevenOptions?.default_model || elevenModels[0] || "eleven_multilingual_v2",
        audioVoice: added.id,
      }));
      setSettingsNotice("audio", "success", `Shared ElevenLabs voice added and selected: ${added.label || added.id}.`);
    } catch (e: any) {
      setSettingsNotice("audio", "error", e?.message || String(e));
    } finally {
      setTtsOptionsBusy(false);
    }
  }

  async function onPreviewTtsVoice(sampleText: string): Promise<string> {
    const text = sampleText.trim();
    const provider = (settings.audioProvider || "openai").trim().toLowerCase();
    if (!text) {
      throw new Error("Preview text is required.");
    }
    if (!settings.userToken.trim()) {
      setSettingsNotice("access", "warning", "Invite token is required to preview TTS voices.");
      throw new Error("Invite token is required.");
    }
    if (!settings.audioVoice.trim()) {
      throw new Error("Select a TTS voice before previewing it.");
    }
    if (provider === "elevenlabs" && !settings.audioModel.trim()) {
      throw new Error("Select an ElevenLabs TTS model before previewing the voice.");
    }

    const payload: TTSPreviewRequest = {
      provider,
      model: settings.audioModel || undefined,
      voice: settings.audioVoice,
      text,
    };
    const res = await fetch(`${settings.apiBase || ""}/api/tts/preview`, {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify(payload),
    });
    const data = (await res.json().catch(() => null)) as any;
    if (!res.ok) {
      throw new Error(apiErrorText(data, res.status));
    }
    const preview = data as TTSPreviewResponse;
    if (!preview.audio_b64) {
      throw new Error("TTS preview returned no audio.");
    }
    setSettingsNotice("audio", "success", `Voice preview ready (${preview.model}, ${preview.voice}).`);
    return `data:audio/mpeg;base64,${preview.audio_b64}`;
  }

  async function onGenerate() {
    const tempParsed = parseTemperatureValue(settings.temperature);
    if (tempParsed.error) {
      setGenerateNotice("run", "error", tempParsed.error);
      return;
    }

    clearGenerate();

    const totalRows = parsed.items.length;
    const audioRequested = settings.generateAudio;
    const textProgressCap = audioRequested ? 72 : 94;
    const startedAt = Date.now();
    const WAITING_PROVIDER_MS = 4000;
    const TEXT_BATCH_SIZE = totalRows > 80 ? 20 : totalRows > 40 ? 16 : totalRows > 20 ? 10 : 10;
    let progressTimer: number | null = null;
    let generatedRows = 0;

    const refreshUsageSilently = async () => {
      try {
        const usageUrl = `${settings.apiBase || ""}/api/usage?limit=50`;
        const usageRes = await fetch(usageUrl, { method: "GET", headers: apiHeaders() });
        const usageData = (await usageRes.json().catch(() => null)) as any;
        if (usageRes.ok) {
          setUsage(usageData as UsageListResponse);
        }
      } catch {
        // Silent refresh: ignore errors.
      }
    };

    const updateProgressMeta = (patch: Partial<GenerateProgressMeta>) => {
      setGenerateProgressMeta((prev) => ({
        stage: patch.stage ?? prev?.stage ?? "queued",
        done: patch.done ?? prev?.done ?? 0,
        total: patch.total ?? prev?.total ?? (audioRequested ? 0 : totalRows),
        batchIndex: patch.batchIndex ?? prev?.batchIndex ?? 0,
        batchTotal: patch.batchTotal ?? prev?.batchTotal ?? 0,
        elapsedMs: patch.elapsedMs ?? Date.now() - startedAt,
        waitingProvider: patch.waitingProvider ?? prev?.waitingProvider ?? false,
      }));
    };

    setBusy(true);
    setResponse(null);
    setAudioMediaMap({});
    setAudioRunSummary(null);
    setGenerateProgress(3);
    setGenerateProgressLabel(`Queued ${totalRows} row(s) for generation.`);
    setGenerateNotice("run", "info", `Queued ${totalRows} row(s) for generation.`);
    updateProgressMeta({
      stage: "queued",
      done: 0,
      total: totalRows,
      batchIndex: 0,
      batchTotal: 0,
      waitingProvider: false,
      elapsedMs: 0,
    });

    try {
      const runId = generateRunId();
      const url = `${settings.apiBase || ""}/api/generate`;
      const jobsCreateUrl = `${settings.apiBase || ""}/api/jobs/generate`;
      const jobsWorkerUrl = `${settings.apiBase || ""}/api/jobs/generate/worker`;
      const headers = apiHeaders();
      const useAsyncGenerate = (window.localStorage.getItem("use_async_generate") || "1").trim() !== "0";
      // For small runs, synchronous mode is usually faster on Vercel because it avoids
      // queue polling + extra DB/auth roundtrips.
      const preferSyncForSmallRuns = totalRows <= 40;
      const preferDirectForTextReuse = settings.reuseTextCards;
      let asyncGenerateEnabled = useAsyncGenerate && !preferSyncForSmallRuns;
      if (useAsyncGenerate && preferSyncForSmallRuns) {
        setGenerateNotice("run", "info", "Small run detected: using direct mode for lower latency.");
      }
      if (useAsyncGenerate && preferDirectForTextReuse) {
        asyncGenerateEnabled = false;
        setGenerateNotice("run", "info", "Saved-card reuse enabled: using direct mode to avoid queue overhead.");
      }
      const totalTextBatches = Math.max(1, Math.ceil(totalRows / TEXT_BATCH_SIZE));
      const mergedItems: GenerateResponse["items"] = [];
      let mergedElapsedMs = 0;
      let mergedTextCacheHits = 0;
      let mergedTextAssetsStored = 0;
      let mergedTextCacheErrors = 0;
      let lastPayload: GenerateResponse | null = null;
      const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

      const runBatchViaJobQueue = async (
        req: GenerateRequest,
        batchIndex: number,
        doneBefore: number,
        batchSize: number
      ): Promise<GenerateResponse> => {
        const workerChunkSize = Math.max(1, Math.min(batchSize, settings.reuseTextCards ? 12 : 8));
        const createdRes = await fetch(jobsCreateUrl, { method: "POST", headers, body: JSON.stringify(req) });
        const createdData = (await createdRes.json().catch(() => null)) as any;
        if (!createdRes.ok) {
          throw new Error(`Batch ${batchIndex}/${totalTextBatches}: ${apiErrorText(createdData, createdRes.status)}`);
        }
        const created = createdData as GenerateJobCreateResponse;
        if (!created?.job_id) {
          throw new Error(`Batch ${batchIndex}/${totalTextBatches}: job_id was not returned by API.`);
        }
        const jobId = created.job_id;
        const pollDeadline = Date.now() + 15 * 60 * 1000;
        let lastStatus: GenerateJobStatusResponse | null = null;
        while (Date.now() < pollDeadline) {
          const workerRes = await fetch(jobsWorkerUrl, {
            method: "POST",
            headers,
            body: JSON.stringify({ job_id: jobId, max_items: workerChunkSize }),
          }).catch(() => null);
          if (workerRes) {
            const workerData = (await workerRes.json().catch(() => null)) as any;
            if (!workerRes.ok) {
              throw new Error(`Batch ${batchIndex}/${totalTextBatches}: ${apiErrorText(workerData, workerRes.status)}`);
            }
            const workerStatus = String(workerData?.status || "").toLowerCase();
            if (workerStatus === "failed") {
              throw new Error(`Batch ${batchIndex}/${totalTextBatches}: ${workerData?.message || "worker failed"}`);
            }
          }

          const statusRes = await fetch(`${jobsCreateUrl}/${encodeURIComponent(jobId)}`, {
            method: "GET",
            headers,
          });
          const statusData = (await statusRes.json().catch(() => null)) as any;
          if (!statusRes.ok) {
            throw new Error(`Batch ${batchIndex}/${totalTextBatches}: ${apiErrorText(statusData, statusRes.status)}`);
          }
          const status = statusData as GenerateJobStatusResponse;
          lastStatus = status;
          const processed = Math.max(0, Math.min(batchSize, status.processed_items || 0));
          const doneNow = Math.min(totalRows, doneBefore + processed);
          updateProgressMeta({
            stage: "text",
            done: doneNow,
            total: totalRows,
            batchIndex,
            batchTotal: totalTextBatches,
            waitingProvider: status.status === "running",
            elapsedMs: Date.now() - startedAt,
          });
          if (status.status === "done") {
            if (!status.result) {
              throw new Error(`Batch ${batchIndex}/${totalTextBatches}: completed without result payload.`);
            }
            return status.result;
          }
          if (status.status === "failed") {
            throw new Error(
              `Batch ${batchIndex}/${totalTextBatches}: ${status.error || "generation job failed on server"}`
            );
          }
          await sleep(850);
        }
        const suffix = lastStatus?.status ? ` (last status: ${lastStatus.status})` : "";
        throw new Error(`Batch ${batchIndex}/${totalTextBatches}: timeout while waiting for job${suffix}.`);
      };

      for (let offset = 0; offset < totalRows; offset += TEXT_BATCH_SIZE) {
        const batchIndex = Math.floor(offset / TEXT_BATCH_SIZE) + 1;
        const batchItems = parsed.items.slice(offset, offset + TEXT_BATCH_SIZE);
        const doneBefore = mergedItems.length;
        const batchStartedAt = Date.now();

        const baseProgress = Math.min(
          textProgressCap,
          6 + (doneBefore / Math.max(totalRows, 1)) * Math.max(1, textProgressCap - 6)
        );
        const batchCapProgress = Math.min(
          textProgressCap,
          6 + ((doneBefore + batchItems.length * 0.9) / Math.max(totalRows, 1)) * Math.max(1, textProgressCap - 6)
        );
        const progressSpan = Math.max(0.5, batchCapProgress - baseProgress);
        let waitingProgress = baseProgress;

        setGenerateProgress((prev) => (baseProgress > prev ? baseProgress : prev));
        setGenerateProgressLabel(`Generating cards: ${doneBefore}/${totalRows}. Batch ${batchIndex}/${totalTextBatches} in progress...`);
        updateProgressMeta({
          stage: "text",
          done: doneBefore,
          total: totalRows,
          batchIndex,
          batchTotal: totalTextBatches,
          waitingProvider: false,
          elapsedMs: Date.now() - startedAt,
        });

        if (progressTimer != null) {
          window.clearInterval(progressTimer);
          progressTimer = null;
        }

        progressTimer = window.setInterval(() => {
          const step = Math.max(0.08, progressSpan / 18);
          waitingProgress = Math.min(batchCapProgress, waitingProgress + step);
          const waitingProvider = Date.now() - batchStartedAt >= WAITING_PROVIDER_MS;
          const ratio = Math.max(0, Math.min(1, (waitingProgress - baseProgress) / progressSpan));
          const estimatedDone = Math.min(totalRows, doneBefore + Math.round(ratio * batchItems.length * 0.9));

          setGenerateProgress((prev) => (waitingProgress > prev ? waitingProgress : prev));
          setGenerateProgressLabel(
            `Generating cards: ${doneBefore}/${totalRows}. Batch ${batchIndex}/${totalTextBatches} in progress${
              waitingProvider ? " (waiting provider...)" : ""
            }...`
          );
          updateProgressMeta({
            stage: "text",
            done: estimatedDone,
            total: totalRows,
            batchIndex,
            batchTotal: totalTextBatches,
            waitingProvider,
            elapsedMs: Date.now() - startedAt,
          });
        }, 350);

        const req: GenerateRequest = {
          run_id: runId,
          prompt_version: settings.promptVersion,
          provider: settings.provider,
          model: settings.model,
          cefr: settings.cefr,
          profile: settings.profile,
          l1: settings.l1,
          temperature: tempParsed.ratio,
          flags: {
            force_schema: true,
            allow_repair: true,
            reuse_text_cache: settings.reuseTextCards,
          },
          items: batchItems,
        };

        try {
          let batchPayload: GenerateResponse;
          if (asyncGenerateEnabled) {
            try {
              batchPayload = await runBatchViaJobQueue(req, batchIndex, doneBefore, batchItems.length);
            } catch (jobErr: any) {
              const msg = String(jobErr?.message || jobErr || "");
              if (msg.includes("404") || msg.toLowerCase().includes("not found")) {
                asyncGenerateEnabled = false;
                setGenerateNotice("run", "warning", "Async generate endpoints are unavailable; falling back to sync mode.");
                const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(req) });
                const data = (await res.json().catch(() => null)) as any;
                if (!res.ok) {
                  throw new Error(`Batch ${batchIndex}/${totalTextBatches}: ${apiErrorText(data, res.status)}`);
                }
                batchPayload = data as GenerateResponse;
              } else {
                throw jobErr;
              }
            }
          } else {
            const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(req) });
            const data = (await res.json().catch(() => null)) as any;
            if (!res.ok) {
              const canRetryAsync = useAsyncGenerate && [502, 503, 504].includes(res.status);
              if (canRetryAsync) {
                setGenerateNotice(
                  "run",
                  "warning",
                  `Direct batch ${batchIndex}/${totalTextBatches} hit HTTP ${res.status}; retrying through job queue.`
                );
                batchPayload = await runBatchViaJobQueue(req, batchIndex, doneBefore, batchItems.length);
              } else {
              throw new Error(`Batch ${batchIndex}/${totalTextBatches}: ${apiErrorText(data, res.status)}`);
              }
            } else {
              batchPayload = data as GenerateResponse;
            }
          }
          lastPayload = batchPayload;
          mergedElapsedMs += Number(batchPayload.timing?.elapsed_ms || 0);
          mergedTextCacheHits += Number(batchPayload.timing?.text_cache_hits || 0);
          mergedTextAssetsStored += Number(batchPayload.timing?.text_assets_stored || 0);
          mergedTextCacheErrors += Number(batchPayload.timing?.text_cache_errors || 0);
          mergedItems.push(...(Array.isArray(batchPayload.items) ? batchPayload.items : []));
          generatedRows = mergedItems.length;

          const partialPayload: GenerateResponse = {
            ...batchPayload,
            run_id: runId,
            items: [...mergedItems],
            timing: {
              elapsed_ms: mergedElapsedMs > 0 ? mergedElapsedMs : Date.now() - startedAt,
              text_cache_hits: mergedTextCacheHits,
              text_assets_stored: mergedTextAssetsStored,
              text_cache_errors: mergedTextCacheErrors,
            },
          };
          setResponse(partialPayload);

          const textProgress = Math.min(
            textProgressCap,
            6 + (generatedRows / Math.max(totalRows, 1)) * Math.max(1, textProgressCap - 6)
          );
          setGenerateProgress((prev) => (textProgress > prev ? textProgress : prev));
          setGenerateProgressLabel(`Text ready: ${generatedRows}/${totalRows}. Batch ${batchIndex}/${totalTextBatches} complete.`);
          updateProgressMeta({
            stage: "text",
            done: generatedRows,
            total: totalRows,
            batchIndex,
            batchTotal: totalTextBatches,
            waitingProvider: false,
            elapsedMs: Date.now() - startedAt,
          });
        } finally {
          if (progressTimer != null) {
            window.clearInterval(progressTimer);
            progressTimer = null;
          }
        }
      }

      if (!lastPayload) {
        throw new Error("No generate response payload returned by API.");
      }

      let payload: GenerateResponse = {
        ...lastPayload,
        run_id: runId,
        items: mergedItems,
        timing: {
          elapsed_ms: mergedElapsedMs > 0 ? mergedElapsedMs : Date.now() - startedAt,
          text_cache_hits: mergedTextCacheHits,
          text_assets_stored: mergedTextAssetsStored,
          text_cache_errors: mergedTextCacheErrors,
        },
      };
      setResponse(payload);

      setGenerateProgress(textProgressCap);
      setGenerateProgressLabel(`Text ready: ${payload.items.length}/${totalRows}.`);
      updateProgressMeta({
        stage: "text",
        done: payload.items.length,
        total: totalRows,
        batchIndex: totalTextBatches,
        batchTotal: totalTextBatches,
        waitingProvider: false,
        elapsedMs: Date.now() - startedAt,
      });

      if (settings.generateAudio) {
        setGenerateProgressLabel("Preparing audio synthesis...");
        updateProgressMeta({
          stage: "audio",
          done: 0,
          total: 0,
          batchIndex: 0,
          batchTotal: 0,
          waitingProvider: false,
          elapsedMs: Date.now() - startedAt,
        });

        const audioVerification = await verifyAttachedAudioAssets(
          payload,
          {
            includeWord: settings.includeAudioWord,
            includeSentence: settings.includeAudioSentence,
          },
          headers
        );
        if (audioVerification.payload !== payload) {
          payload = audioVerification.payload;
          setResponse(payload);
        }

        const ttsItems = buildTtsItems(payload, {
          includeWord: settings.includeAudioWord,
          includeSentence: settings.includeAudioSentence,
        });
        const attachedAudioClips = countAttachedAudioClips(payload, {
          includeWord: settings.includeAudioWord,
          includeSentence: settings.includeAudioSentence,
        });

        if (ttsItems.length > 0) {
          const totalClips = ttsItems.length;
          const audioProviderForRun = (settings.audioProvider || "openai").trim().toLowerCase();
          const ttsBatchSize = 6;
          const ttsBatchTimeoutMs =
            audioProviderForRun === "elevenlabs" ? ELEVENLABS_TTS_BATCH_TIMEOUT_MS : OPENAI_TTS_BATCH_TIMEOUT_MS;
          const totalBatches = Math.ceil(totalClips / ttsBatchSize);
          let doneClips = 0;
          let okCount = 0;
          let failedCount = 0;
          let storedClipCount = 0;
          let cachedClipCount = 0;
          let durableCachedClipCount = 0;
          let storedReusableAssetCount = 0;
          let persistedAudioReady = true;
          const errorSamples: string[] = [];
          const storageErrors: string[] = [];
          const diagnosticLines: string[] = [...audioVerification.diagnostics];
          const mediaMap: Record<string, string> = {};
          const audioProgressStart = Math.min(94, textProgressCap + 4);
          const audioProgressSpan = Math.max(1, 99 - audioProgressStart);

          const updateAudioProgress = (
            done: number,
            batchIndex: number,
            waitingProvider: boolean,
            suffix = ""
          ) => {
            const ratio = totalClips > 0 ? done / totalClips : 1;
            const next = Math.min(99, audioProgressStart + ratio * audioProgressSpan);
            setGenerateProgress((prev) => (next > prev ? next : prev));
            const tail = suffix ? ` ${suffix}` : "";
            const waitingText = waitingProvider ? " waiting provider..." : "";
            setGenerateProgressLabel(`Generating audio: ${done}/${totalClips} clips.${tail}${waitingText}`);
            updateProgressMeta({
              stage: "audio",
              done,
              total: totalClips,
              batchIndex,
              batchTotal: totalBatches,
              waitingProvider,
              elapsedMs: Date.now() - startedAt,
            });
          };

          updateAudioProgress(0, 0, false, "Batch 1 in queue.");

          try {
            const ttsUrl = `${settings.apiBase || ""}/api/tts`;
            const ttsQueue: TTSRequest["items"][] = [];
            for (let offset = 0; offset < totalClips; offset += ttsBatchSize) {
              ttsQueue.push(ttsItems.slice(offset, offset + ttsBatchSize));
            }
            let batchIndex = 0;
            while (ttsQueue.length > 0) {
              batchIndex += 1;
              const batch = ttsQueue.shift() || [];
              if (batch.length === 0) continue;
              const displayBatchTotal = Math.max(totalBatches, batchIndex + ttsQueue.length);
              const doneTarget = Math.min(totalClips, doneClips + batch.length);
              const baseRatio = totalClips > 0 ? doneClips / totalClips : 1;
              const doneRatio = totalClips > 0 ? doneTarget / totalClips : 1;
              const baseProgress = Math.min(99, audioProgressStart + baseRatio * audioProgressSpan);
              const waitingCap = Math.min(
                99,
                audioProgressStart + (baseRatio + (doneRatio - baseRatio) * 0.85) * audioProgressSpan
              );
              let waitingProgress = baseProgress;
              const batchStartedAt = Date.now();

              updateAudioProgress(doneClips, batchIndex, false, `Batch ${batchIndex}/${displayBatchTotal} in progress...`);

              let waitTimer: number | null = window.setInterval(() => {
                const step = Math.max(0.08, (waitingCap - baseProgress) / 16);
                waitingProgress = Math.min(waitingCap, waitingProgress + step);
                const waitingProvider = Date.now() - batchStartedAt >= WAITING_PROVIDER_MS;
                setGenerateProgress((prev) => (waitingProgress > prev ? waitingProgress : prev));
                updateAudioProgress(
                  doneClips,
                  batchIndex,
                  waitingProvider,
                  `Batch ${batchIndex}/${displayBatchTotal} in progress...`
                );
              }, 350);

              const ttsReq: TTSRequest = {
                run_id: payload.run_id || runId,
                provider: settings.audioProvider || "openai",
                model: settings.audioModel || undefined,
                voice: settings.audioVoice || undefined,
                items: batch,
              };

              try {
                const controller = new AbortController();
                const timeoutHandle = window.setTimeout(() => controller.abort(), ttsBatchTimeoutMs);
                let ttsRes: Response;
                try {
                  ttsRes = await fetch(ttsUrl, {
                    method: "POST",
                    headers,
                    body: JSON.stringify(ttsReq),
                    signal: controller.signal,
                  });
                } finally {
                  window.clearTimeout(timeoutHandle);
                }
                const ttsData = (await ttsRes.json().catch(() => null)) as any;
                if (!ttsRes.ok) {
                  const detail = apiErrorText(ttsData, ttsRes.status);
                  const retryableStatus = [429, 502, 503, 504].includes(ttsRes.status);
                  diagnosticLines.push("Batch " + batchIndex + "/" + displayBatchTotal + ": HTTP " + ttsRes.status + " after " + formatElapsedMs(Date.now() - batchStartedAt) + ".");
                  appendUniqueError(errorSamples, detail);
                  if (retryableStatus && batch.length > 1) {
                    const midpoint = Math.ceil(batch.length / 2);
                    ttsQueue.unshift(batch.slice(0, midpoint), batch.slice(midpoint));
                    updateAudioProgress(doneClips, batchIndex, false, "Retrying smaller audio batch.");
                    continue;
                  }
                  failedCount += batch.length;
                  doneClips = doneTarget;
                  updateAudioProgress(doneClips, batchIndex, false, "Continuing after audio batch error.");
                  continue;
                }

                const ttsPayload = ttsData as TTSResponse;
                const batchHttpElapsedMs = Date.now() - batchStartedAt;
                payload = mergeTtsIntoResponse(payload, ttsPayload);
                const batchAudios = Array.isArray(ttsPayload.audios) ? ttsPayload.audios : [];

                for (const audio of batchAudios) {
                  if (audio.status !== "failed" && audio.filename && audio.audio_b64) {
                    mediaMap[audio.filename] = audio.audio_b64;
                  }
                }

                let batchOk = batchAudios.filter((audio) => audio.status === "ok" || audio.status === "cached").length;
                let batchFailed = batchAudios.filter((audio) => audio.status === "failed").length;

                for (const audio of batchAudios) {
                  if (audio.status === "failed") appendUniqueError(errorSamples, audio.error);
                }

                const missingStatuses = Math.max(0, batch.length - (batchOk + batchFailed));
                if (missingStatuses > 0) {
                  batchFailed += missingStatuses;
                  appendUniqueError(errorSamples, "Some clip statuses were missing in TTS response.");
                }

                okCount += batchOk;
                failedCount += batchFailed;

                for (const err of ttsPayload.summary?.errors || []) {
                  appendUniqueError(errorSamples, err);
                }

                const storedClips = Math.max(0, Number(ttsPayload.storage?.stored_clips || 0));
                storedClipCount += storedClips;
                const expectedStoredAssets = new Set(
                  batchAudios
                    .filter((audio) => (audio.status === "ok" || audio.status === "cached") && !!audio.filename)
                    .map((audio) => audio.filename as string)
                ).size;
                if (expectedStoredAssets > 0) {
                  const batchPersisted = !!ttsPayload.storage?.persisted && storedClips >= expectedStoredAssets;
                  if (!batchPersisted) {
                    persistedAudioReady = false;
                    appendUniqueError(
                      storageErrors,
                      ttsPayload.storage?.error ||
                        `Server-side audio persistence failed for batch ${batchIndex}/${displayBatchTotal}.`
                    );
                  }
                } else if (ttsPayload.storage?.error) {
                  appendUniqueError(storageErrors, ttsPayload.storage.error);
                }

                const backendElapsedMs = Number(ttsPayload.timing?.elapsed_ms || 0);
                const synthesisMs = Number(ttsPayload.timing?.synthesis_ms || 0);
                const storageMs = Number(ttsPayload.timing?.storage_ms || 0);
                const cacheHits = Number(ttsPayload.timing?.cache_hits || ttsPayload.summary?.cached || 0);
                const durableCacheHits = Number(ttsPayload.timing?.durable_cache_hits || 0);
                const audioAssetsStored = Number(ttsPayload.timing?.audio_assets_stored || 0);
                cachedClipCount += cacheHits;
                durableCachedClipCount += durableCacheHits;
                storedReusableAssetCount += audioAssetsStored;
                const durableCacheError = String(ttsPayload.timing?.durable_cache_error || "");
                const audioAssetsStorageError = String(ttsPayload.timing?.audio_assets_storage_error || "");
                const uniqueMedia = Number(ttsPayload.timing?.unique_media_files || expectedStoredAssets || 0);
                let diagnosticLine =
                  "Batch " +
                    batchIndex +
                    "/" +
                    displayBatchTotal +
                    ": " +
                    formatElapsedMs(batchHttpElapsedMs) +
                    " round trip, " +
                    (backendElapsedMs ? formatElapsedMs(backendElapsedMs) : "n/a") +
                    " backend (synth " +
                    (synthesisMs ? formatElapsedMs(synthesisMs) : "n/a") +
                    ", storage " +
                    formatElapsedMs(storageMs) +
                    "), ok " +
                    batchOk +
                    ", cached " +
                    cacheHits +
                    " (durable " +
                    durableCacheHits +
                    ", stored assets " +
                    audioAssetsStored +
                    ")" +
                    ", failed " +
                    batchFailed +
                    ", media " +
                    uniqueMedia +
                    ".";
                if (durableCacheError) diagnosticLine += " Durable cache read: " + durableCacheError + ".";
                if (audioAssetsStorageError) diagnosticLine += " Durable cache write: " + audioAssetsStorageError + ".";
                diagnosticLines.push(diagnosticLine);

                doneClips = doneTarget;
                setResponse(payload);
                updateAudioProgress(doneClips, batchIndex, false, `Batch ${batchIndex}/${displayBatchTotal} complete.`);
              } catch (fetchErr: any) {
                const isAbort = fetchErr?.name === "AbortError";
                const detail = isAbort
                  ? `TTS batch timed out after ${formatElapsedMs(ttsBatchTimeoutMs)}.`
                  : fetchErr?.message || String(fetchErr);
                diagnosticLines.push(
                  "Batch " +
                    batchIndex +
                    "/" +
                    displayBatchTotal +
                    ": " +
                    detail +
                    " after " +
                    formatElapsedMs(Date.now() - batchStartedAt) +
                    "."
                );
                appendUniqueError(errorSamples, detail);
                if (batch.length > 1) {
                  const midpoint = Math.ceil(batch.length / 2);
                  ttsQueue.unshift(batch.slice(0, midpoint), batch.slice(midpoint));
                  updateAudioProgress(doneClips, batchIndex, false, "Retrying smaller audio batch after timeout.");
                  continue;
                }
                failedCount += batch.length;
                doneClips = doneTarget;
                updateAudioProgress(doneClips, batchIndex, false, "Continuing after audio timeout.");
                continue;
              } finally {
                if (waitTimer != null) {
                  window.clearInterval(waitTimer);
                  waitTimer = null;
                }
              }
            }

            setAudioMediaMap(mediaMap);
            setResponse(payload);

            const resolvedOk = Math.min(okCount, totalClips);
            const resolvedFailed = Math.max(failedCount, totalClips - resolvedOk);
            const combinedErrors = [...errorSamples];
            for (const storageError of storageErrors) {
              appendUniqueError(combinedErrors, storageError);
            }
            const finalPersisted = resolvedOk > 0 ? persistedAudioReady : true;

            setAudioRunSummary({
              requested: true,
              total: totalClips,
              ok: resolvedOk,
              failed: resolvedFailed,
              errors: combinedErrors,
              persisted: finalPersisted,
              storedClips: storedClipCount,
              cachedClips: cachedClipCount,
              durableCachedClips: durableCachedClipCount,
              storedReusableAssets: storedReusableAssetCount,
              storageError: storageErrors[0] || null,
              diagnostics: diagnosticLines,
            });

            if (resolvedFailed > 0) {
              const firstErr = combinedErrors.find((err) => typeof err === "string" && err.trim().length > 0);
              const details = firstErr ? `First error: ${firstErr}` : undefined;
              if (resolvedOk > 0) {
                setGenerateNotice(
                  "audio",
                  "warning",
                  `Audio partial: ${resolvedOk} succeeded, ${resolvedFailed} failed.`,
                  details
                );
              } else {
                setGenerateNotice("audio", "error", `Audio failed for all clips (${resolvedFailed}).`, details);
              }
            } else if (!finalPersisted) {
              const details = storageErrors[0] || "Large APKG export on Vercel may fail until server-side storage is available.";
              setGenerateNotice(
                "audio",
                "warning",
                `Audio ready: ${resolvedOk} clip(s), but server-side export storage is unavailable.`,
                details
              );
            } else {
              setGenerateNotice("audio", "success", `Audio ready: ${resolvedOk} clip(s).`);
            }
          } catch (ttsErr: any) {
            const detail = ttsErr?.message || String(ttsErr);
            appendUniqueError(errorSamples, detail);
            const unresolvedFailed = Math.max(totalClips - okCount, failedCount);
            setAudioRunSummary({
              requested: true,
              total: totalClips,
              ok: okCount,
              failed: unresolvedFailed,
              errors: errorSamples,
              persisted: false,
              storedClips: 0,
              cachedClips: cachedClipCount,
              durableCachedClips: durableCachedClipCount,
              storedReusableAssets: storedReusableAssetCount,
              storageError: null,
              diagnostics: diagnosticLines,
            });
            setGenerateNotice("audio", "error", "Audio synthesis did not complete.", detail);
          }
        } else {
          const diagnostics = [...audioVerification.diagnostics];
          if (attachedAudioClips > 0) {
            diagnostics.push("Audio synthesis skipped: selected clips were already attached to saved cards.");
          }
          setAudioRunSummary({ requested: true, total: attachedAudioClips, ok: attachedAudioClips, failed: 0, errors: [], persisted: true, storedClips: 0, cachedClips: attachedAudioClips, durableCachedClips: attachedAudioClips, storedReusableAssets: 0, storageError: null, diagnostics });
          setGenerateProgress((prev) => (95 > prev ? 95 : prev));
          setGenerateProgressLabel(
            attachedAudioClips > 0
              ? `Audio already attached: ${attachedAudioClips} clip(s).`
              : "Audio is enabled, but there are no clips to synthesize."
          );
          updateProgressMeta({
            stage: "audio",
            done: 0,
            total: 0,
            batchIndex: 0,
            batchTotal: 0,
            waitingProvider: false,
            elapsedMs: Date.now() - startedAt,
          });
          setGenerateNotice(
            "audio",
            attachedAudioClips > 0 ? "success" : "info",
            attachedAudioClips > 0
              ? `Audio already attached: ${attachedAudioClips} clip(s), synthesis skipped.`
              : "Audio enabled, but no eligible clips were found."
          );
        }
      } else {
        setAudioRunSummary({ requested: false, total: 0, ok: 0, failed: 0, errors: [], persisted: false, storedClips: 0, cachedClips: 0, durableCachedClips: 0, storedReusableAssets: 0, storageError: null, diagnostics: [] });
        setGenerateNotice("audio", "info", "Audio generation is disabled for this run.");
      }

      setGenerateProgress(100);
      setGenerateProgressLabel(`Done: ${payload.items.length}/${totalRows} row(s).`);
      updateProgressMeta({
        stage: "done",
        waitingProvider: false,
        elapsedMs: Date.now() - startedAt,
      });

      await refreshUsageSilently();
      setGenerateNotice("run", "success", `Generation completed: ${payload.items.length}/${totalRows} row(s).`);
    } catch (e: any) {
      await refreshUsageSilently();
      const detail = e?.message || String(e);
      const suffix = generatedRows > 0 ? ` Processed ${generatedRows}/${totalRows} row(s) before failure.` : "";
      setGenerateNotice("run", "error", `${detail}${suffix}`);

      if (generatedRows <= 0) {
        setGenerateProgress(0);
        setGenerateProgressLabel("");
        setGenerateProgressMeta(null);
      } else {
        const partialProgress = Math.min(
          textProgressCap,
          6 + (generatedRows / Math.max(totalRows, 1)) * Math.max(1, textProgressCap - 6)
        );
        setGenerateProgress((prev) => (partialProgress > prev ? partialProgress : prev));
        setGenerateProgressLabel(`Stopped: ${generatedRows}/${totalRows} row(s) generated.`);
        updateProgressMeta({
          stage: "text",
          done: generatedRows,
          total: totalRows,
          waitingProvider: false,
          elapsedMs: Date.now() - startedAt,
        });
      }
    } finally {
      if (progressTimer != null) {
        window.clearInterval(progressTimer);
      }
      setBusy(false);
    }
  }

  function onSaveInputText() {
    if (!inputText.trim()) {
      setGenerateNotice("input", "warning", "Input is empty. Add rows first.");
      return;
    }
    const base = normalizedDeckName(settings.defaultDeck || "dutch_words");
    const fileName = `${base}.txt`;
    const blob = new Blob([normalizeImportedText(inputText)], { type: "text/plain;charset=utf-8" });
    downloadBlobFile(blob, fileName);
    setGenerateNotice("input", "success", `Saved ${fileName}.`);
  }

  async function onExportDeck(format: ExportFormat) {
    if (!response) return;
    if (exportCards.length === 0) {
      setGenerateNotice("export", "warning", "No successful cards to export.");
      return;
    }

    const exportWarning =
      hasAudioFailures && settings.generateAudio
        ? `Audio is incomplete (${audioRunSummary?.ok || 0}/${audioRunSummary?.total || 0} clips).`
        : "";

    setExportBusy(format);
    setNotices((prev) => withGenerateNotice(prev, "export", null));

    try {
      const inlineApkgMedia =
        format === "apkg" && !canUsePersistedAudioForApkg && Object.keys(audioMediaMap).length > 0 ? audioMediaMap : undefined;
      const req: ExportDeckRequest = {
        run_id: response.run_id || generateRunId(),
        l1: settings.l1,
        cefr: settings.cefr,
        profile: settings.profile,
        model: settings.model,
        deck_name: settings.defaultDeck || "Dutch",
        guid_policy: "stable",
        include_basic_reversed: settings.includeBasicReversed,
        include_basic_typein: settings.includeBasicTypein,
        use_persisted_media: format === "apkg" ? canUsePersistedAudioForApkg : false,
        media_map: inlineApkgMedia,
        cards: exportCards,
      };

      if (format === "apkg" && req.media_map) {
        const estimatedBytes = estimateExportRequestSizeBytes(req);
        if (estimatedBytes >= EXPORT_REQUEST_SOFT_LIMIT_BYTES) {
          throw new Error(
            `APKG export request is too large for Vercel (${formatMb(estimatedBytes)} MB estimated). Server-side audio storage is not available for this run, so try fewer cards, disable audio, or regenerate after storage is fixed.`
          );
        }
      }

      const url = `${settings.apiBase || ""}/api/export/${format}`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(req) });
      if (!res.ok) {
        const data = (await res.json().catch(() => null)) as any;
        throw new Error(apiErrorText(data, res.status));
      }

      if (format === "apkg") {
        const blob = await res.blob();
        const fileName = parseAttachmentFilename(res.headers.get("Content-Disposition")) || `${normalizedDeckName(settings.defaultDeck || "Dutch")}.apkg`;
        const cardCount = Number(res.headers.get("X-Card-Count") || exportCards.length || 0);
        downloadBlobFile(blob, fileName);
        const successDetails = [exportWarning, canUsePersistedAudioForApkg ? "Used server-stored audio for APKG export." : ""]
          .filter(Boolean)
          .join(" ") || undefined;
        setGenerateNotice("export", "success", `Downloaded ${fileName} (${cardCount} cards).`, successDetails);
      } else {
        const data = (await res.json().catch(() => null)) as any;
        const payload = data as ExportFileResponse;
        const buffer = decodeBase64ToArrayBuffer(payload.content_b64);
        const blob = new Blob([buffer], { type: payload.mime_type || "application/octet-stream" });
        downloadBlobFile(blob, payload.file_name);
        const details = exportWarning || undefined;
        setGenerateNotice("export", "success", `Downloaded ${payload.file_name} (${payload.card_count} cards).`, details);
      }
    } catch (e: any) {
      setGenerateNotice("export", "error", e?.message || String(e));
    } finally {
      setExportBusy(null);
    }
  }

  function openInputFilePicker(mode: "replace" | "append") {
    fileImportModeRef.current = mode;
    fileInputRef.current?.click();
  }

  async function onInputFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length) return;

    try {
      const chunks = await Promise.all(
        files.map(async (file) => {
          const text = await file.text();
          return normalizeImportedText(text).replace(/^\n+|\n+$/g, "");
        })
      );

      const merged = chunks.filter((chunk) => chunk.length > 0).join("\n");
      if (!merged.trim()) {
        setGenerateNotice("input", "warning", "Selected file is empty.");
        return;
      }

      const mode = fileImportModeRef.current;
      setInputText((prev) => {
        if (mode === "append" && prev.trim()) {
          return `${prev.replace(/\n+$/g, "")}\n${merged.replace(/^\n+/g, "")}`;
        }
        return merged;
      });

      setResponse(null);
      setAudioMediaMap({});
      setAudioRunSummary(null);
      setGenerateProgress(0);
      setGenerateProgressLabel("");
      setGenerateProgressMeta(null);

      const shown = files
        .slice(0, 2)
        .map((file) => file.name)
        .join(", ");
      const extra = files.length > 2 ? ` +${files.length - 2} more` : "";
      clearGenerate();
      setGenerateNotice(
        "input",
        "success",
        `${mode === "append" ? "Appended" : "Loaded"} ${files.length} file(s): ${shown}${extra}.`
      );
    } catch (e: any) {
      setGenerateNotice("input", "error", e?.message || "Failed to read selected file.");
    }
  }

  return (
    <AppShell activeTab={activeTab} adminEnabled={adminEnabled} onTabChange={setActiveTab}>
      {activeTab === "generate" && (
        <GenerateTab
          settings={settings}
          parsedCount={parsed.items.length}
          warnings={warnings}
          inputText={inputText}
          onInputTextChange={setInputText}
          onSettingsPatch={patchSettings}
          onGenerate={onGenerate}
          canGenerate={canGenerate}
          busy={busy}
          onOpenFilePicker={openInputFilePicker}
          fileInputRef={fileInputRef}
          onInputFilesSelected={onInputFilesSelected}
          onSaveInputText={onSaveInputText}
          response={response}
          generateProgress={generateProgress}
          generateProgressLabel={generateProgressLabel}
          generateProgressMeta={generateProgressMeta}
          notices={notices.generate}
          onExportDeck={onExportDeck}
          exportBusy={exportBusy}
          exportCardCount={exportCards.length}
          deckPreview={deckPreview}
          temperatureState={temperatureState}
          audioClipCount={audioClipCount}
          audioRunSummary={audioRunSummary}
          hasAudioFailures={hasAudioFailures}
          onGoSettings={() => setActiveTab("settings")}
        />
      )}

      {activeTab === "settings" && (
        <SettingsTab
          settings={settings}
          onSettingsPatch={patchSettings}
          onAudioProviderChange={onAudioProviderChange}
          onLoadSettings={onLoadSettings}
          onSaveSettings={onSaveSettings}
          onRevertSettings={onRevertSettings}
          onLoadUsage={onLoadUsage}
          usage={usage}
          busy={busy}
          isDirty={isDirty}
          temperatureState={temperatureState}
          textModelOptions={textModelOptions}
          availableAudioProviders={availableAudioProviders}
          availableAudioModelOptions={availableAudioModelOptions}
          availableAudioVoiceOptions={availableAudioVoiceOptions}
          audioVoiceLabels={audioVoiceLabels}
          ttsOptionsBusy={ttsOptionsBusy}
          onReloadTtsOptions={() => onLoadTtsOptions(false)}
          onCheckElevenLabsVoiceId={onCheckElevenLabsVoiceId}
          onAddElevenLabsSharedVoice={onAddElevenLabsSharedVoice}
          onPreviewTtsVoice={onPreviewTtsVoice}
          notices={notices.settings}
          adminEnabled={adminEnabled}
        />
      )}

      {activeTab === "admin" && adminEnabled && (
        <AdminTab
          adminBusy={adminBusy}
          hasAdminKey={adminEnabled}
          users={users}
          newInvite={newInvite}
          adminUsage={adminUsage}
          onCreateInvite={() => adminCreateInvite("")}
          onListUsers={adminListUsers}
          onSetStatus={adminSetStatus}
          onRotate={adminRotate}
          onLoadUsage={adminLoadUsage}
          notices={notices.admin}
        />
      )}
    </AppShell>
  );
}
