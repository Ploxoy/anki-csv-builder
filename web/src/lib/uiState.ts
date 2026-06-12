export type TabId = "generate" | "settings" | "admin";
export type ExportFormat = "csv" | "apkg";

export type TemperatureParseResult = {
  ratio: number | null;
  percent: number | null;
  error: string | null;
  usedLegacyScale: boolean;
};

export type ProgressStage = "queued" | "text" | "audio" | "done";

export type GenerateProgressMeta = {
  stage: ProgressStage;
  done: number;
  total: number;
  batchIndex: number;
  batchTotal: number;
  elapsedMs: number;
  waitingProvider: boolean;
};

export type AudioRunSummary = {
  requested: boolean;
  total: number;
  ok: number;
  failed: number;
  errors: string[];
  persisted: boolean;
  storedClips: number;
  cachedClips: number;
  durableCachedClips: number;
  storedReusableAssets: number;
  storageError?: string | null;
  diagnostics: string[];
};

export type Settings = {
  apiBase: string;
  xApiKey: string;
  userToken: string;
  promptVersion: string;
  provider: string;
  model: string;
  cefr: string;
  profile: string;
  l1: string;
  audioProvider: string;
  audioModel: string;
  audioVoice: string;
  temperature: string;
  includeFlagged: boolean;
  generateAudio: boolean;
  includeAudioWord: boolean;
  includeAudioSentence: boolean;
  reuseTextCards: boolean;
  includeBasicReversed: boolean;
  includeBasicTypein: boolean;
  defaultDeck: string;
};

export const SETTINGS_KEY = "doedutch.settings.v1";

export const DEFAULT_SETTINGS: Settings = {
  apiBase: "",
  xApiKey: "",
  userToken: "",
  promptVersion: "p0",
  provider: "openai",
  model: "gpt-4.1-mini",
  cefr: "B1",
  profile: "balanced",
  l1: "EN",
  audioProvider: "openai",
  audioModel: "",
  audioVoice: "",
  temperature: "",
  includeFlagged: false,
  generateAudio: false,
  includeAudioWord: true,
  includeAudioSentence: true,
  reuseTextCards: false,
  includeBasicReversed: false,
  includeBasicTypein: false,
  defaultDeck: "Dutch",
};

export function settingsFingerprint(settings: Settings): string {
  return JSON.stringify(settings);
}

export function generateRunId(): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyCrypto = crypto as any;
    if (anyCrypto?.randomUUID) return anyCrypto.randomUUID();
  } catch {
    // ignore
  }
  return `u_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function shortJson(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

export function apiErrorText(data: any, status: number): string {
  const candidate = data?.detail ?? data?.error?.message ?? data;
  if (typeof candidate === "string" && candidate.trim()) return candidate;
  if (candidate != null) return shortJson(candidate);
  return `HTTP ${status}`;
}

export function formatPercent(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(Math.round(rounded)) : String(rounded);
}

export function temperatureToDisplayString(raw: number | null | undefined): string {
  if (raw == null || !Number.isFinite(raw)) return "";
  const normalized = raw <= 1 ? raw * 100 : raw;
  return formatPercent(normalized);
}

export function parseTemperatureValue(raw: string): TemperatureParseResult {
  const trimmed = raw.trim();
  if (!trimmed) return { ratio: null, percent: null, error: null, usedLegacyScale: false };
  const value = Number(trimmed);
  if (!Number.isFinite(value)) {
    return {
      ratio: null,
      percent: null,
      error: "Temperature must be a number between 0 and 100.",
      usedLegacyScale: false,
    };
  }

  const looksLegacyDecimal = value > 0 && value < 1 && trimmed.includes(".");
  if (looksLegacyDecimal) {
    return {
      ratio: value,
      percent: value * 100,
      error: null,
      usedLegacyScale: true,
    };
  }

  if (value < 0 || value > 100) {
    return {
      ratio: null,
      percent: null,
      error: "Temperature must be in range 0..100.",
      usedLegacyScale: false,
    };
  }

  return {
    ratio: value / 100,
    percent: value,
    error: null,
    usedLegacyScale: false,
  };
}

export function formatElapsedMs(ms: number): string {
  const safe = Math.max(0, Math.floor(ms));
  if (safe < 1000) return `${safe} ms`;
  const totalSeconds = Math.floor(safe / 1000);
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  if (mins <= 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
}

export function appendUniqueError(errors: string[], candidate: unknown): void {
  const text = typeof candidate === "string" ? candidate.trim() : "";
  if (!text) return;
  if (!errors.includes(text)) errors.push(text);
}

export function progressStageLabel(stage: ProgressStage): string {
  if (stage === "queued") return "Queued";
  if (stage === "text") return "Text generation";
  if (stage === "audio") return "Audio synthesis";
  return "Completed";
}

export function normalizeImportedText(raw: string): string {
  return raw.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
}

export function decodeBase64ToArrayBuffer(contentB64: string): ArrayBuffer {
  const binary = atob(contentB64);
  const buffer = new ArrayBuffer(binary.length);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return buffer;
}

export function downloadBlobFile(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function parseAttachmentFilename(headerValue: string | null): string | null {
  if (!headerValue) return null;
  const utf8Match = headerValue.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return utf8Match[1];
    }
  }
  const plainMatch = headerValue.match(/filename="?([^";]+)"?/i);
  return plainMatch?.[1]?.trim() || null;
}

export function normalizedDeckName(rawName: string): string {
  const safe = rawName
    .trim()
    .replace(/[^\w.-]+/g, "_")
    .replace(/^[_\-.]+|[_\-.]+$/g, "");
  return safe || "deck";
}
