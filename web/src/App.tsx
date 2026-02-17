import { useEffect, useMemo, useState } from "react";
import { parseItems } from "./lib/parse";
import { loadJson, saveJson } from "./lib/storage";
import {
  GenerateRequest,
  GenerateResponse,
  UsageListResponse,
  UserSettingsResponse,
  UserSettingsUpsertRequest,
  UserListResponse,
  InviteCreateResponse
} from "./types";

type TabId = "generate" | "settings" | "admin";
type ResultFilter = "all" | "errors" | "repaired";
type TemperatureParseResult = {
  ratio: number | null;
  percent: number | null;
  error: string | null;
  usedLegacyScale: boolean;
};

type Settings = {
  apiBase: string; // keep for prod; in dev we can use Vite proxy with empty string
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
  includeBasicReversed: boolean;
  includeBasicTypein: boolean;
  defaultDeck: string;
};

const SETTINGS_KEY = "doedutch.settings.v1";

const DEFAULT_SETTINGS: Settings = {
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
  includeBasicReversed: false,
  includeBasicTypein: false,
  defaultDeck: "Dutch"
};

function generateRunId(): string {
  try {
    // Modern browsers (Chrome/Edge/Firefox) support this.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyCrypto = crypto as any;
    if (anyCrypto?.randomUUID) return anyCrypto.randomUUID();
  } catch {
    // ignore
  }
  return `u_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function shortJson(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function formatPercent(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(Math.round(rounded)) : String(rounded);
}

function temperatureToDisplayString(raw: number | null | undefined): string {
  if (raw == null || !Number.isFinite(raw)) return "";
  const normalized = raw <= 1 ? raw * 100 : raw;
  return formatPercent(normalized);
}

function parseTemperatureValue(raw: string): TemperatureParseResult {
  const trimmed = raw.trim();
  if (!trimmed) return { ratio: null, percent: null, error: null, usedLegacyScale: false };
  const value = Number(trimmed);
  if (!Number.isFinite(value)) {
    return { ratio: null, percent: null, error: "Temperature must be a number between 0 and 100.", usedLegacyScale: false };
  }

  // Backward compatibility: old UI often stored decimal 0..1.
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
    return { ratio: null, percent: null, error: "Temperature must be in range 0..100.", usedLegacyScale: false };
  }

  return {
    ratio: value / 100,
    percent: value,
    error: null,
    usedLegacyScale: false,
  };
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(() => ({
    ...DEFAULT_SETTINGS,
    ...loadJson(SETTINGS_KEY, DEFAULT_SETTINGS)
  }));
  const [inputText, setInputText] = useState<string>(
    "aanraken\tiets voelen\tto touch\nbegrijpen\tsnappen wat iets betekent\tto understand"
  );
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [response, setResponse] = useState<GenerateResponse | null>(null);
  const [serverMsg, setServerMsg] = useState<string>("");
  const [usage, setUsage] = useState<UsageListResponse | null>(null);
  const [users, setUsers] = useState<UserListResponse | null>(null);
  const [newInvite, setNewInvite] = useState<InviteCreateResponse | null>(null);
  const [adminUsage, setAdminUsage] = useState<UsageListResponse | null>(null);
  const [adminBusy, setAdminBusy] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("generate");
  const [showUserToken, setShowUserToken] = useState(false);
  const [showXApiKey, setShowXApiKey] = useState(false);
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all");

  const deckPreview = useMemo(() => {
    const base = settings.defaultDeck?.trim() || "Deck";
    return `${base}.csv / ${base}.apkg`;
  }, [settings.defaultDeck]);
  const parsed = useMemo(() => parseItems(inputText), [inputText]);
  const adminEnabled = settings.xApiKey.trim().length > 0;
  const canGenerateByInput = parsed.items.length > 0;
  const temperatureState = useMemo(() => parseTemperatureValue(settings.temperature), [settings.temperature]);
  const canGenerate = canGenerateByInput && !temperatureState.error;

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

  const filteredItems = useMemo(() => {
    const items = response?.items || [];
    if (resultFilter === "all") return items;
    if (resultFilter === "repaired") return items.filter((it) => it.status === "repaired");
    return items.filter((it) => it.status === "failed" || it.status === "flagged" || !!it.error);
  }, [response, resultFilter]);

  function apiHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (settings.userToken.trim()) headers["Authorization"] = `Bearer ${settings.userToken.trim()}`;
    if (settings.xApiKey.trim()) headers["X-API-Key"] = settings.xApiKey.trim(); // admin/legacy only
    return headers;
  }

  async function onLoadSettings() {
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + "/api/settings";
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        const detail = data?.detail || data?.error?.message || shortJson(data) || `HTTP ${res.status}`;
        throw new Error(detail);
      }
      const payload = data as UserSettingsResponse;
      setSettings((s) => ({
        ...s,
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
        includeBasicReversed: payload.settings.include_basic_reversed,
        includeBasicTypein: payload.settings.include_basic_typein,
        defaultDeck: payload.settings.default_deck_name
      }));
      setServerMsg(`Loaded settings for ${payload.user_id} (${payload.updated_at || "no timestamp"}).`);
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }

  async function onSaveSettings() {
    setServerMsg("");
    setError("");
    const tempParsed = parseTemperatureValue(settings.temperature);
    if (tempParsed.error) {
      setError(tempParsed.error);
      return;
    }
    try {
      const url = (settings.apiBase || "") + "/api/settings";
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
          include_basic_reversed: settings.includeBasicReversed,
          include_basic_typein: settings.includeBasicTypein,
          default_deck_name: settings.defaultDeck
        }
      };
      const res = await fetch(url, { method: "PUT", headers, body: JSON.stringify(body) });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        const detail = data?.detail || data?.error?.message || shortJson(data) || `HTTP ${res.status}`;
        throw new Error(detail);
      }
      const payload = data as UserSettingsResponse;
      setServerMsg(`Saved settings for ${payload.user_id} (${payload.updated_at || "no timestamp"}).`);
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }

  async function onLoadUsage() {
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + "/api/usage?limit=50";
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        const detail = data?.detail || data?.error?.message || shortJson(data) || `HTTP ${res.status}`;
        throw new Error(detail);
      }
      setUsage(data as UsageListResponse);
      setServerMsg("Loaded usage events.");
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }

  async function adminCreateInvite(label: string) {
    setAdminBusy(true);
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + "/api/admin/invite";
      const headers = apiHeaders();
      const res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify({ label })
      });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(data?.detail || shortJson(data) || `HTTP ${res.status}`);
      setNewInvite(data as InviteCreateResponse);
      setServerMsg("Invite created.");
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminListUsers() {
    setAdminBusy(true);
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + "/api/admin/users";
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(data?.detail || shortJson(data) || `HTTP ${res.status}`);
      setUsers(data as UserListResponse);
      setServerMsg("Loaded users.");
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminSetStatus(userId: string, status: "active" | "blocked") {
    setAdminBusy(true);
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + `/api/admin/users/${userId}/status`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "POST", headers, body: JSON.stringify({ status }) });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(data?.detail || shortJson(data) || `HTTP ${res.status}`);
      setServerMsg(`User ${userId} -> ${status}`);
      await adminListUsers();
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminRotate(userId: string) {
    setAdminBusy(true);
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + `/api/admin/users/${userId}/rotate`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "POST", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(data?.detail || shortJson(data) || `HTTP ${res.status}`);
      setNewInvite(data as InviteCreateResponse);
      setServerMsg(`Rotated token for ${userId}`);
      await adminListUsers();
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function adminLoadUsage(userId: string) {
    setAdminBusy(true);
    setServerMsg("");
    setError("");
    try {
      const url = (settings.apiBase || "") + `/api/admin/usage?user_id=${encodeURIComponent(userId)}&limit=100`;
      const headers = apiHeaders();
      const res = await fetch(url, { method: "GET", headers });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) throw new Error(data?.detail || shortJson(data) || `HTTP ${res.status}`);
      setAdminUsage(data as UsageListResponse);
      setServerMsg(`Loaded usage for ${userId}`);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setAdminBusy(false);
    }
  }

  async function onGenerate() {
    const tempParsed = parseTemperatureValue(settings.temperature);
    if (tempParsed.error) {
      setError(tempParsed.error);
      return;
    }
    setBusy(true);
    setError("");
    setResponse(null);
    setServerMsg("");
    try {
      const runId = generateRunId();
      const req: GenerateRequest = {
        run_id: runId,
        prompt_version: settings.promptVersion,
        provider: settings.provider,
        model: settings.model,
        cefr: settings.cefr,
        profile: settings.profile,
        l1: settings.l1,
        temperature: tempParsed.ratio,
        items: parsed.items
      };

      const url = (settings.apiBase || "") + "/api/generate";
      const headers = apiHeaders();

      const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(req) });
      const data = (await res.json().catch(() => null)) as any;
      if (!res.ok) {
        const detail = data?.detail || data?.error?.message || shortJson(data) || `HTTP ${res.status}`;
        throw new Error(detail);
      }
      setResponse(data as GenerateResponse);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <div className="title">Doedutch</div>
          <div className="subtitle">Minimal UI (React + Vite)</div>
        </div>
        {activeTab === "generate" && (
          <div className="actions">
            <button className="btn primary" onClick={onGenerate} disabled={busy || !canGenerate}>
              {busy ? "Generating..." : `Generate (${parsed.items.length})`}
            </button>
          </div>
        )}
      </header>

      <div className="tabs">
        <button className={`tab ${activeTab === "generate" ? "active" : ""}`} onClick={() => setActiveTab("generate")}>
          Generate
        </button>
        <button className={`tab ${activeTab === "settings" ? "active" : ""}`} onClick={() => setActiveTab("settings")}>
          Settings
        </button>
        {adminEnabled && (
          <button className={`tab ${activeTab === "admin" ? "active" : ""}`} onClick={() => setActiveTab("admin")}>
            Admin
          </button>
        )}
      </div>

      {serverMsg && <p className="hint info-banner">{serverMsg}</p>}

      {activeTab === "generate" && (
        <>
          <section className="card">
            <h2>Quick setup</h2>
            <div className="chips">
              <span className="chip">Model: {settings.model}</span>
              <span className={`chip ${temperatureState.error ? "chip-danger" : ""}`}>
                Temp: {temperatureState.percent != null ? `${formatPercent(temperatureState.percent)}%` : "default"}
              </span>
              <span className="chip">CEFR: {settings.cefr}</span>
              <span className="chip">Profile: {settings.profile}</span>
              <span className="chip">Audio: {settings.audioProvider}</span>
              <span className="chip">Deck: {settings.defaultDeck || "—"}</span>
            </div>
            <div className="quick-toggles">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.generateAudio}
                  onChange={(e) => setSettings((s) => ({ ...s, generateAudio: e.target.checked }))}
                />
                <span>Generate audio</span>
              </label>
              <label className={`toggle ${!settings.generateAudio ? "disabled" : ""}`}>
                <input
                  type="checkbox"
                  checked={settings.includeAudioWord}
                  disabled={!settings.generateAudio}
                  onChange={(e) => setSettings((s) => ({ ...s, includeAudioWord: e.target.checked }))}
                />
                <span>Audio: word</span>
              </label>
              <label className={`toggle ${!settings.generateAudio ? "disabled" : ""}`}>
                <input
                  type="checkbox"
                  checked={settings.includeAudioSentence}
                  disabled={!settings.generateAudio}
                  onChange={(e) => setSettings((s) => ({ ...s, includeAudioSentence: e.target.checked }))}
                />
                <span>Audio: sentence</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.includeBasicReversed}
                  onChange={(e) => setSettings((s) => ({ ...s, includeBasicReversed: e.target.checked }))}
                />
                <span>Basic (reversed)</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.includeBasicTypein}
                  onChange={(e) => setSettings((s) => ({ ...s, includeBasicTypein: e.target.checked }))}
                />
                <span>Basic (type-in)</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings.includeFlagged}
                  onChange={(e) => setSettings((s) => ({ ...s, includeFlagged: e.target.checked }))}
                />
                <span>Force flagged</span>
              </label>
            </div>
            {!settings.generateAudio && (
              <p className="hint subtle">Enable “Generate audio” to use word/sentence audio options.</p>
            )}
            <div className="meta">
              <div>
                <span className="k">Deck preview</span> {deckPreview}
              </div>
              <div>
                <button className="btn" onClick={() => setActiveTab("settings")}>Edit in Settings</button>
              </div>
              <div>
                <button className="btn" onClick={onLoadUsage} disabled={busy}>
                  Load usage
                </button>
              </div>
            </div>
          </section>

          <section className="card">
            <h2>Input</h2>
            <p className="hint flow-hint">
              1. Paste your rows. 2. Click Generate. 3. Review errors/repaired items below.
            </p>
            <div className="input-toolbar">
              <div className="meta">
                <div>
                  <span className="k">rows parsed</span> {parsed.items.length}
                </div>
              </div>
              <button className="btn primary" onClick={onGenerate} disabled={busy || !canGenerate}>
                {busy ? "Generating..." : `Generate (${parsed.items.length})`}
              </button>
            </div>
            <textarea value={inputText} onChange={(e) => setInputText(e.target.value)} rows={6} />
            {warnings.length > 0 && (
              <div className="warnings">
                {warnings.map((w) => (
                  <div key={w} className="warning">
                    {w}
                  </div>
                ))}
              </div>
            )}
            {temperatureState.error && <div className="warning">{temperatureState.error}</div>}
            <p className="hint">
              Accepted formats per line: TSV, `woord ;; def ;; translation`, `woord — def — translation`.
            </p>
          </section>

          <section className="card">
            <h2>Result</h2>
            {response && (
              <div className="result-filters">
                <button className={`btn small ${resultFilter === "all" ? "active" : ""}`} onClick={() => setResultFilter("all")}>
                  All
                </button>
                <button className={`btn small ${resultFilter === "errors" ? "active" : ""}`} onClick={() => setResultFilter("errors")}>
                  Errors
                </button>
                <button className={`btn small ${resultFilter === "repaired" ? "active" : ""}`} onClick={() => setResultFilter("repaired")}>
                  Repaired
                </button>
              </div>
            )}
            {error && <div className="error">{error}</div>}
            {!error && !response && (
              <div className="empty">
                No generation yet. Add at least one row in Input, then click <strong>Generate</strong>.
              </div>
            )}
            {response && (
              <div className="result">
                <div className="meta">
                  <div>
                    <span className="k">elapsed</span> {response.timing?.elapsed_ms} ms
                  </div>
                  <div>
                    <span className="k">model</span> {response.model}
                  </div>
                  <div>
                    <span className="k">items</span> {response.items.length}
                  </div>
                  <div>
                    <span className="k">shown</span> {filteredItems.length}
                  </div>
                </div>
                <div className="table">
                  {filteredItems.length === 0 && <div className="empty">No rows match the selected filter.</div>}
                  {filteredItems.map((it) => (
                    <div key={it.id} className="row">
                      <div className={"status " + it.status}>{it.status}</div>
                      <div className="id">#{it.id}</div>
                      <div className="content">
                        {it.error && <div className="errline">{it.error}</div>}
                        {it.card && (
                          <details>
                            <summary>{it.card.L2_word}</summary>
                            <pre className="pre">{shortJson(it.card)}</pre>
                          </details>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                <details className="raw">
                  <summary>Raw response</summary>
                  <pre className="pre">{shortJson(response)}</pre>
                </details>
              </div>
            )}
          </section>

          {usage && (
            <section className="card">
              <h2>Usage (latest)</h2>
              <div className="meta">
                <div>
                  <span className="k">events</span> {usage.summary.events}
                </div>
                <div>
                  <span className="k">tokens</span> {usage.summary.input_tokens}+{usage.summary.output_tokens} (cached{" "}
                  {usage.summary.cached_tokens})
                </div>
                <div>
                  <span className="k">audio chars</span> {usage.summary.audio_chars}
                </div>
                {usage.summary.raw_cost_usd != null && (
                  <div>
                    <span className="k">est USD</span> {usage.summary.raw_cost_usd}
                  </div>
                )}
              </div>
              <details className="raw">
                <summary>Raw usage</summary>
                <pre className="pre">{shortJson(usage)}</pre>
              </details>
            </section>
          )}
        </>
      )}

      {activeTab === "settings" && (
        <section className="card">
          <h2>Settings</h2>
          <div className="settings-section">
            <h3>Access</h3>
            <div className="grid">
              <label>
                <span>Invite token (beta)</span>
                <div className="secret-input">
                  <input
                    type={showUserToken ? "text" : "password"}
                    value={settings.userToken}
                    onChange={(e) => setSettings((s) => ({ ...s, userToken: e.target.value }))}
                    placeholder="paste token from /api/admin/invite"
                  />
                  <button type="button" className="btn tiny" onClick={() => setShowUserToken((v) => !v)}>
                    {showUserToken ? "Hide" : "Show"}
                  </button>
                </div>
              </label>
            </div>
            <details className="advanced">
              <summary>Advanced access (dev/admin)</summary>
              <div className="grid">
                <label>
                  <span>API base</span>
                  <input
                    value={settings.apiBase}
                    onChange={(e) => setSettings((s) => ({ ...s, apiBase: e.target.value }))}
                    placeholder="(empty = use Vite proxy)"
                  />
                </label>
                <label>
                  <span>X-API-Key (admin only)</span>
                  <div className="secret-input">
                    <input
                      type={showXApiKey ? "text" : "password"}
                      value={settings.xApiKey}
                      onChange={(e) => setSettings((s) => ({ ...s, xApiKey: e.target.value }))}
                      placeholder="API_SHARED_SECRET"
                    />
                    <button type="button" className="btn tiny" onClick={() => setShowXApiKey((v) => !v)}>
                      {showXApiKey ? "Hide" : "Show"}
                    </button>
                  </div>
                </label>
              </div>
            </details>
            {!adminEnabled && <p className="hint subtle">Add X-API-Key in Advanced access to unlock the Admin tab.</p>}
          </div>

          <div className="settings-triad">
            <div className="settings-section settings-block">
              <h3>Generation</h3>
              <div className="grid">
                <label>
                  <span>Model</span>
                  <input value={settings.model} onChange={(e) => setSettings((s) => ({ ...s, model: e.target.value }))} />
                </label>
                <label>
                  <span>Temperature (%) 0..100</span>
                  <input
                    type="number"
                    step="1"
                    min="0"
                    max="100"
                    value={settings.temperature}
                    onChange={(e) => setSettings((s) => ({ ...s, temperature: e.target.value }))}
                    placeholder="empty = model default"
                  />
                </label>
                <label>
                  <span>CEFR</span>
                  <select value={settings.cefr} onChange={(e) => setSettings((s) => ({ ...s, cefr: e.target.value }))}>
                    <option value="A1">A1</option>
                    <option value="A2">A2</option>
                    <option value="B1">B1</option>
                    <option value="B2">B2</option>
                    <option value="C1">C1</option>
                    <option value="C2">C2</option>
                  </select>
                </label>
                <label>
                  <span>Profile</span>
                  <select value={settings.profile} onChange={(e) => setSettings((s) => ({ ...s, profile: e.target.value }))}>
                    <option value="strict">strict</option>
                    <option value="balanced">balanced</option>
                    <option value="exam">exam</option>
                    <option value="creative">creative</option>
                  </select>
                </label>
                <label>
                  <span>L1</span>
                  <select value={settings.l1} onChange={(e) => setSettings((s) => ({ ...s, l1: e.target.value }))}>
                    <option value="EN">EN</option>
                    <option value="RU">RU</option>
                    <option value="ES">ES</option>
                    <option value="DE">DE</option>
                  </select>
                </label>
              </div>
              <label className="checkbox-control">
                <input
                  type="checkbox"
                  checked={settings.includeFlagged}
                  onChange={(e) => setSettings((s) => ({ ...s, includeFlagged: e.target.checked }))}
                />
                <span>Force generate flagged rows</span>
              </label>
              {temperatureState.error && <div className="warning">{temperatureState.error}</div>}
              {temperatureState.usedLegacyScale && (
                <p className="hint subtle">Legacy decimal detected and auto-converted (e.g. 0.8 → 80%).</p>
              )}
            </div>

            <div className="settings-section settings-block">
              <h3>Audio</h3>
              <div className="checkbox-group">
                <label className="checkbox-control">
                  <input
                    type="checkbox"
                    checked={settings.generateAudio}
                    onChange={(e) => setSettings((s) => ({ ...s, generateAudio: e.target.checked }))}
                  />
                  <span>Generate audio</span>
                </label>
                <label className={`checkbox-control ${!settings.generateAudio ? "disabled-control" : ""}`}>
                  <input
                    type="checkbox"
                    checked={settings.includeAudioWord}
                    disabled={!settings.generateAudio}
                    onChange={(e) => setSettings((s) => ({ ...s, includeAudioWord: e.target.checked }))}
                  />
                  <span>Include audio: word</span>
                </label>
                <label className={`checkbox-control ${!settings.generateAudio ? "disabled-control" : ""}`}>
                  <input
                    type="checkbox"
                    checked={settings.includeAudioSentence}
                    disabled={!settings.generateAudio}
                    onChange={(e) => setSettings((s) => ({ ...s, includeAudioSentence: e.target.checked }))}
                  />
                  <span>Include audio: sentence</span>
                </label>
              </div>
              <div className="grid">
                <label>
                  <span>TTS provider</span>
                  <input
                    value={settings.audioProvider}
                    onChange={(e) => setSettings((s) => ({ ...s, audioProvider: e.target.value }))}
                    placeholder="openai|elevenlabs"
                  />
                </label>
                <label>
                  <span>TTS model</span>
                  <input
                    value={settings.audioModel}
                    onChange={(e) => setSettings((s) => ({ ...s, audioModel: e.target.value }))}
                    placeholder="gpt-4o-mini-tts-... / elevenlabs id"
                  />
                </label>
                <label>
                  <span>TTS voice</span>
                  <input
                    value={settings.audioVoice}
                    onChange={(e) => setSettings((s) => ({ ...s, audioVoice: e.target.value }))}
                    placeholder="alloy / elevenlabs voice id"
                  />
                </label>
              </div>
              {!settings.generateAudio && (
                <p className="hint subtle">Audio word/sentence options are disabled until “Generate audio” is enabled.</p>
              )}
            </div>

            <div className="settings-section settings-block">
              <h3>Export</h3>
              <div className="checkbox-group">
                <label className="checkbox-control">
                  <input
                    type="checkbox"
                    checked={settings.includeBasicReversed}
                    onChange={(e) => setSettings((s) => ({ ...s, includeBasicReversed: e.target.checked }))}
                  />
                  <span>Include basic (reversed)</span>
                </label>
                <label className="checkbox-control">
                  <input
                    type="checkbox"
                    checked={settings.includeBasicTypein}
                    onChange={(e) => setSettings((s) => ({ ...s, includeBasicTypein: e.target.checked }))}
                  />
                  <span>Include basic (type-in)</span>
                </label>
              </div>
              <div className="grid">
                <label>
                  <span>Default deck name (CSV/APKG filename)</span>
                  <input
                    value={settings.defaultDeck}
                    onChange={(e) => setSettings((s) => ({ ...s, defaultDeck: e.target.value }))}
                    placeholder="e.g., Dutch"
                  />
                </label>
              </div>
            </div>
          </div>
          <div className="actions">
            <button className="btn" onClick={onLoadSettings} disabled={busy}>
              Load settings
            </button>
            <button className="btn" onClick={onSaveSettings} disabled={busy}>
              Save settings
            </button>
            <button className="btn" onClick={onLoadUsage} disabled={busy}>
              Load usage
            </button>
          </div>
          <p className="hint">
            Leave API base empty and run the dev server with proxy to avoid CORS. For beta, use an invite token.
          </p>
        </section>
      )}

      {activeTab === "admin" && (
        <>
          <section className="card">
            <h2>Admin (invite + users)</h2>
            <div className="actions">
              <button className="btn" onClick={() => adminCreateInvite("")} disabled={adminBusy || !settings.xApiKey}>
                Create invite
              </button>
              <button className="btn" onClick={adminListUsers} disabled={adminBusy || !settings.xApiKey}>
                List users
              </button>
            </div>
            {newInvite && (
              <div className="meta">
                <div>
                  <span className="k">invite user_id</span> {newInvite.user_id}
                </div>
                <div>
                  <span className="k">token</span> {newInvite.token}
                </div>
              </div>
            )}
            {users && users.items.length > 0 && (
              <div className="table">
                {users.items.map((u) => (
                  <div key={u.id} className="row">
                    <div className={"status " + (u.status === "active" ? "ok" : "failed")}>{u.status}</div>
                    <div className="id">{u.id}</div>
                    <div className="content">
                      <div className="meta">
                        {u.label && (
                          <div>
                            <span className="k">label</span> {u.label}
                          </div>
                        )}
                        <div>
                          <span className="k">created</span> {u.created_at}
                        </div>
                        {u.last_used_at && (
                          <div>
                            <span className="k">last</span> {u.last_used_at}
                          </div>
                        )}
                      </div>
                      <div className="actions">
                        <button className="btn" onClick={() => adminSetStatus(u.id, u.status === "active" ? "blocked" : "active")} disabled={adminBusy || !settings.xApiKey}>
                          {u.status === "active" ? "Block" : "Unblock"}
                        </button>
                        <button className="btn" onClick={() => adminRotate(u.id)} disabled={adminBusy || !settings.xApiKey}>
                          Rotate token
                        </button>
                        <button className="btn" onClick={() => adminLoadUsage(u.id)} disabled={adminBusy || !settings.xApiKey}>
                          Usage
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!users && <p className="hint">Admin actions need X-API-Key. Use invite token for normal calls.</p>}
          </section>

          {adminUsage && (
            <section className="card">
              <h2>Admin: usage for {adminUsage.user_id}</h2>
              <div className="meta">
                <div>
                  <span className="k">events</span> {adminUsage.summary.events}
                </div>
                <div>
                  <span className="k">tokens</span> {adminUsage.summary.input_tokens}+{adminUsage.summary.output_tokens} (cached{" "}
                  {adminUsage.summary.cached_tokens})
                </div>
                <div>
                  <span className="k">audio chars</span> {adminUsage.summary.audio_chars}
                </div>
                {adminUsage.summary.raw_cost_usd != null && (
                  <div>
                    <span className="k">est USD</span> {adminUsage.summary.raw_cost_usd}
                  </div>
                )}
              </div>
              <details className="raw">
                <summary>Raw usage</summary>
                <pre className="pre">{shortJson(adminUsage)}</pre>
              </details>
            </section>
          )}
        </>
      )}
    </div>
  );
}
