import { useState } from "react";
import { UsageListResponse } from "../../types";
import { NoticeMessage } from "../../lib/messages";
import { Settings, TemperatureParseResult, shortJson } from "../../lib/uiState";
import { Notice } from "../../ui/Notice";

type SettingsTabProps = {
  settings: Settings;
  onSettingsPatch: (patch: Partial<Settings>) => void;
  onAudioProviderChange: (providerId: string) => void;
  onLoadSettings: () => void;
  onSaveSettings: () => void;
  onRevertSettings: () => void;
  onLoadUsage: () => void;
  usage: UsageListResponse | null;
  busy: boolean;
  isDirty: boolean;
  temperatureState: TemperatureParseResult;
  textModelOptions: string[];
  availableAudioProviders: string[];
  availableAudioModelOptions: string[];
  availableAudioVoiceOptions: string[];
  audioVoiceLabels: Record<string, string>;
  ttsOptionsBusy: boolean;
  onReloadTtsOptions: () => void;
  onCheckElevenLabsVoiceId: (voiceId: string) => void;
  onPreviewTtsVoice: (sampleText: string) => Promise<string>;
  notices: {
    toolbar: NoticeMessage | null;
    access: NoticeMessage | null;
    generation: NoticeMessage | null;
    audio: NoticeMessage | null;
    export: NoticeMessage | null;
  };
  adminEnabled: boolean;
};

export function SettingsTab({
  settings,
  onSettingsPatch,
  onAudioProviderChange,
  onLoadSettings,
  onSaveSettings,
  onRevertSettings,
  onLoadUsage,
  usage,
  busy,
  isDirty,
  temperatureState,
  textModelOptions,
  availableAudioProviders,
  availableAudioModelOptions,
  availableAudioVoiceOptions,
  audioVoiceLabels,
  ttsOptionsBusy,
  onReloadTtsOptions,
  onCheckElevenLabsVoiceId,
  onPreviewTtsVoice,
  notices,
  adminEnabled,
}: SettingsTabProps) {
  const [showUserToken, setShowUserToken] = useState(false);
  const [showXApiKey, setShowXApiKey] = useState(false);
  const [customVoiceId, setCustomVoiceId] = useState("");
  const [previewText, setPreviewText] = useState("Dit is een voorbeeld van deze stem.");
  const [previewAudioUrl, setPreviewAudioUrl] = useState("");
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const audioProviderKey = (settings.audioProvider || "").trim().toLowerCase();

  async function runVoicePreview() {
    setPreviewBusy(true);
    setPreviewError("");
    setPreviewAudioUrl("");
    try {
      const audioUrl = await onPreviewTtsVoice(previewText);
      setPreviewAudioUrl(audioUrl);
    } catch (e: any) {
      setPreviewError(e?.message || String(e));
    } finally {
      setPreviewBusy(false);
    }
  }

  return (
    <section className="tab-layout">
      <section className="card">
        <div className="section-head">
          <h2>Settings</h2>
          <span className={`dirty-pill ${isDirty ? "dirty" : "clean"}`}>{isDirty ? "Unsaved changes" : "Saved"}</span>
        </div>

        <div className="toolbar row-wrap">
          <div className="toolbar-actions">
            <button className="btn primary" type="button" onClick={onSaveSettings} disabled={busy || !isDirty || !!temperatureState.error}>
              Save
            </button>
            <button className="btn" type="button" onClick={onRevertSettings} disabled={busy || !isDirty}>
              Revert
            </button>
            <button className="btn" type="button" onClick={onLoadSettings} disabled={busy}>
              Reload
            </button>
            <button className="btn" type="button" onClick={onLoadUsage} disabled={busy}>
              Load usage
            </button>
          </div>
        </div>

        <Notice notice={notices.toolbar} />
      </section>

      <section className="card section-card">
        <div className="section-head">
          <h3>Access</h3>
          <span className="muted">Critical</span>
        </div>

        <label>
          <span>Invite token (beta)</span>
          <div className="inline-actions">
            <input
              type={showUserToken ? "text" : "password"}
              value={settings.userToken}
              onChange={(e) => onSettingsPatch({ userToken: e.target.value })}
              placeholder="paste token from /api/admin/invite"
            />
            <button className="btn tiny" type="button" onClick={() => setShowUserToken((v) => !v)}>
              {showUserToken ? "Hide" : "Show"}
            </button>
          </div>
        </label>

        <details className="advanced-panel">
          <summary>Advanced access (dev/admin)</summary>
          <div className="grid two-col">
            <label>
              <span>API base</span>
              <input
                value={settings.apiBase}
                onChange={(e) => onSettingsPatch({ apiBase: e.target.value })}
                placeholder="empty = use Vite proxy"
              />
            </label>

            <label>
              <span>X-API-Key (admin only)</span>
              <div className="inline-actions">
                <input
                  type={showXApiKey ? "text" : "password"}
                  value={settings.xApiKey}
                  onChange={(e) => onSettingsPatch({ xApiKey: e.target.value })}
                  placeholder="API_SHARED_SECRET"
                />
                <button className="btn tiny" type="button" onClick={() => setShowXApiKey((v) => !v)}>
                  {showXApiKey ? "Hide" : "Show"}
                </button>
              </div>
            </label>
          </div>
        </details>

        {!adminEnabled && <p className="hint subtle">Add X-API-Key in Advanced access to unlock Admin tab.</p>}
        <Notice notice={notices.access} />
      </section>

      <section className="card section-card">
        <div className="section-head">
          <h3>Generation defaults</h3>
          <button className="btn tiny" type="button" onClick={onReloadTtsOptions} disabled={ttsOptionsBusy || busy}>
            {ttsOptionsBusy ? "Loading..." : "Reload models & voices"}
          </button>
        </div>

        <div className="grid two-col">
          <label>
            <span>Model</span>
            <select value={settings.model} onChange={(e) => onSettingsPatch({ model: e.target.value })}>
              {textModelOptions.map((modelId) => (
                <option key={modelId} value={modelId}>
                  {modelId}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Temperature (%) 0..100</span>
            <input
              type="number"
              min="0"
              max="100"
              step="1"
              value={settings.temperature}
              onChange={(e) => onSettingsPatch({ temperature: e.target.value })}
              placeholder="empty = model default"
            />
          </label>

          <label>
            <span>CEFR</span>
            <select value={settings.cefr} onChange={(e) => onSettingsPatch({ cefr: e.target.value })}>
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
            <select value={settings.profile} onChange={(e) => onSettingsPatch({ profile: e.target.value })}>
              <option value="strict">strict</option>
              <option value="balanced">balanced</option>
              <option value="exam">exam</option>
              <option value="creative">creative</option>
            </select>
          </label>

          <label>
            <span>L1</span>
            <select value={settings.l1} onChange={(e) => onSettingsPatch({ l1: e.target.value })}>
              <option value="EN">EN</option>
              <option value="RU">RU</option>
              <option value="ES">ES</option>
              <option value="DE">DE</option>
            </select>
          </label>
        </div>

        <label className="check-tile single">
          <input
            type="checkbox"
            checked={settings.includeFlagged}
            onChange={(e) => onSettingsPatch({ includeFlagged: e.target.checked })}
          />
          <span>Force generate flagged rows</span>
        </label>

        <label className="check-tile single">
          <input
            type="checkbox"
            checked={settings.reuseTextCards}
            onChange={(e) => onSettingsPatch({ reuseTextCards: e.target.checked })}
          />
          <span>Reuse saved cards when input and generation settings match</span>
        </label>

        {temperatureState.error && <Notice notice={{ level: "error", message: temperatureState.error }} />}
        {temperatureState.usedLegacyScale && (
          <Notice
            notice={{ level: "info", message: "Legacy decimal temperature detected and converted (e.g. 0.8 -> 80%)." }}
          />
        )}
        <Notice notice={notices.generation} />
      </section>

      <section className="card section-card">
        <div className="section-head">
          <h3>Audio defaults</h3>
          <span className="muted">Optional</span>
        </div>

        <div className="toggle-grid">
          <label className="check-tile">
            <input
              type="checkbox"
              checked={settings.generateAudio}
              onChange={(e) => onSettingsPatch({ generateAudio: e.target.checked })}
            />
            <span>Generate audio</span>
          </label>
          <label className={`check-tile ${!settings.generateAudio ? "disabled" : ""}`}>
            <input
              type="checkbox"
              checked={settings.includeAudioWord}
              disabled={!settings.generateAudio}
              onChange={(e) => onSettingsPatch({ includeAudioWord: e.target.checked })}
            />
            <span>Include audio: word</span>
          </label>
          <label className={`check-tile ${!settings.generateAudio ? "disabled" : ""}`}>
            <input
              type="checkbox"
              checked={settings.includeAudioSentence}
              disabled={!settings.generateAudio}
              onChange={(e) => onSettingsPatch({ includeAudioSentence: e.target.checked })}
            />
            <span>Include audio: sentence</span>
          </label>
        </div>

        <div className="grid two-col">
          <label>
            <span>TTS provider</span>
            <select value={(settings.audioProvider || "").trim().toLowerCase()} onChange={(e) => onAudioProviderChange(e.target.value)}>
              {availableAudioProviders.map((providerId) => (
                <option key={providerId} value={providerId}>
                  {providerId}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>TTS model</span>
            <select value={settings.audioModel} onChange={(e) => onSettingsPatch({ audioModel: e.target.value })}>
              {availableAudioModelOptions.length === 0 && <option value="">(load models)</option>}
              {availableAudioModelOptions.map((modelId) => (
                <option key={modelId} value={modelId}>
                  {modelId}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>TTS voice</span>
            <select value={settings.audioVoice} onChange={(e) => onSettingsPatch({ audioVoice: e.target.value })}>
              {availableAudioVoiceOptions.length === 0 && <option value="">(load voices)</option>}
              {availableAudioVoiceOptions.map((voiceId) => (
                <option key={voiceId} value={voiceId}>
                  {audioVoiceLabels[voiceId] || voiceId}
                </option>
              ))}
            </select>
          </label>
        </div>

        {audioProviderKey === "elevenlabs" && (
          <div className="manual-voice-panel">
            <label>
              <span>Custom ElevenLabs voice ID</span>
              <div className="inline-actions">
                <input
                  value={customVoiceId}
                  onChange={(e) => setCustomVoiceId(e.target.value)}
                  placeholder="paste voice_id from your ElevenLabs library"
                />
                <button
                  className="btn"
                  type="button"
                  onClick={() => onCheckElevenLabsVoiceId(customVoiceId)}
                  disabled={busy || ttsOptionsBusy || !customVoiceId.trim()}
                >
                  Check & use voice ID
                </button>
              </div>
            </label>
            <p className="hint subtle">
              Use this when a library voice is available to your ElevenLabs API key but does not appear in the loaded catalogue.
            </p>
          </div>
        )}

        <div className="voice-preview-panel">
          <label>
            <span>Voice preview text</span>
            <div className="inline-actions">
              <input
                value={previewText}
                onChange={(e) => setPreviewText(e.target.value)}
                maxLength={300}
                placeholder="Dutch sample text"
              />
              <button
                className="btn"
                type="button"
                onClick={runVoicePreview}
                disabled={busy || ttsOptionsBusy || previewBusy || !previewText.trim() || !settings.audioVoice.trim()}
              >
                {previewBusy ? "Preparing..." : "Preview voice"}
              </button>
            </div>
          </label>
          <p className="hint subtle">This sends one short TTS request and may consume provider quota.</p>
          {previewError && <Notice notice={{ level: "error", message: previewError }} />}
          {previewAudioUrl && (
            <audio className="voice-preview-player" controls src={previewAudioUrl}>
              Your browser does not support audio playback.
            </audio>
          )}
        </div>

        {availableAudioModelOptions.length === 0 && (
          <p className="hint subtle">Reload models & voices to fetch available TTS models. The list also refreshes automatically.</p>
        )}
        <Notice notice={notices.audio} />
      </section>

      <section className="card section-card">
        <div className="section-head">
          <h3>Export defaults</h3>
          <span className="muted">Optional</span>
        </div>

        <div className="toggle-grid">
          <label className="check-tile">
            <input
              type="checkbox"
              checked={settings.includeBasicReversed}
              onChange={(e) => onSettingsPatch({ includeBasicReversed: e.target.checked })}
            />
            <span>Include basic (reversed)</span>
          </label>
          <label className="check-tile">
            <input
              type="checkbox"
              checked={settings.includeBasicTypein}
              onChange={(e) => onSettingsPatch({ includeBasicTypein: e.target.checked })}
            />
            <span>Include basic (type-in)</span>
          </label>
        </div>

        <label>
          <span>Default deck name (CSV/APKG filename)</span>
          <input
            value={settings.defaultDeck}
            onChange={(e) => onSettingsPatch({ defaultDeck: e.target.value })}
            placeholder="e.g. Dutch"
          />
        </label>

        <Notice notice={notices.export} />
      </section>

      <section className="card">
        <div className="section-head">
          <h3>Usage</h3>
        </div>

        {!usage && <div className="empty-state">No usage loaded yet.</div>}

        {usage && (
          <>
            <div className="summary-strip">
              <div className="summary-item">
                <span className="k">events</span> {usage.summary.events}
              </div>
              <div className="summary-item">
                <span className="k">tokens</span> {usage.summary.input_tokens}+{usage.summary.output_tokens}
              </div>
              <div className="summary-item">
                <span className="k">cached</span> {usage.summary.cached_tokens}
              </div>
              <div className="summary-item">
                <span className="k">audio chars</span> {usage.summary.audio_chars}
              </div>
              {usage.summary.raw_cost_usd != null && (
                <div className="summary-item">
                  <span className="k">est USD</span> {usage.summary.raw_cost_usd}
                </div>
              )}
            </div>
            <details className="raw-block">
              <summary>Usage details</summary>
              <pre>{shortJson(usage)}</pre>
            </details>
          </>
        )}
      </section>
    </section>
  );
}
