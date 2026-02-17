import { ChangeEvent, RefObject, useMemo, useState } from "react";
import { GenerateResponse } from "../../types";
import { NoticeMessage } from "../../lib/messages";
import {
  AudioRunSummary,
  ExportFormat,
  GenerateProgressMeta,
  Settings,
  TemperatureParseResult,
  formatPercent,
  shortJson,
} from "../../lib/uiState";
import { Notice } from "../../ui/Notice";
import { ProgressPanel } from "../../ui/ProgressPanel";

type GenerateTabProps = {
  settings: Settings;
  parsedCount: number;
  warnings: string[];
  inputText: string;
  onInputTextChange: (value: string) => void;
  onSettingsPatch: (patch: Partial<Settings>) => void;
  onGenerate: () => void;
  canGenerate: boolean;
  busy: boolean;
  onOpenFilePicker: (mode: "replace" | "append") => void;
  fileInputRef: RefObject<HTMLInputElement>;
  onInputFilesSelected: (event: ChangeEvent<HTMLInputElement>) => void;
  onSaveInputText: () => void;
  response: GenerateResponse | null;
  generateProgress: number;
  generateProgressLabel: string;
  generateProgressMeta: GenerateProgressMeta | null;
  notices: {
    input: NoticeMessage | null;
    run: NoticeMessage | null;
    audio: NoticeMessage | null;
    export: NoticeMessage | null;
  };
  onExportDeck: (format: ExportFormat) => void;
  exportBusy: ExportFormat | null;
  exportCardCount: number;
  deckPreview: string;
  temperatureState: TemperatureParseResult;
  audioClipCount: number;
  audioRunSummary: AudioRunSummary | null;
  hasAudioFailures: boolean;
  onGoSettings: () => void;
};

type ResultFilter = "all" | "errors" | "repaired";

export function GenerateTab({
  settings,
  parsedCount,
  warnings,
  inputText,
  onInputTextChange,
  onSettingsPatch,
  onGenerate,
  canGenerate,
  busy,
  onOpenFilePicker,
  fileInputRef,
  onInputFilesSelected,
  onSaveInputText,
  response,
  generateProgress,
  generateProgressLabel,
  generateProgressMeta,
  notices,
  onExportDeck,
  exportBusy,
  exportCardCount,
  deckPreview,
  temperatureState,
  audioClipCount,
  audioRunSummary,
  hasAudioFailures,
  onGoSettings,
}: GenerateTabProps) {
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all");
  const [showAllAudioErrors, setShowAllAudioErrors] = useState(false);

  const filteredItems = useMemo(() => {
    const items = response?.items || [];
    if (resultFilter === "all") return items;
    if (resultFilter === "repaired") return items.filter((it) => it.status === "repaired");
    return items.filter((it) => it.status === "failed" || it.status === "flagged" || !!it.error);
  }, [response, resultFilter]);

  const audioErrorPreview = useMemo(() => {
    if (!audioRunSummary?.errors?.length) return [];
    return audioRunSummary.errors.slice(0, 5);
  }, [audioRunSummary]);

  const summaryCounts = useMemo(() => {
    const items = response?.items || [];
    const ok = items.filter((it) => it.status === "ok").length;
    const repaired = items.filter((it) => it.status === "repaired").length;
    const failed = items.filter((it) => it.status === "failed" || it.status === "flagged" || !!it.error).length;
    return { total: items.length, ok, repaired, failed };
  }, [response]);

  return (
    <div className="tab-layout">
      <section className="card">
        <div className="section-head">
          <h2>Input</h2>
          <span className="muted">Step 1</span>
        </div>
        <p className="hint flow-hint">Load text rows, then run generation. One row per line.</p>

        <div className="toolbar row-wrap">
          <div className="meta-chip">
            <span className="k">rows parsed</span> {parsedCount}
          </div>
          <div className="toolbar-actions">
            <button className="btn small" type="button" onClick={() => onOpenFilePicker("replace")} disabled={busy}>
              Load file
            </button>
            <button className="btn small" type="button" onClick={() => onOpenFilePicker("append")} disabled={busy}>
              Append file
            </button>
            <button className="btn small" type="button" onClick={onSaveInputText} disabled={busy || !inputText.trim()}>
              Save text
            </button>
            <button className="btn" type="button" onClick={onGoSettings}>
              Open settings
            </button>
          </div>
        </div>

        <Notice notice={notices.input} />

        <textarea
          value={inputText}
          onChange={(e) => onInputTextChange(e.target.value)}
          rows={7}
          placeholder="aanraken\tiets voelen\tto touch"
        />

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.tsv,.csv,text/plain,text/tab-separated-values,text/csv"
          onChange={onInputFilesSelected}
          className="hidden-file-input"
        />

        {warnings.length > 0 && (
          <div className="stack-sm">
            {warnings.map((warning) => (
              <div key={warning} className="inline-warning">
                {warning}
              </div>
            ))}
          </div>
        )}

        <p className="hint subtle">Accepted formats: TSV, `woord ;; def ;; translation`, `woord - def - translation`.</p>
      </section>

      <section className="card">
        <div className="section-head">
          <h2>Run</h2>
          <span className="muted">Step 2</span>
        </div>

        <div className="chips">
          <span className="chip">Model: {settings.model}</span>
          <span className={`chip ${temperatureState.error ? "danger" : ""}`}>
            Temp: {temperatureState.percent != null ? `${formatPercent(temperatureState.percent)}%` : "default"}
          </span>
          <span className="chip">CEFR: {settings.cefr}</span>
          <span className="chip">Profile: {settings.profile}</span>
          <span className="chip">Audio: {settings.audioProvider}</span>
          <span className="chip">Deck: {settings.defaultDeck || "-"}</span>
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
            <span>Include word audio</span>
          </label>
          <label className={`check-tile ${!settings.generateAudio ? "disabled" : ""}`}>
            <input
              type="checkbox"
              checked={settings.includeAudioSentence}
              disabled={!settings.generateAudio}
              onChange={(e) => onSettingsPatch({ includeAudioSentence: e.target.checked })}
            />
            <span>Include sentence audio</span>
          </label>
          <label className="check-tile">
            <input
              type="checkbox"
              checked={settings.includeFlagged}
              onChange={(e) => onSettingsPatch({ includeFlagged: e.target.checked })}
            />
            <span>Force flagged rows</span>
          </label>
          <label className="check-tile">
            <input
              type="checkbox"
              checked={settings.includeBasicReversed}
              onChange={(e) => onSettingsPatch({ includeBasicReversed: e.target.checked })}
            />
            <span>Export basic reversed</span>
          </label>
          <label className="check-tile">
            <input
              type="checkbox"
              checked={settings.includeBasicTypein}
              onChange={(e) => onSettingsPatch({ includeBasicTypein: e.target.checked })}
            />
            <span>Export basic type-in</span>
          </label>
        </div>

        {temperatureState.error && <Notice notice={{ level: "error", message: temperatureState.error }} />}
        <Notice notice={notices.run} />

        <div className="run-row">
          <div className="meta-chip">
            <span className="k">deck preview</span> {deckPreview}
          </div>
          <button className="btn primary" type="button" onClick={onGenerate} disabled={busy || !canGenerate}>
            {busy ? "Generating..." : `Generate (${parsedCount})`}
          </button>
        </div>

        {(busy || (response && generateProgress >= 100)) && (
          <ProgressPanel
            busy={busy}
            progress={generateProgress}
            label={generateProgressLabel}
            meta={generateProgressMeta}
          />
        )}
      </section>

      <section className="card">
        <div className="section-head">
          <h2>Review</h2>
          <span className="muted">Step 3</span>
        </div>

        {!response && <div className="empty-state">No generation yet. Run Step 2 to review cards.</div>}

        {response && (
          <>
            <div className="summary-strip">
              <div className="summary-item">
                <span className="k">total</span> {summaryCounts.total}
              </div>
              <div className="summary-item">
                <span className="k">ok</span> {summaryCounts.ok}
              </div>
              <div className="summary-item">
                <span className="k">repaired</span> {summaryCounts.repaired}
              </div>
              <div className="summary-item">
                <span className="k">issues</span> {summaryCounts.failed}
              </div>
              <div className="summary-item">
                <span className="k">elapsed</span> {response.timing?.elapsed_ms} ms
              </div>
            </div>

            <div className="toolbar row-wrap compact-top">
              <div className="toolbar-actions">
                <button
                  className={`btn small ${resultFilter === "all" ? "active" : ""}`}
                  type="button"
                  onClick={() => setResultFilter("all")}
                >
                  All
                </button>
                <button
                  className={`btn small ${resultFilter === "errors" ? "active" : ""}`}
                  type="button"
                  onClick={() => setResultFilter("errors")}
                >
                  Errors
                </button>
                <button
                  className={`btn small ${resultFilter === "repaired" ? "active" : ""}`}
                  type="button"
                  onClick={() => setResultFilter("repaired")}
                >
                  Repaired
                </button>
              </div>
            </div>

            {filteredItems.length === 0 && <div className="empty-state">No rows match this filter.</div>}

            {filteredItems.length > 0 && (
              <div className="result-list">
                {filteredItems.map((item) => (
                  <article className="result-row" key={item.id}>
                    <span className={`status-pill ${item.status}`}>{item.status}</span>
                    <div className="result-main">
                      <div className="result-id">#{item.id}</div>
                      {item.error && <p className="error-line">{item.error}</p>}
                      {item.card && (
                        <details>
                          <summary>{item.card.L2_word}</summary>
                          <pre>{shortJson(item.card)}</pre>
                        </details>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}

            <details className="raw-block">
              <summary>Raw response</summary>
              <pre>{shortJson(response)}</pre>
            </details>
          </>
        )}
      </section>

      <section className="card">
        <div className="section-head">
          <h2>Export</h2>
          <span className="muted">Step 4</span>
        </div>

        {!response && <div className="empty-state">Export becomes available after generation.</div>}

        {response && (
          <>
            <div className="toolbar row-wrap">
              <div className="toolbar-actions">
                <button
                  className="btn small"
                  type="button"
                  onClick={() => onExportDeck("csv")}
                  disabled={busy || exportBusy != null || exportCardCount === 0}
                >
                  {exportBusy === "csv" ? "Preparing CSV..." : `Download CSV (${exportCardCount})`}
                </button>
                <button
                  className="btn small"
                  type="button"
                  onClick={() => onExportDeck("apkg")}
                  disabled={busy || exportBusy != null || exportCardCount === 0}
                >
                  {exportBusy === "apkg" ? "Preparing APKG..." : "Download APKG"}
                </button>
              </div>
              <div className="meta-chip">
                <span className="k">audio clips</span> {audioClipCount}
              </div>
            </div>

            {audioRunSummary?.requested && (
              <p className="hint subtle">
                Audio summary: {audioRunSummary.ok}/{audioRunSummary.total} ready, {audioRunSummary.failed} failed.
              </p>
            )}

            <Notice notice={notices.audio} />

            {hasAudioFailures && audioErrorPreview.length > 0 && (
              <div className="inline-warning">
                <div>First error: {audioErrorPreview[0]}</div>
                {(audioRunSummary?.errors.length || 0) > 1 && (
                  <button className="btn tiny" type="button" onClick={() => setShowAllAudioErrors((v) => !v)}>
                    {showAllAudioErrors ? "Hide all errors" : `Show all errors (${audioRunSummary?.errors.length || 0})`}
                  </button>
                )}
                {showAllAudioErrors && <pre>{(audioRunSummary?.errors || []).join("\n")}</pre>}
              </div>
            )}

            <Notice notice={notices.export} />
          </>
        )}
      </section>
    </div>
  );
}
