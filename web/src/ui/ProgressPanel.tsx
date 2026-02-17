import { useMemo, useState } from "react";
import { formatElapsedMs, GenerateProgressMeta, progressStageLabel } from "../lib/uiState";

type ProgressPanelProps = {
  busy: boolean;
  progress: number;
  label: string;
  meta: GenerateProgressMeta | null;
};

export function ProgressPanel({ busy, progress, label, meta }: ProgressPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const percent = Math.max(0, Math.min(100, progress));
  const stageText = useMemo(() => (meta ? progressStageLabel(meta.stage) : "Queued"), [meta]);

  if (!busy && progress < 100 && !meta) return null;

  return (
    <section className={`progress-panel ${busy ? "active" : "done"}`} aria-live="polite">
      <div className="progress-topline">
        <strong>{busy ? "Generation in progress" : "Generation complete"}</strong>
        <span>{Math.round(percent)}%</span>
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>

      <div className="progress-compact">
        <span>
          <span className="k">stage</span> {stageText}
        </span>
        {meta && (
          <button className="btn tiny" type="button" onClick={() => setExpanded((v) => !v)}>
            {expanded ? "Hide details" : "Show details"}
          </button>
        )}
      </div>

      {expanded && meta && (
        <div className="progress-details">
          <span>
            <span className="k">done</span> {meta.done}/{meta.total}
          </span>
          <span>
            <span className="k">batch</span> {meta.batchTotal > 0 ? `${meta.batchIndex}/${meta.batchTotal}` : "-"}
          </span>
          <span>
            <span className="k">elapsed</span> {formatElapsedMs(meta.elapsedMs)}
          </span>
          <span className={meta.waitingProvider ? "waiting-pill waiting" : "waiting-pill"}>
            {meta.waitingProvider ? "waiting provider..." : "provider responsive"}
          </span>
        </div>
      )}

      {label && <p className="hint subtle">{label}</p>}
    </section>
  );
}
