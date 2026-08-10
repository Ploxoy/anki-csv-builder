export type TextModelPricing = {
  input: number;
  cached_input?: number;
  cache_write?: number;
  output: number;
};

export type BudgetTier = "low" | "medium" | "high" | "very-high" | "unknown";

export type BudgetImpact = {
  model: string;
  pricing: TextModelPricing | null;
  tier: BudgetTier;
  label: string;
  multiplierVsLuna: number | null;
  estimatedPer100CardsUsd: number | null;
};

export const TEXT_MODEL_PRICING_USD_PER_1M: Record<string, TextModelPricing> = {
  "gpt-5.6-sol": { input: 5.0, cached_input: 0.5, cache_write: 6.25, output: 30.0 },
  "gpt-5.6": { input: 5.0, cached_input: 0.5, cache_write: 6.25, output: 30.0 },
  "gpt-5.6-terra": { input: 2.0, cached_input: 0.2, cache_write: 2.5, output: 12.0 },
  "gpt-5.6-luna": { input: 0.2, cached_input: 0.02, cache_write: 0.25, output: 1.2 },
  "gpt-5.5": { input: 5.0, cached_input: 0.5, output: 30.0 },
  "gpt-5.5-pro": { input: 30.0, output: 180.0 },
  "gpt-5.4": { input: 2.5, cached_input: 0.25, output: 15.0 },
  "gpt-5.4-mini": { input: 0.75, cached_input: 0.075, output: 4.5 },
  "gpt-5.4-nano": { input: 0.2, cached_input: 0.02, output: 1.25 },
  "gpt-5.4-pro": { input: 30.0, output: 180.0 },
};

export const BUDGET_COMPARISON_MODELS = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.5-pro"];

// Conservative UI estimate for one generated card. Actual run_report remains authoritative.
const EST_INPUT_TOKENS_PER_CARD = 1200;
const EST_OUTPUT_TOKENS_PER_CARD = 450;
const BASELINE_MODEL = "gpt-5.6-luna";

export function resolveTextPricing(modelId?: string | null): TextModelPricing | null {
  const clean = (modelId || "").trim();
  if (!clean) return null;
  const keys = Object.keys(TEXT_MODEL_PRICING_USD_PER_1M).sort((a, b) => b.length - a.length);
  const match = keys.find((key) => clean.startsWith(key));
  return match ? TEXT_MODEL_PRICING_USD_PER_1M[match] : null;
}

export function estimateTextCostUsd(modelId: string, cardCount: number): number | null {
  const pricing = resolveTextPricing(modelId);
  if (!pricing) return null;
  const safeCount = Math.max(0, cardCount);
  const inputTokens = safeCount * EST_INPUT_TOKENS_PER_CARD;
  const outputTokens = safeCount * EST_OUTPUT_TOKENS_PER_CARD;
  return (inputTokens / 1_000_000) * pricing.input + (outputTokens / 1_000_000) * pricing.output;
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "n/a";
  if (value === 0) return "$0";
  if (value < 0.001) return `$${value.toFixed(5)}`;
  if (value < 0.1) return `$${value.toFixed(4)}`;
  if (value < 10) return `$${value.toFixed(2)}`;
  return `$${Math.round(value).toLocaleString()}`;
}

export function formatMultiplier(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "n/a";
  if (value < 1.05) return "1x";
  if (value < 10) return `${value.toFixed(1)}x`;
  return `${Math.round(value)}x`;
}

export function getBudgetImpact(modelId: string): BudgetImpact {
  const pricing = resolveTextPricing(modelId);
  const estimatedPer100CardsUsd = estimateTextCostUsd(modelId, 100);
  const baseline = estimateTextCostUsd(BASELINE_MODEL, 100);
  const multiplierVsLuna = estimatedPer100CardsUsd != null && baseline ? estimatedPer100CardsUsd / baseline : null;
  let tier: BudgetTier = "unknown";
  let label = "Unknown cost";

  if (multiplierVsLuna != null) {
    if (multiplierVsLuna < 2) {
      tier = "low";
      label = "Low cost";
    } else if (multiplierVsLuna < 8) {
      tier = "medium";
      label = "Medium";
    } else if (multiplierVsLuna < 25) {
      tier = "high";
      label = "High";
    } else {
      tier = "very-high";
      label = "Very expensive";
    }
  }

  return { model: modelId, pricing, tier, label, multiplierVsLuna, estimatedPer100CardsUsd };
}

export function modelOptionLabel(modelId: string): string {
  const impact = getBudgetImpact(modelId);
  if (!impact.pricing) return modelId;
  return `${modelId} — ${impact.label} · ${formatMultiplier(impact.multiplierVsLuna)} Luna · ${formatUsd(impact.estimatedPer100CardsUsd)}/100`;
}

function readPathNumber(obj: unknown, path: string[]): number | null {
  let cur: any = obj;
  for (const key of path) {
    if (cur == null || typeof cur !== "object") return null;
    cur = cur[key];
  }
  const value = Number(cur);
  return Number.isFinite(value) ? value : null;
}

export function actualRunCostUsd(runReport: Record<string, unknown> | null | undefined): {
  total: number | null;
  text: number | null;
  audio: number | null;
} {
  return {
    total: readPathNumber(runReport, ["cost", "estimated_usd"]),
    text: readPathNumber(runReport, ["cost", "text", "estimated_usd"]),
    audio: readPathNumber(runReport, ["cost", "audio", "estimated_usd"]),
  };
}

export function buildBudgetComparison(cardCount: number): Array<{ model: string; cost: number | null; impact: BudgetImpact }> {
  return BUDGET_COMPARISON_MODELS.map((model) => ({ model, cost: estimateTextCostUsd(model, cardCount), impact: getBudgetImpact(model) }));
}
