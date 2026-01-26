import { GenerateItem } from "../types";

type ParseResult = { items: GenerateItem[]; warnings: string[] };

function normalizeLine(line: string): string {
  return line.replace(/\r/g, "").trim();
}

export function parseItems(text: string): ParseResult {
  const warnings: string[] = [];
  const items: GenerateItem[] = [];

  const lines = text.split("\n").map(normalizeLine).filter(Boolean);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const rowNum = i + 1;

    let parts: string[] = [];
    if (line.includes("\t")) {
      parts = line.split("\t").map((s) => s.trim());
    } else if (line.includes(";;")) {
      parts = line.split(";;").map((s) => s.trim());
    } else if (line.includes("—")) {
      parts = line.split("—").map((s) => s.trim());
    } else if (line.includes(" - ")) {
      parts = line.split(" - ").map((s) => s.trim());
    } else {
      parts = [line];
    }

    const woord = (parts[0] || "").trim();
    if (!woord) {
      warnings.push(`Line ${rowNum}: missing woord`);
      continue;
    }

    const def_nl = (parts[1] || "").trim();
    const translation = (parts[2] || "").trim();

    items.push({
      id: String(rowNum),
      woord,
      def_nl: def_nl || undefined,
      translation: translation || undefined
    });
  }

  if (items.length === 0) {
    warnings.push("No items parsed. Provide at least one line like: woord<TAB>def_nl<TAB>translation");
  }

  return { items, warnings };
}

