/**
 * Сведения о расчёте для выгрузки (задание P4): по файлу результата можно
 * доказать, чем и на каком файле он посчитан. JSON — блок `provenance`, Excel —
 * лист «Сведения», под результатом — одна строка. CSV не меняется: служебные
 * строки ломают чтение в R и Excel.
 */

import type { Cell } from "./resultExport";

declare const __APP_VERSION__: string;
declare const __BUILD_DATE__: string;

/** Короткий хэш коммита сборки и дата сборки (vite.config.ts, define). */
export const APP_VERSION: string = typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "dev";
export const BUILD_DATE: string = typeof __BUILD_DATE__ === "string" ? __BUILD_DATE__ : "";

export interface InputFile {
  name: string;
  size_bytes: number;
  sha256: string;
}

export interface Provenance {
  app_version: string;
  build_date: string;
  computed_at: string;
  input_file: (InputFile & { encoding: string; delimiter: string; decimal: string; has_header: boolean }) | null;
  runtime: Record<string, string>;
  method: { id: string; name: string };
  parameters: Record<string, unknown>;
  columns: Record<string, unknown>;
  language: string;
  study_context: string;
}

/** SHA-256 тех же байтов, что ушли в parseFile, — шестнадцатеричной строкой. */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** ISO 8601 с местным смещением часового пояса: 2026-11-20T22:10:05+00:00. */
export function isoWithOffset(date: Date): string {
  const pad = (n: number) => String(Math.trunc(Math.abs(n))).padStart(2, "0");
  const off = -date.getTimezoneOffset();
  const local = new Date(date.getTime() + off * 60_000).toISOString().slice(0, 19);
  return `${local}${off >= 0 ? "+" : "-"}${pad(off / 60)}:${pad(off % 60)}`;
}

/** Все сведения о расчёте одним объектом (поля — как в задании P4). */
export function buildProvenance(args: Omit<Provenance, "app_version" | "build_date">): Provenance {
  return { app_version: APP_VERSION, build_date: BUILD_DATE, ...args };
}

/** Значение для ячейки Excel: вложенное — JSON одной строкой. */
function flat(v: unknown): Cell {
  if (v == null) return "";
  if (typeof v === "number" || typeof v === "string") return v;
  return JSON.stringify(v);
}

/** Лист «Сведения»: «поле — значение», вложенные объекты разворачиваются в «a.b». */
export function provenanceRows(p: Provenance, t: (key: string) => string): Cell[][] {
  const rows: Cell[][] = [[t("export.item"), t("export.value")]];
  const walk = (prefix: string, value: unknown) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const entries = Object.entries(value as Record<string, unknown>);
      if (entries.length === 0) rows.push([prefix, ""]);
      entries.forEach(([k, v]) => walk(prefix ? `${prefix}.${k}` : k, v));
    } else {
      rows.push([prefix, flat(value)]);
    }
  };
  walk("", p);
  return rows;
}

/** Строка под результатом: «BioStat 4f9a250 · 2026-11-20 22:10 · файл SHA-256 1a2b3c4d5e6f». */
export function provenanceLine(p: Provenance, t: (key: string) => string): string {
  const when = p.computed_at.slice(0, 16).replace("T", " ");
  const file = p.input_file ? ` · ${t("provenance.file")} SHA-256 ${p.input_file.sha256.slice(0, 12)}` : "";
  return `BioStat ${p.app_version} · ${when}${file}`;
}
