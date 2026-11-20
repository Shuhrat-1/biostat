/**
 * Выгрузка результата метода в CSV и Excel (аудит 05-07, 04-12).
 *
 * Один набор на результат: раздел «Итог» (метод, n, P, вывод, статистики с
 * подписями) и по разделу на каждую таблицу — оценки, ANOVA, дополнительные.
 * Excel — лист на раздел; CSV — разделы подряд, между ними пустая строка.
 * Ячейки CSV проходят csvCell (защита от формул, аудит 03-04).
 */

import { alphaOf, conclusionLine } from "./conclusion";
import { csvCell } from "./escape";
import { columnLabel, estimateLabel, extraTitle, statLabel, termLabel } from "./extraLabels";
import type { Lang } from "./i18n";
import type { StatResult } from "./pyodide/bridge";

export type Cell = string | number;
export interface Section {
  title: string;
  rows: Cell[][];
}

type Raw = Record<string, unknown>;
type T = (key: string) => string;

/** Данные графиков и служебные ключи: в таблицы не выгружаются. */
const NOT_TABLES = new Set(["plots", "curve", "curves", "points", "line", "column_names", "anova_table"]);

const isObject = (v: unknown): v is Raw => !!v && typeof v === "object" && !Array.isArray(v);
const isScalar = (v: unknown) => v == null || ["number", "string", "boolean"].includes(typeof v);
const isRows = (v: unknown): v is Raw[] => Array.isArray(v) && v.length > 0 && v.every(isObject);

/**
 * Ячейка: числа — 12 значащих цифр (полная точность расчёта без хвостов
 * двоичной арифметики вроде 5.530000000000001), NaN и бесконечность — пусто.
 */
function cell(v: unknown): Cell {
  if (v == null) return "";
  if (typeof v === "number") return Number.isFinite(v) ? Number(v.toPrecision(12)) : "";
  if (typeof v === "boolean") return v ? "✓" : "—";
  if (Array.isArray(v)) return v.every(isScalar) ? v.map((x) => String(cell(x))).join(" ") : `[${v.length}]`;
  if (isObject(v)) return Object.entries(v).map(([k, x]) => `${k}=${cell(x)}`).join(", ");
  return String(v);
}

function summarySection(result: StatResult, t: T, lang: Lang, label: string, caveat: string | null): Section {
  const rows: Cell[][] = [[t("export.item"), t("export.value")], [t("export.method"), label],
    ["n", cell(result.n)]];
  if (result.p_value !== null) rows.push(["P", cell(result.p_value)], ["α", alphaOf(result)]);
  const line = conclusionLine(result, t);
  if (line) rows.push([t("export.conclusion"), line]);
  if (caveat) rows.push([t("export.caveat"), caveat]);
  for (const [k, v] of Object.entries(result.statistics)) rows.push([statLabel(k, lang), cell(v)]);
  return { title: t("export.summary"), rows };
}

function estimatesSection(result: StatResult, t: T, lang: Lang): Section | null {
  if (!result.estimates.length) return null;
  const head = [t("export.estimate"), t("export.value"), t("export.ciLower"), t("export.ciUpper"), "SE"];
  return {
    title: t("export.estimates"),
    rows: [head, ...result.estimates.map((e) => [estimateLabel(e.name, result.parameters?.labels, lang),
      cell(e.value), cell(e.ci_lower),
      cell(e.ci_upper), cell(e.std_error)])],
  };
}

/** Строки однотипных объектов → таблица; колонки — объединение ключей по порядку. */
function rowsTable(rows: Raw[], lang: Lang): Cell[][] {
  const keys: string[] = [];
  rows.forEach((r) => Object.keys(r).forEach((k) => keys.includes(k) || keys.push(k)));
  return [keys.map((k) => columnLabel(k, lang)), ...rows.map((r) => keys.map((k) => cell(r[k])))];
}

/** Значение параметра → разделы (вложенные словари — до двух уровней). */
function sectionsOf(value: unknown, title: string, lang: Lang, depth = 0): Section[] {
  if (isRows(value)) return [{ title, rows: rowsTable(value, lang) }];
  if (isObject(value) && Array.isArray(value.rows) && Array.isArray(value.columns)
      && Array.isArray(value.counts)) {
    const counts = value.counts as unknown[][];
    return [{ title, rows: [["", ...value.columns.map(cell)],
      ...value.rows.map((r, i) => [cell(r), ...(counts[i] ?? []).map(cell)])] }];
  }
  if (!isObject(value)) return [];
  const entries = Object.entries(value);
  if (entries.length && entries.every(([, v]) => isScalar(v))) {
    return [{ title, rows: entries.map(([k, v]) => [columnLabel(k, lang), cell(v)]) }];
  }
  if (depth >= 2) return [];
  return entries.flatMap(([k, v]) => sectionsOf(v, `${title} · ${columnLabel(k, lang)}`, lang, depth + 1));
}

/** Все разделы результата в порядке экрана. */
export function resultSections(result: StatResult, t: T, lang: Lang, methodLabel?: string,
  caveat: string | null = null): Section[] {
  const params = (result.parameters ?? {}) as Raw;
  const out: Section[] = [summarySection(result, t, lang, methodLabel ?? result.method, caveat)];
  const est = estimatesSection(result, t, lang);
  if (est) out.push(est);
  if (isRows(params.anova_table)) {
    const rows = params.anova_table.map((r) => ({ ...r, term: termLabel(String(r.term), lang) }));
    out.push({ title: t("export.anova"), rows: rowsTable(rows, lang) });
  }
  for (const [k, v] of Object.entries(params)) {
    if (!NOT_TABLES.has(k)) out.push(...sectionsOf(v, extraTitle(k, lang, result.method), lang));
  }
  return out;
}

/** CSV: разделы подряд, заголовок раздела строкой, пустая строка между; «;» и BOM для Excel. */
export function sectionsToCsv(sections: Section[]): string {
  const lines = sections.flatMap((sec, i) => [
    ...(i > 0 ? [""] : []), csvCell(sec.title), ...sec.rows.map((r) => r.map((x) => csvCell(x)).join(";")),
  ]);
  return "﻿" + lines.join("\n") + "\n";
}

/** Имена листов Excel: без []:*?/\, не длиннее 31 знака, без повторов. */
export function sheetNames(sections: Section[]): string[] {
  const used = new Set<string>();
  return sections.map((sec) => {
    const base = sec.title.replace(/[[\]:*?/\\]/g, " ").trim().slice(0, 31) || "Sheet";
    let name = base;
    for (let n = 2; used.has(name.toLowerCase()); n++) name = `${base.slice(0, 31 - `(${n})`.length - 1)} (${n})`;
    used.add(name.toLowerCase());
    return name;
  });
}
