/** Общие стили приложения. Вынесены отдельно, чтобы компоненты
 *  оставались читаемыми. */

import type { CSSProperties } from "react";

/**
 * Цветовые токены (аудит 4, часть Б). Контраст с белым: текст ≥ 4.5:1,
 * элементы интерфейса и графиков ≥ 3:1; проверяет tests/test_palette.py.
 */
export const c = {
  bg: "#fafafa",
  panel: "#ffffff",
  border: "#e4e4e7", // декоративные линии: панели, таблицы, сетка (1.27:1)
  borderInput: "#8e8e96", // рамка поля ввода и списка (3.25:1)
  text: "#18181b", // 17.7:1
  muted: "#63636b", // вторичный текст: 5.95:1 на белом, 5.42:1 на subtle (было #71717a)
  subtle: "#f4f4f5", // фон плашки P, бейджей
  accent: "#0f766e", // действие, выбранное, первая группа (5.47:1)
  selectedBg: "#f0fdfa", // фон выбранного элемента
  ok: "#166534",
  okBg: "#f0fdf4",
  warn: "#92400e",
  warnBg: "#fffbeb",
  warnLine: "#fde68a", // рамка карточки с замечанием (декоративная)
  bad: "#991b1b",
  badBg: "#fef2f2",
  info: "#1e40af",
  infoBg: "#eff6ff",
} as const;

/** Контуры узлов карты: сила рекомендации. Различимы при протан- и дейтанопии (ΔE 23 и 32). */
export const mapEdge = {
  strong: "#0f766e", // «указывают данные», 5.47:1 (было #16a34a)
  hint: "#b45309", // «подсказка», 5.02:1 (было #d97706)
  off: "#a1a1aa", // вне пути — рамка; текст таких узлов — c.muted
} as const;

/** Цвета графиков. Красный — только выброс: линии подгонки и опоры нейтральны. */
export const chart = {
  axis: "#52525b", // подписи и имена осей, 7.73:1
  axisLine: "#a1a1aa",
  grid: "#e4e4e7",
  ref: "#64748b", // опорные линии, границы ДИ, 4.76:1 (было #94a3b8)
  fit: "#1f2937", // линия подгонки, 14.7:1
  outlier: "#c81e1e", // выброс — и только он, 5.74:1
  /**
   * Группы, порядок фиксирован: цвет — по порядку группы, один во всех графиках
   * результата. Подобрано от фирменного бирюзового: контраст с белым ≥ 3:1,
   * минимальная ΔE2000 между любыми двумя ≥ 16.8 при всех трёх видах дальтонизма
   * (Okabe–Ito: 10.9–12.2). Больше 8 групп — цвета не добавляются.
   */
  categorical: ["#0f766e", "#aa3c08", "#2f92ff", "#ea479b", "#5c15a9", "#8f9762", "#1f2937", "#5e3b33"],
  /** Последовательная: один оттенок, L* равномерно 98 → 30. */
  sequential: ["#f0fdfa", "#bdeee6", "#7fd3c7", "#3fae9f", "#0f766e", "#134e4a"],
  /** Расходящаяся вокруг 0 (ColorBrewer RdBu): концы равной светлоты. */
  diverging: ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"],
} as const;

/** Типы линий и маркеры для кривых сверх восьми групп: цвет повторяется, форма — нет. */
const LINE_TYPES = ["solid", "dashed", "dotted"] as const;
const SYMBOLS = ["circle", "rect", "triangle", "diamond"] as const;

/** Цвет, линия и маркер кривой номер i (кривые по группам, взаимодействие). */
export function seriesStyle(i: number): { color: string; lineType: string; symbol: string } {
  const n = chart.categorical.length;
  const round = Math.floor(i / n);
  return {
    color: chart.categorical[i % n],
    lineType: LINE_TYPES[round % LINE_TYPES.length],
    symbol: SYMBOLS[round % SYMBOLS.length],
  };
}

/**
 * Цвет группы на графике, где группы стоят по оси (боксплот, средние): по её
 * номеру; при больше чем 8 группах — один цвет, различают подписи на оси.
 */
export function groupColor(i: number, nGroups: number): string {
  if (i < 0 || nGroups > chart.categorical.length) return c.accent;
  return chart.categorical[i];
}

export const s: Record<string, CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: c.bg,
    color: c.text,
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
    fontSize: 15,
  },
  container: { maxWidth: 1000, margin: "0 auto", padding: "2rem 1.5rem 4rem" },
  h1: { fontSize: "1.6rem", margin: "0 0 0.25rem" },
  sub: { color: c.muted, margin: "0 0 1.5rem" },
  panel: {
    background: c.panel,
    border: `1px solid ${c.border}`,
    borderRadius: 8,
    padding: "1.25rem",
    marginBottom: "1.25rem",
  },
  h2: { fontSize: "1.05rem", margin: "0 0 0.75rem", fontWeight: 600 },
  drop: {
    border: `2px dashed ${c.border}`,
    borderRadius: 8,
    padding: "2.5rem 1rem",
    textAlign: "center",
    cursor: "pointer",
    background: c.panel,
    transition: "border-color 0.15s",
  },
  dropActive: { borderColor: c.accent, background: c.selectedBg },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "0.85rem",
    fontVariantNumeric: "tabular-nums",
  },
  th: {
    textAlign: "left",
    padding: "0.4rem 0.6rem",
    borderBottom: `2px solid ${c.border}`,
    whiteSpace: "nowrap",
  },
  td: {
    padding: "0.35rem 0.6rem",
    borderBottom: `1px solid ${c.bg}`,
    whiteSpace: "nowrap",
  },
  mono: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" },
  label: {
    display: "block",
    fontSize: "0.8rem",
    color: c.muted,
    marginBottom: "0.2rem",
  },
  select: {
    padding: "0.4rem 0.5rem",
    border: `1px solid ${c.borderInput}`,
    borderRadius: 6,
    background: c.panel,
    fontSize: "0.9rem",
    minWidth: 160,
  },
  button: {
    padding: "0.5rem 1.1rem",
    border: "none",
    borderRadius: 6,
    background: c.accent,
    color: "#fff",
    fontSize: "0.9rem",
    fontWeight: 500,
    cursor: "pointer",
  },
  buttonDisabled: { background: c.border, color: c.muted, cursor: "not-allowed" },
  row: { display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-end" },
  badge: {
    display: "inline-block",
    padding: "0.1rem 0.45rem",
    borderRadius: 4,
    fontSize: "0.72rem",
    fontWeight: 600,
  },
  error: {
    padding: "0.85rem 1rem",
    background: c.badBg,
    color: c.bad,
    borderRadius: 6,
    marginBottom: "1rem",
    whiteSpace: "pre-wrap",
  },
};

/** Цвета бейджа под тип колонки. */
export function typeBadgeStyle(type: string): CSSProperties {
  const map: Record<string, [string, string]> = {
    numeric: ["#eff6ff", "#1e40af"],
    date: ["#f5f3ff", "#5b21b6"],
    categorical: ["#f0fdf4", "#166534"],
    id: ["#fef3c7", "#92400e"],
    empty: [c.subtle, c.muted],
  };
  const [bg, fg] = map[type] ?? [c.subtle, c.muted];
  return { ...s.badge, background: bg, color: fg };
}
