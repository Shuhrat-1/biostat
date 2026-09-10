/** Общие стили приложения. Вынесены отдельно, чтобы компоненты
 *  оставались читаемыми. */

import type { CSSProperties } from "react";

export const c = {
  bg: "#fafafa",
  panel: "#ffffff",
  border: "#e4e4e7",
  text: "#18181b",
  muted: "#71717a",
  accent: "#0f766e",
  ok: "#166534",
  okBg: "#f0fdf4",
  warn: "#92400e",
  warnBg: "#fffbeb",
  bad: "#991b1b",
  badBg: "#fef2f2",
} as const;

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
  dropActive: { borderColor: c.accent, background: "#f0fdfa" },
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
    border: `1px solid ${c.border}`,
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
    empty: ["#f4f4f5", c.muted],
  };
  const [bg, fg] = map[type] ?? ["#f4f4f5", c.muted];
  return { ...s.badge, background: bg, color: fg };
}
