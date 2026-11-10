/**
 * Плашка долгого расчёта: сколько секунд идёт, кнопка «Остановить».
 *
 * Расчёт идёт в Web Worker'е, поэтому страница не замирает и плашка
 * обновляется. Появляется через секунду (короткие расчёты её не показывают);
 * подсказка про долгий метод — сразу.
 */

import { useEffect, useState, type ReactNode } from "react";
import { useT } from "./i18n";
import { c, s } from "./styles";

/** Методы, долгие на большом файле: порог строк и ключ подсказки. */
const SLOW: Record<string, { rows: number; key: string }> = {
  mixed_model: { rows: 500, key: "run.slow_mixed" },
};

/** Подсказка о долгом расчёте или null. */
export function slowHint(method: string, rows: number): string | null {
  const slow = SLOW[method];
  return slow && rows > slow.rows ? slow.key : null;
}

export function RunStatus({
  active,
  hint,
  restarting,
  onCancel,
}: {
  active: boolean;
  hint: string | null;
  restarting: boolean;
  onCancel: () => void;
}) {
  const t = useT();
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    setSeconds(0);
    if (!active) return;
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 500);
    return () => clearInterval(timer);
  }, [active]);

  if (restarting) return <Bar>{t("run.restarting")}</Bar>;
  if (!active || (seconds < 1 && !hint)) return null;
  return (
    <Bar>
      <span>
        {t("run.elapsed").replace("{s}", String(seconds))}
        {hint && <span style={{ display: "block", color: c.muted }}>{t(hint)}</span>}
      </span>
      <button onClick={onCancel} style={{ ...s.button, padding: "0.3rem 0.8rem" }}>
        {t("run.cancel")}
      </button>
    </Bar>
  );
}

function Bar({ children }: { children: ReactNode }) {
  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: "fixed", right: 16, bottom: 16, left: 16, maxWidth: 520, marginLeft: "auto",
        display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.8rem",
        background: c.panel, border: `1px solid ${c.border}`, borderRadius: 8,
        padding: "0.6rem 0.9rem", boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
        fontSize: "0.88rem", zIndex: 50,
      }}
    >
      {children}
    </div>
  );
}
