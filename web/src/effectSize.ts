/**
 * Подписи размера эффекта по ориентирам Коэна (пакет W2, ключ effect_size_label).
 *
 * Ориентиры взяты из Witte & Witte (2017), куда пришли из поведенческих наук
 * (Cohen 1988); сам Witte оговаривает, что их трудно интерпретировать. Поэтому
 * подпись — справка рядом с числом, а не вердикт: для биолога главное —
 * разность в единицах отклика с ДИ (Welham).
 */

export type EffectSize = "small" | "medium" | "large";

interface Family {
  /** Ориентиры «малый / средний / большой». */
  marks: [number, number, number];
  /** Источник ориентиров для подписи. */
  source: string;
}

const D: Family = { marks: [0.2, 0.5, 0.8], source: "Witte, Table 14.2" };
const ETA: Family = { marks: [0.01, 0.09, 0.25], source: "Witte, Table 16.7" };
const PHI: Family = { marks: [0.01, 0.09, 0.25], source: "Witte, Table 19.6" };
const W: Family = { marks: [0.1, 0.3, 0.5], source: "Cohen 1988 [предл.]" };
/** Связь (r Пирсона, ρ Спирмена) и согласованность W Кендалла: ориентиры
 *  Коэна для r; для W — по аналогии с r, как принято [предл.]. */
const R: Family = { marks: [0.1, 0.3, 0.5], source: "Cohen 1988 [предл.]" };

/** Ключ результата ядра → семейство ориентиров. */
export function familyOf(key: string): Family | null {
  if (["cohens_d", "hedges_g", "cohens_dz"].includes(key)) return D;
  if (["eta_squared", "eta_squared_partial", "omega_squared"].includes(key) || key.startsWith("eta_p_")) return ETA;
  if (["phi_c_squared", "cramers_v_squared"].includes(key)) return PHI;
  if (key === "cohens_w") return W;
  if (["r", "rho", "kendalls_w"].includes(key)) return R;
  return null;
}

/**
 * Категория по ближайшему ориентиру («в окрестности», Witte стр. 262):
 * границы — середины между ориентирами; ниже «малого» — тоже «малый».
 */
export function classify(value: number, family: Family): EffectSize {
  const [s, m, l] = family.marks;
  const v = Math.abs(value);
  if (v < (s + m) / 2) return "small";
  return v < (m + l) / 2 ? "medium" : "large";
}

export interface EffectRow {
  key: string;
  value: number;
  size: EffectSize;
  source: string;
}

/** Размеры эффекта из statistics результата, в порядке появления. */
export function effectRows(statistics: Record<string, unknown>): EffectRow[] {
  const rows: EffectRow[] = [];
  for (const [key, raw] of Object.entries(statistics)) {
    const family = familyOf(key);
    if (!family || typeof raw !== "number" || !Number.isFinite(raw)) continue;
    rows.push({ key, value: raw, size: classify(raw, family), source: family.source });
  }
  return rows;
}
