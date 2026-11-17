/**
 * Строка вывода под P (аудит 04-02): что значит P словами, со статистикой.
 *
 * «Средние различаются: F(2, 15) = 48.8, P < 0.001». Отказ от нулевой гипотезы
 * никогда не пишется как «различий нет» — только «не обнаружены».
 * Решение — по α метода (parameters.alpha), иначе 0.05.
 */

import type { StatResult } from "./pyodide/bridge";

/** Семейство гипотезы: от него зависят слова вывода (conclusion.<семейство>.yes/no). */
const FAMILY: Record<string, string> = {
  two_sample_t: "means", paired_t: "means", one_way_anova: "means", welch_anova: "means",
  rm_anova: "means", transformed_means: "means", rcbd_anova: "means",
  rcbd_subsampling_anova: "means", anova_permutation_test: "means",
  one_sample_t: "one_mean",
  mann_whitney: "ranks", kruskal_wallis: "ranks", friedman: "ranks", wilcoxon: "ranks",
  one_sample_wilcoxon: "one_location",
  one_proportion: "one_prop",
  two_proportions: "props", mcnemar_test: "props",
  chi2_goodness_of_fit: "gof",
  normality_test: "normality",
  bartlett_test: "variances",
  pearson_correlation: "association", spearman_correlation: "association",
  contingency_test: "association",
  simple_linear_regression: "model", regression_details: "model",
  x_transform_regression: "model", polynomial_regression: "model",
  trigonometric_regression: "model", nonlinear_regression: "model",
  multiple_regression: "model", blocked_regression: "model", glm_analysis: "model",
  count_regression: "model", ordinal_regression: "model",
  glmm_analysis: "random",
};

/** Статистика при P: символ, ключ, степени свободы (ключи или числа; "table" — из таблицы ANOVA). */
interface StatSpec {
  sym: string;
  key: string;
  df?: Array<string | number> | "table";
}

const F_GROUPS: StatSpec = { sym: "F", key: "F", df: ["df_between", "df_within"] };
const F_MODEL: StatSpec = { sym: "F", key: "F_model", df: "table" };

/**
 * Статистика, которой получено главное P. Где P поправлено (повторные: Хюйн–Фельдт)
 * или точное (одна доля, Мак-Немар, GLM с квази-F), df или сама статистика не пишутся.
 */
const STAT: Record<string, StatSpec> = {
  two_sample_t: { sym: "t", key: "t", df: ["df"] },
  paired_t: { sym: "t", key: "t", df: ["df"] },
  one_sample_t: { sym: "t", key: "t", df: ["df"] },
  one_way_anova: F_GROUPS, welch_anova: F_GROUPS, transformed_means: F_GROUPS,
  anova_permutation_test: F_GROUPS,
  rm_anova: { sym: "F", key: "F" },
  rcbd_anova: { sym: "F", key: "F", df: ["df_treatment", "df_resid"] },
  rcbd_subsampling_anova: { sym: "F", key: "F", df: ["df_treatment", "df_resid"] },
  mann_whitney: { sym: "U", key: "U" },
  kruskal_wallis: { sym: "H", key: "H", df: ["df"] },
  friedman: { sym: "χ²", key: "chi2", df: ["df"] },
  wilcoxon: { sym: "W", key: "W" },
  one_sample_wilcoxon: { sym: "W", key: "W" },
  two_proportions: { sym: "z", key: "z" },
  chi2_goodness_of_fit: { sym: "χ²", key: "chi2", df: ["df"] },
  normality_test: { sym: "W", key: "W" },
  bartlett_test: { sym: "χ²", key: "chi2", df: ["df"] },
  pearson_correlation: { sym: "r", key: "r", df: ["df"] },
  spearman_correlation: { sym: "ρ", key: "rho" },
  contingency_test: { sym: "χ²", key: "chi2", df: ["df"] },
  simple_linear_regression: { sym: "F", key: "f_statistic", df: [1, "df_resid"] },
  blocked_regression: { sym: "F", key: "F_x", df: [1, "df_resid"] },
  regression_details: F_MODEL, x_transform_regression: F_MODEL,
  polynomial_regression: F_MODEL, trigonometric_regression: F_MODEL,
  nonlinear_regression: F_MODEL, multiple_regression: F_MODEL,
  count_regression: { sym: "χ²", key: "lr_model", df: ["df_model"] },
  ordinal_regression: { sym: "χ²", key: "lr_model", df: ["df_model"] },
  glmm_analysis: { sym: "LR", key: "lr_random" },
};

/** P для текста: меньше 0.001 — «< 0.001», иначе три знака. */
export function pText(p: number): string {
  return p < 0.001 ? "< 0.001" : `= ${p.toFixed(3)}`;
}

/** α метода, если он его задаёт, иначе 0.05. */
export function alphaOf(result: StatResult): number {
  const a = result.parameters?.alpha;
  return typeof a === "number" && a > 0 && a < 1 ? a : 0.05;
}

function num(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) >= 100 ? v.toFixed(1) : String(Number(v.toPrecision(3)));
}

/** df модели по таблице ANOVA: общая минус остаточная. */
function tableDf(result: StatResult): number[] | null {
  const rows = result.parameters?.anova_table;
  if (!Array.isArray(rows)) return null;
  const df = (term: string) => (rows as Array<{ term: string; df: number }>)
    .find((r) => r.term === term)?.df;
  const total = df("Total");
  const resid = df("Residual");
  return total != null && resid != null ? [total - resid, resid] : null;
}

/** «F(2, 15) = 48.8» или null, если статистики в результате нет. */
export function statText(result: StatResult): string | null {
  const spec = STAT[result.method];
  const value = spec ? result.statistics[spec.key] : undefined;
  if (!spec || typeof value !== "number" || !Number.isFinite(value)) return null;
  let df: number[] | null = null;
  if (spec.df === "table") df = tableDf(result);
  else if (spec.df) {
    const vals = spec.df.map((d) => (typeof d === "number" ? d : result.statistics[d]));
    df = vals.every((v) => typeof v === "number" && Number.isFinite(v)) ? (vals as number[]) : null;
  }
  return `${spec.sym}${df ? `(${df.map(num).join(", ")})` : ""} = ${num(value)}`;
}

/** Ключ i18n вывода: conclusion.<семейство>.yes|no; null — метода нет в таблице семейств. */
export function conclusionKey(result: StatResult): string | null {
  const family = FAMILY[result.method];
  if (!family || result.p_value === null) return null;
  return `conclusion.${family}.${result.p_value < alphaOf(result) ? "yes" : "no"}`;
}

/** Полная строка: «Средние различаются: F(2, 15) = 48.8, P < 0.001». */
export function conclusionLine(result: StatResult, t: (key: string) => string): string | null {
  const key = conclusionKey(result);
  if (!key || result.p_value === null) return null;
  const stat = statText(result);
  return `${t(key)}: ${stat ? `${stat}, ` : ""}P ${pText(result.p_value)}`;
}

/** Методы с выводом словами — для тестов и проверки полноты. */
export const CONCLUSION_METHODS = Object.keys(FAMILY);
