/**
 * Мост между JS и расчётным ядром на Python.
 *
 * ЕДИНСТВЕННАЯ точка вызова Python из интерфейса. При добавлении других
 * бэкендов (WebGPU, cloud) меняется только этот файл — компоненты UI
 * продолжают вызывать те же функции.
 *
 * Обмен идёт через JSON: граница остаётся чистой, JS получает обычные
 * объекты, нет утечек памяти от Python-прокси. Разобранная таблица живёт
 * в Python-глобальных и не гоняется туда-обратно при каждом расчёте.
 */

import { initRuntime } from "./runtime";

export type Severity = "info" | "warning" | "error";

export interface Warning {
  code: string;
  message: string;
  severity: Severity;
  params?: Record<string, unknown>;
}

export interface Estimate {
  name: string;
  value: number;
  ci_lower: number | null;
  ci_upper: number | null;
  ci_level: number | null;
  std_error: number | null;
}

/** Зеркало StatResult из core/report/schema.py */
export interface StatResult {
  method: string;
  estimates: Estimate[];
  statistics: Record<string, number>;
  p_value: number | null;
  warnings: Warning[];
  parameters: Record<string, unknown>;
  n: number | null;
  seed: number | null;
  data_quality: Record<string, unknown> | null;
}

export interface CoreMsg {
  code: string;
  message: string;
  params?: Record<string, unknown>;
}

export interface DialectInfo {
  encoding: string;
  encoding_confidence: number;
  delimiter: string;
  delimiter_confidence: number;
  decimal: string;
  has_header: boolean;
  n_columns: number;
  notes: CoreMsg[];
}

export type ColumnType = "numeric" | "date" | "categorical" | "id" | "empty";

export interface ColumnProfile {
  name: string;
  detected_type: ColumnType;
  confidence: number;
  n_total: number;
  n_missing: number;
  missing_ratio: number;
  n_type_mismatch: number;
  mismatch_examples: string[];
  date_formats: Record<string, number>;
  n_unique: number;
  n_duplicates: number;
  likely_categorical: boolean;
  likely_id: boolean;
  issues: Array<{
    code: string;
    message: string;
    params?: Record<string, unknown>;
  }>;
}

export interface QualitySummary {
  verdict: "ok" | "warnings" | "problems";
  n_rows: number;
  n_columns: number;
  n_missing: number;
  n_type_mismatch: number;
  problem_ratio: number;
  problem_columns: string[];
  top_issues: Array<{
    column: string;
    code: string;
    message: string;
    params?: Record<string, unknown>;
  }>;
}

export interface MethodParam {
  name: string;
  label: string;
  kind: "number" | "select" | "float" | "text";
  default: string | number;
  options: string[];
  min: number | null;
  max: number | null;
  from_summary: string;
}

export interface MethodSuggestion {
  method: string;
  label: string;
  group: string;
  needs: Record<string, string[]>;
  slot_labels: Record<string, string>;
  params: MethodParam[];
  n_groups: number | null;
  description: string;
  /** Слоты, которые можно оставить пустыми (третий фактор, блок). */
  optional_slots: string[];
}

export interface ParsedTable {
  dialect: DialectInfo;
  header: string[];
  preview_rows: string[][];
  profiles: ColumnProfile[];
  n_rows: number;
  warnings: CoreMsg[];
  quality: QualitySummary;
  summary_values: Record<string, Record<string, number>>;
  usability: { ok: boolean; reason: string | null };
  suggestions: MethodSuggestion[];
}

/** Какие колонки нужны методу. */
export interface ColumnSelection {
  value?: string;
  group?: string;
  x?: string;
  y?: string;
  before?: string;
  after?: string;
  block?: string;
  unit?: string;
  factor_a?: string;
  factor_b?: string;
  factor_c?: string;
  member?: string;
  row?: string;
  column?: string;
  whole_plot?: string;
  extraneous?: string;
}

const METHOD_MODULES: Record<string, string> = {
  describe_all: "core.stats.descriptive",
  mean_ci: "core.stats.descriptive",
  one_sample_t: "core.stats.hypothesis",
  two_sample_t: "core.stats.hypothesis",
  mann_whitney: "core.stats.hypothesis",
  paired_t: "core.stats.hypothesis",
  wilcoxon: "core.stats.hypothesis",
  one_way_anova: "core.stats.anova",
  tukey_hsd: "core.stats.anova",
  kruskal_wallis: "core.stats.anova",
  treatment_means: "core.stats.anova",
  anova_permutation_test: "core.stats.anova_checks",
  bartlett_test: "core.stats.anova_checks",
  anova_residuals: "core.stats.anova_checks",
  transformed_means: "core.stats.transforms",
  rcbd_anova: "core.stats.blocking",
  rcbd_subsampling_anova: "core.stats.blocking",
  factorial_anova: "core.stats.factorial",
  nested_anova: "core.stats.factorial",
  treatment_contrasts: "core.stats.contrasts",
  polynomial_contrasts: "core.stats.contrasts",
  mean_comparisons: "core.stats.multcomp",
  latin_square_anova: "core.stats.designs",
  split_plot_anova: "core.stats.designs",
  bibd_anova: "core.stats.designs",
  replication_power: "core.stats.power",
  equivalence_test: "core.stats.power",
  regression_details: "core.stats.linear_regression",
  inverse_prediction: "core.stats.linear_regression",
  regression_diagnostics: "core.stats.regression_checks",
  regression_cross_validation: "core.stats.regression_checks",
  multiple_regression: "core.stats.multiple_regression",
  regression_subsets: "core.stats.multiple_regression",
  stepwise_selection: "core.stats.multiple_regression",
  model_cross_validation: "core.stats.multiple_regression",
  regression_with_groups: "core.stats.regression_groups",
  unbalanced_anova: "core.stats.unbalanced",
  simple_linear_regression: "core.stats.regression",
  pearson_correlation: "core.stats.correlation",
  diagnose: "core.stats.diagnostics",
  normality_test: "core.stats.diagnostics",
};

/** Методы, которым нужна числовая колонка + колонка групп. */
const GROUPED_METHODS = new Set([
  "one_way_anova",
  "treatment_means",
  "anova_permutation_test",
  "bartlett_test",
  "anova_residuals",
  "transformed_means",
  "kruskal_wallis",
  "tukey_hsd",
  "two_sample_t",
  "mann_whitney",
]);

/** Методы, которым нужны две числовые колонки. */
const XY_METHODS = new Set([
  "simple_linear_regression",
  "pearson_correlation",
  "regression_details",
  "inverse_prediction",
  "regression_diagnostics",
  "regression_cross_validation",
]);

/** Блочные дизайны: отклик, варианты, блоки (и делянки) построчно. */
const BLOCKED_METHODS = new Set(["rcbd_anova", "rcbd_subsampling_anova"]);

/** Структуры вариантов (гл. 8): слоты передаются по имени, пустые — пропускаются. */
const STRUCTURED_METHODS = new Set([
  "factorial_anova",
  "nested_anova",
  "treatment_contrasts",
  "polynomial_contrasts",
  "mean_comparisons",
  "latin_square_anova",
  "split_plot_anova",
  "bibd_anova",
  "replication_power",
  "equivalence_test",
  "unbalanced_anova",
]);

/** Множественная регрессия (гл. 14): отклик y и все числовые колонки, выровненные по строкам. */
const MULTI_METHODS = new Set([
  "multiple_regression",
  "regression_subsets",
  "stepwise_selection",
  "model_cross_validation",
]);

/** Регрессия с группами (гл. 15): как multi, плюс колонка групп по строкам. */
const MULTI_GROUP_METHODS = new Set(["regression_with_groups"]);

/** Парные методы: две числовые колонки как связанные измерения (до/после). */
const PAIRED_METHODS = new Set(["paired_t", "wilcoxon"]);

/** Выполнить Python-код с переменными, вернуть разобранный JSON. */
async function runJson<T>(
  code: string,
  variables: Record<string, string> = {},
): Promise<T> {
  const pyodide = await initRuntime();
  try {
    const raw = await pyodide.runPythonAsync(code, {
      globals: pyodide.toPy(variables),
    });
    return JSON.parse(raw) as T;
  } catch (error) {
    throw new Error(formatPythonError(error));
  }
}

/**
 * Разобрать загруженный файл: детекция, парсинг, профили колонок.
 * Таблица сохраняется в Python и переиспользуется всеми расчётами.
 */
export async function parseFile(
  bytes: Uint8Array,
  overrides: Partial<{
    encoding: string;
    delimiter: string;
    decimal: string;
    has_header: boolean;
    type_overrides: Record<string, ColumnType>;
    lang: string;
  }> = {},
): Promise<ParsedTable> {
  const pyodide = await initRuntime();
  const lang = overrides.lang ?? "ru";
  // lang не входит в аргументы parse_file — вынимаем его отдельно.
  const parseOverrides = { ...overrides };
  delete parseOverrides.lang;

  const code = `
import builtins, json
from core.validation.parser import parse_file, suggest_methods

_overrides = json.loads(__overrides_json)
_raw = __file_bytes
if hasattr(_raw, "to_py"):
    _raw = _raw.to_py()
_table = parse_file(bytes(_raw), **_overrides)
builtins.__biostat_table = _table

_payload = _table.to_dict()
_payload["suggestions"] = suggest_methods(_table.profiles, ${JSON.stringify(lang)})
json.dumps(_payload, ensure_ascii=False)
`;
  try {
    const raw = await pyodide.runPythonAsync(code, {
      globals: pyodide.toPy({
        __overrides_json: JSON.stringify(parseOverrides),
        __file_bytes: bytes,
      }),
    });
    return JSON.parse(raw) as ParsedTable;
  } catch (error) {
    throw new Error(formatPythonError(error));
  }
}

/** Запустить статистический метод на колонках разобранной таблицы. */
export async function runMethod(
  method: string,
  columns: ColumnSelection,
  kwargs: Record<string, unknown> = {},
): Promise<StatResult> {
  const modulePath = METHOD_MODULES[method];
  if (!modulePath) throw new Error(`Неизвестный метод: ${method}`);

  const kind =
    method === "describe_all"
      ? "summary"
      : BLOCKED_METHODS.has(method)
        ? "blocked"
      : STRUCTURED_METHODS.has(method)
        ? "structured"
      : MULTI_METHODS.has(method)
        ? "multi"
      : MULTI_GROUP_METHODS.has(method)
        ? "multi_group"
      : GROUPED_METHODS.has(method)
        ? method === "two_sample_t" || method === "mann_whitney"
          ? "two_groups"
          : "many_groups"
        : XY_METHODS.has(method)
          ? "xy"
          : PAIRED_METHODS.has(method)
            ? "paired"
            : "single";

  const code = `
import builtins, json
from ${modulePath} import ${method} as _fn

_table = builtins.__biostat_table
_cols = json.loads(__columns_json)
_kwargs = json.loads(__kwargs_json)
_kind = ${JSON.stringify(kind)}

if _kind == "summary":
    _result = _fn(_table.numeric_columns_all(), **_kwargs)
elif _kind == "many_groups":
    _groups = _table.groups_by(_cols["value"], _cols["group"])
    if len(_groups) < 2:
        raise ValueError("Нужно минимум 2 группы в выбранной колонке")
    _result = _fn(*_groups.values(), labels=list(_groups.keys()), **_kwargs)
elif _kind == "two_groups":
    _groups = _table.groups_by(_cols["value"], _cols["group"])
    _keys = list(_groups.keys())
    if len(_keys) != 2:
        raise ValueError(
            f"Нужно ровно 2 группы, найдено {len(_keys)}: {', '.join(_keys)}"
        )
    _result = _fn(_groups[_keys[0]], _groups[_keys[1]], **_kwargs)
elif _kind == "blocked":
    _factor_keys = ["group", "block"] + (["unit"] if "unit" in _cols else [])
    _values, _factors = _table.factor_columns(
        _cols["value"], [_cols[k] for k in _factor_keys]
    )
    _result = _fn(_values, *_factors, **_kwargs)
elif _kind == "structured":
    _slots = [k for k, v in _cols.items() if k != "value" and v]
    _values, _factors = _table.factor_columns(_cols["value"], [_cols[k] for k in _slots])
    _result = _fn(
        _values, **dict(zip(_slots, _factors)),
        column_names={k: _cols[k] for k in _slots}, **_kwargs,
    )
elif _kind == "multi":
    _names = list(_table.numeric_columns_all())
    _all = dict(zip(_names, _table.aligned_numeric(_names)))
    _result = _fn(_all[_cols["y"]], _all, response_name=_cols["y"], **_kwargs)
elif _kind == "multi_group":
    # Колонка групп не участвует как x, даже если она числовая.
    _names = [n for n in _table.numeric_columns_all() if n != _cols["group"]]
    _all = dict(zip(_names, _table.aligned_numeric(_names)))
    _, (_grp,) = _table.factor_columns(_cols["y"], [_cols["group"]])
    _result = _fn(
        _all[_cols["y"]], _all, _grp, response_name=_cols["y"],
        column_names={"group": _cols["group"]}, **_kwargs,
    )
elif _kind == "xy":
    # Пары по строкам: пропуск → NaN, неполные строки удаляет метод.
    _x, _y = _table.aligned_numeric([_cols["x"], _cols["y"]])
    _result = _fn(_x, _y, **_kwargs)
elif _kind == "paired":
    _before, _after = _table.paired_columns(_cols["before"], _cols["after"])
    _result = _fn(_before, _after, **_kwargs)
else:
    _result = _fn(_table.numeric_column(_cols["value"]), **_kwargs)

builtins.__biostat_last = _result
json.dumps(_result.to_dict(), ensure_ascii=False)
`;
  return runJson<StatResult>(code, {
    __columns_json: JSON.stringify(columns),
    __kwargs_json: JSON.stringify(kwargs),
  });
}

/**
 * Запросить срез строк таблицы для прогрессивного предпросмотра.
 * Данные живут в Python, в браузер приходит только нужная порция.
 */
export async function getRows(
  start: number,
  count: number,
): Promise<string[][]> {
  const code = `
import builtins, json
_t = builtins.__biostat_table
json.dumps(_t.rows_slice(${Math.max(0, start)}, ${Math.max(0, count)}))
`;
  return runJson<string[][]>(code);
}

/** Диагностика допущений по последнему посчитанному результату. */
export async function diagnoseLast(): Promise<StatResult> {
  const code = `
import builtins, json
from core.stats.diagnostics import diagnose
json.dumps(diagnose(builtins.__biostat_last).to_dict(), ensure_ascii=False)
`;
  return runJson<StatResult>(code);
}

/** Какие колонки нужны данному методу — для построения формы выбора. */
export function requiredColumns(method: string): Array<keyof ColumnSelection> {
  if (GROUPED_METHODS.has(method)) return ["value", "group"];
  if (XY_METHODS.has(method)) return ["x", "y"];
  if (MULTI_METHODS.has(method)) return ["y"];
  if (MULTI_GROUP_METHODS.has(method)) return ["y", "group"];
  if (method === "rcbd_subsampling_anova") return ["value", "group", "block", "unit"];
  if (method === "factorial_anova") return ["value", "factor_a", "factor_b"];
  if (method === "nested_anova") return ["value", "group", "member"];
  if (method === "latin_square_anova") return ["value", "factor_a", "row", "column"];
  if (method === "split_plot_anova") return ["value", "factor_a", "factor_b"];
  if (method === "bibd_anova") return ["value", "group", "block"];
  if (method === "unbalanced_anova") return ["value", "factor_a"];
  if (STRUCTURED_METHODS.has(method)) return ["value", "group"];
  if (BLOCKED_METHODS.has(method)) return ["value", "group", "block"];
  return ["value"];
}

/** Данные графиков для последнего расчёта. Форма зависит от метода. */
export interface PlotBundle {
  histogram?: {
    bin_edges: number[];
    bin_centers: number[];
    counts: number[];
    normal_curve: { x: number[]; y: number[] } | null;
  };
  boxplot?: { labels: string[]; boxes: number[][]; outliers: number[][] };
  bar_means?: { labels: string[]; means: number[]; errors: number[] };
  scatter?: { points: number[][]; line: number[][] };
  qq?: { theoretical_quantiles: number[]; sample_quantiles: number[] };
  residual_vs_fitted?: { fitted: number[]; residuals: number[] };
}

/**
 * Построить данные графиков для последнего расчёта.
 *
 * Использует таблицу и результат из Python-глобальных. Набор графиков
 * подбирается под метод: гистограмма для describe, боксплот для ANOVA,
 * scatter+диагностика для регрессии.
 */
export async function getPlotData(
  method: string,
  columns: ColumnSelection,
): Promise<PlotBundle> {
  const code = `
import builtins, json
from core.stats import plotdata
from core.stats.diagnostics import diagnose

_t = builtins.__biostat_table
_res = builtins.__biostat_last
_cols = json.loads(__columns_json)
_method = ${JSON.stringify(method)}
_bundle = {}

if _method in ("normality_test", "mean_ci", "one_sample_t"):
    _vals = _t.numeric_column(_cols["value"])
    _bundle["histogram"] = plotdata.histogram_data(_vals)

elif _method in (
    "one_way_anova", "kruskal_wallis", "tukey_hsd", "treatment_means",
    "anova_permutation_test", "bartlett_test", "transformed_means",
):
    _groups = _t.groups_by(_cols["value"], _cols["group"])
    _bundle["boxplot"] = plotdata.boxplot_data(_groups)
    _bundle["bar_means"] = plotdata.bar_means_data(_groups)

elif _method in ("two_sample_t", "mann_whitney"):
    _groups = _t.groups_by(_cols["value"], _cols["group"])
    _bundle["boxplot"] = plotdata.boxplot_data(_groups)

elif _method == "simple_linear_regression":
    _x, _y = _t.aligned_numeric([_cols["x"], _cols["y"]])
    _slope = next(e["value"] for e in _res.to_dict()["estimates"] if e["name"] == "slope")
    _icept = next(e["value"] for e in _res.to_dict()["estimates"] if e["name"] == "intercept")
    _bundle["scatter"] = plotdata.scatter_regression_data(_x, _y, _slope, _icept)
    # Диагностика остатков — из того же результата.
    _diag = diagnose(_res).parameters.get("plots", {})
    if "qq" in _diag:
        _bundle["qq"] = _diag["qq"]
    if "residual_vs_fitted" in _diag:
        _bundle["residual_vs_fitted"] = _diag["residual_vs_fitted"]

elif _method in ("rcbd_anova", "rcbd_subsampling_anova") and "plots" in _res.parameters:
    # Варианты по группам + графики остатков модели делянок (гл. 5, 7).
    _bundle["boxplot"] = plotdata.boxplot_data(_t.groups_by(_cols["value"], _cols["group"]))
    _p = _res.parameters["plots"]
    _bundle["residual_vs_fitted"] = _p["residual_vs_fitted"]
    _bundle["qq"] = _p["qq"]

elif _method in (
    "factorial_anova", "nested_anova", "treatment_contrasts",
    "polynomial_contrasts", "mean_comparisons", "latin_square_anova",
    "split_plot_anova", "bibd_anova", "replication_power", "equivalence_test",
    "unbalanced_anova",
) and "plots" in _res.parameters:
    # Графики остатков модели со всеми членами (гл. 5, 8).
    _p = _res.parameters["plots"]
    _bundle["residual_vs_fitted"] = _p["residual_vs_fitted"]
    _bundle["qq"] = _p["qq"]

elif _method in (
    "regression_details", "inverse_prediction", "regression_diagnostics",
    "regression_cross_validation",
) and "points" in _res.parameters:
    # Данные (в шкале анализа) с подогнанной прямой; остатки — гл. 12.
    _pts, _line = _res.parameters["points"], _res.parameters["line"]
    _bundle["scatter"] = plotdata.scatter_regression_data(
        _pts["x"], _pts["y"], _line["slope"], _line["intercept"]
    )
    if "plots" in _res.parameters:
        _p = _res.parameters["plots"]
        if "residual_vs_fitted" in _p:
            _bundle["residual_vs_fitted"] = _p["residual_vs_fitted"]
            _bundle["qq"] = _p["qq"]
        if "deletion_residuals" in _p:
            _bundle["histogram"] = plotdata.histogram_data(_p["deletion_residuals"])

elif _method in ("multiple_regression", "regression_with_groups") and "plots" in _res.parameters:
    # Остатки МЛР по стандартизованным остаткам (гл. 14.6).
    _p = _res.parameters["plots"]
    _bundle["residual_vs_fitted"] = _p["residual_vs_fitted"]
    _bundle["qq"] = _p["qq"]
    _bundle["histogram"] = plotdata.histogram_data(_p["std_residuals"])

elif _method == "anova_residuals" and "plots" in _res.parameters:
    # Графики допущений по стандартизованным остаткам (гл. 5.2).
    _p = _res.parameters["plots"]
    _bundle["residual_vs_fitted"] = _p["residual_vs_fitted"]
    _bundle["qq"] = _p["qq"]
    _bundle["histogram"] = plotdata.histogram_data(_p["std_residuals"])

elif _method == "pearson_correlation" and "x" in _res.parameters:
    # Пары уже очищены от NaN в ядре; линия МНК лишь показывает линейный тренд.
    from scipy import stats as _st
    _x, _y = _res.parameters["x"], _res.parameters["y"]
    _lr = _st.linregress(_x, _y)
    _bundle["scatter"] = plotdata.scatter_regression_data(
        _x, _y, float(_lr.slope), float(_lr.intercept)
    )

json.dumps(_bundle, ensure_ascii=False)
`;
  return runJson<PlotBundle>(code, {
    __columns_json: JSON.stringify(columns),
  });
}

function formatPythonError(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error);
  const lines = text
    .trim()
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  // Ищем строку вида "SomeError: сообщение" — она несёт суть, а не хвост
  // traceback со ссылкой на документацию.
  const errorLine = [...lines]
    .reverse()
    .find((l) => /^[A-Za-z_.]*(Error|Exception):/.test(l));
  if (errorLine) {
    return errorLine.replace(/^[A-Za-z_.]*(Error|Exception):\s*/, "");
  }
  return lines.filter((l) => !l.startsWith("See http")).slice(-2).join("\n");
}
