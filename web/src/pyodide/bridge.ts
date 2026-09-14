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
  kind: "number" | "select" | "float";
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
  simple_linear_regression: "core.stats.regression",
  diagnose: "core.stats.diagnostics",
  normality_test: "core.stats.diagnostics",
};

/** Методы, которым нужна числовая колонка + колонка групп. */
const GROUPED_METHODS = new Set([
  "one_way_anova",
  "kruskal_wallis",
  "tukey_hsd",
  "two_sample_t",
  "mann_whitney",
]);

/** Методы, которым нужны две числовые колонки. */
const XY_METHODS = new Set(["simple_linear_regression"]);

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
elif _kind == "xy":
    _result = _fn(
        _table.numeric_column(_cols["x"]),
        _table.numeric_column(_cols["y"]),
        **_kwargs,
    )
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

elif _method in ("one_way_anova", "kruskal_wallis", "tukey_hsd"):
    _groups = _t.groups_by(_cols["value"], _cols["group"])
    _bundle["boxplot"] = plotdata.boxplot_data(_groups)
    _bundle["bar_means"] = plotdata.bar_means_data(_groups)

elif _method in ("two_sample_t", "mann_whitney"):
    _groups = _t.groups_by(_cols["value"], _cols["group"])
    _bundle["boxplot"] = plotdata.boxplot_data(_groups)

elif _method == "simple_linear_regression":
    _x = _t.numeric_column(_cols["x"])
    _y = _t.numeric_column(_cols["y"])
    _slope = next(e["value"] for e in _res.to_dict()["estimates"] if e["name"] == "slope")
    _icept = next(e["value"] for e in _res.to_dict()["estimates"] if e["name"] == "intercept")
    _bundle["scatter"] = plotdata.scatter_regression_data(_x, _y, _slope, _icept)
    # Диагностика остатков — из того же результата.
    _diag = diagnose(_res).parameters.get("plots", {})
    if "qq" in _diag:
        _bundle["qq"] = _diag["qq"]
    if "residual_vs_fitted" in _diag:
        _bundle["residual_vs_fitted"] = _diag["residual_vs_fitted"]

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
