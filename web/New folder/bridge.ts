/**
 * Мост между JS и расчётным ядром на Python.
 *
 * ЕДИНСТВЕННАЯ точка вызова Python из интерфейса. При добавлении других
 * бэкендов (WebGPU, cloud) меняется только этот файл — компоненты UI
 * продолжают вызывать те же функции.
 *
 * Обмен идёт через JSON, а не через прокси-объекты Pyodide: граница
 * остаётся чистой, JS получает обычные объекты, нет утечек памяти от
 * Python-прокси. Та же сигнатура подойдёт для облачного бэкенда.
 */

import { initRuntime } from "./runtime";

/** Уровень серьёзности предупреждения — зеркалит Severity в schema.py */
export type Severity = "info" | "warning" | "error";

export interface Warning {
  code: string;
  message: string;
  severity: Severity;
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

/** Карта: имя метода -> путь импорта в ядре. */
const METHOD_MODULES: Record<string, string> = {
  describe: "core.stats.descriptive",
  mean_ci: "core.stats.descriptive",
  one_sample_t: "core.stats.hypothesis",
  two_sample_t: "core.stats.hypothesis",
  paired_t: "core.stats.hypothesis",
  mann_whitney: "core.stats.hypothesis",
  wilcoxon: "core.stats.hypothesis",
  one_way_anova: "core.stats.anova",
  tukey_hsd: "core.stats.anova",
  kruskal_wallis: "core.stats.anova",
  simple_linear_regression: "core.stats.regression",
  predict: "core.stats.regression",
  diagnose: "core.stats.diagnostics",
  normality_test: "core.stats.diagnostics",
};

/**
 * Вызвать метод расчётного ядра.
 *
 * @param method - имя функции из METHOD_MODULES
 * @param args - позиционные аргументы (массивы данных и т.п.)
 * @param kwargs - именованные аргументы (level, alternative, labels…)
 */
export async function callMethod(
  method: string,
  args: unknown[] = [],
  kwargs: Record<string, unknown> = {},
): Promise<StatResult> {
  const modulePath = METHOD_MODULES[method];
  if (!modulePath) {
    throw new Error(`Неизвестный метод: ${method}`);
  }

  const pyodide = await initRuntime();

  const code = `
import json
from ${modulePath} import ${method} as _fn

_payload = json.loads(__args_json)
_result = _fn(*_payload["args"], **_payload["kwargs"])
json.dumps(_result.to_dict(), ensure_ascii=False)
`;

  try {
    const raw = await pyodide.runPythonAsync(code, {
      globals: pyodide.toPy({
        __args_json: JSON.stringify({ args, kwargs }),
      }),
    });
    return JSON.parse(raw) as StatResult;
  } catch (error) {
    throw new Error(formatPythonError(error));
  }
}

/**
 * Особый случай: диагностика принимает результат регрессии, а не массивы.
 * Пересобирает StatResult на стороне Python из переданного объекта.
 */
export async function diagnoseResult(source: StatResult): Promise<StatResult> {
  const pyodide = await initRuntime();

  const code = `
import json
from core.report.schema import StatResult
from core.stats.diagnostics import diagnose

_raw = json.loads(__source_json)
_src = StatResult(
    method=_raw["method"],
    parameters=_raw["parameters"],
    n=_raw["n"],
)
json.dumps(diagnose(_src).to_dict(), ensure_ascii=False)
`;

  try {
    const raw = await pyodide.runPythonAsync(code, {
      globals: pyodide.toPy({
        __source_json: JSON.stringify(source),
      }),
    });
    return JSON.parse(raw) as StatResult;
  } catch (error) {
    throw new Error(formatPythonError(error));
  }
}

/** Привести Python-исключение к читаемому сообщению. */
function formatPythonError(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error);
  // Питоновский traceback многословен — берём последнюю содержательную строку.
  const lines = text.trim().split("\n").filter(Boolean);
  const last = lines[lines.length - 1] ?? text;
  return last.replace(/^\w+Error:\s*/, "");
}

/** Список доступных методов — для построения UI. */
export function availableMethods(): string[] {
  return Object.keys(METHOD_MODULES);
}
