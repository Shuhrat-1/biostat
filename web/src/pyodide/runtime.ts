/**
 * Инициализация Pyodide и загрузка расчётного ядра.
 *
 * Pyodide тяжёлый на первую загрузку (Python + numpy + scipy + statsmodels
 * в WASM), поэтому грузится один раз и кешируется браузером. Прогресс
 * сообщается через колбэк, чтобы UI не выглядел зависшим.
 */

import type { PyodideInterface } from "pyodide";

/** Пакеты, нужные расчётному ядру. Порядок не важен, ставятся вместе.
 *  charset-normalizer нужен модулю детекции кодировок (dialect.py). */
const REQUIRED_PACKAGES = [
  "numpy",
  "scipy",
  "pandas",
  "statsmodels",
  "charset-normalizer",
] as const;

/** Версия Pyodide — фиксируем явно для воспроизводимости результатов. */
const PYODIDE_VERSION = "0.26.4";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

export type LoadStage =
  | "idle"
  | "loading-runtime"
  | "loading-packages"
  | "loading-core"
  | "ready"
  | "error";

export interface LoadProgress {
  stage: LoadStage;
  message: string;
}

type ProgressCallback = (progress: LoadProgress) => void;

let pyodideInstance: PyodideInterface | null = null;
let loadPromise: Promise<PyodideInterface> | null = null;
// Подписчики на прогресс: при двойном монтировании (React StrictMode) или
// параллельных вызовах загрузка идёт одна, а слушателей может быть несколько.
const progressListeners = new Set<ProgressCallback>();
let lastProgress: LoadProgress = { stage: "idle", message: "Инициализация…" };

/**
 * Загрузить Pyodide, установить пакеты и распаковать ядро.
 *
 * Повторные вызовы не запускают вторую загрузку, а подписываются на уже
 * идущую. Это делает функцию безопасной при двойном монтировании в
 * StrictMode и при параллельных вызовах из разных компонентов.
 */
export async function initRuntime(
  onProgress?: ProgressCallback,
): Promise<PyodideInterface> {
  if (pyodideInstance) {
    onProgress?.({ stage: "ready", message: "Готово" });
    return pyodideInstance;
  }
  if (loadPromise) {
    // Загрузка уже идёт — подписываемся, а не начинаем заново.
    if (onProgress) {
      progressListeners.add(onProgress);
      onProgress(lastProgress);
    }
    return loadPromise;
  }

  loadPromise = load((p) => {
    lastProgress = p;
    progressListeners.forEach((fn) => fn(p));
  });
  if (onProgress) {
    progressListeners.add(onProgress);
    onProgress(lastProgress);
  }

  try {
    pyodideInstance = await loadPromise;
    return pyodideInstance;
  } catch (error) {
    loadPromise = null; // разрешаем повторную попытку после сбоя
    throw error;
  } finally {
    progressListeners.clear();
  }
}

async function load(onProgress?: ProgressCallback): Promise<PyodideInterface> {
  const report = (stage: LoadStage, message: string) =>
    onProgress?.({ stage, message });

  report("loading-runtime", "Загрузка Python-рантайма…");
  const { loadPyodide } = await import(
    /* @vite-ignore */ `${PYODIDE_CDN}pyodide.mjs`
  );
  const pyodide: PyodideInterface = await loadPyodide({ indexURL: PYODIDE_CDN });

  report("loading-packages", "Установка numpy, scipy, statsmodels…");
  await pyodide.loadPackage([...REQUIRED_PACKAGES]);

  report("loading-core", "Загрузка расчётного ядра…");
  await loadCore(pyodide);

  report("ready", "Готово");
  return pyodide;
}

/** Скачать core.zip и распаковать в виртуальную ФС Pyodide. */
async function loadCore(pyodide: PyodideInterface): Promise<void> {
  const response = await fetch("/core/core.zip");
  if (!response.ok) {
    throw new Error(`Не удалось загрузить ядро: HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  pyodide.unpackArchive(buffer, "zip");

  // Делаем корень доступным для импорта `core.stats...`
  await pyodide.runPythonAsync(`
import sys
if "" not in sys.path:
    sys.path.insert(0, "")
import core.stats.descriptive  # проверка, что ядро импортируется
`);
}

/** Признак готовности рантайма — для условного рендера UI. */
export function isReady(): boolean {
  return pyodideInstance !== null;
}
