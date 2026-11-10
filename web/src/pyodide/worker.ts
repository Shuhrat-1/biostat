/// <reference lib="webworker" />
/**
 * Web Worker с Pyodide: весь Python исполняется здесь, а не в основном потоке.
 *
 * Страница не замирает на долгом расчёте, а отменить его можно, остановив
 * Worker (см. runtime.ts). Задачи выполняются по одной, по приоритету:
 * расчёт пользователя (0) → подсказки карты (1) → фоновый прогрев импортов (2).
 */

import type { PyodideInterface } from "pyodide";
import type { LoadStage, WorkerRequest, WorkerResponse } from "./protocol";

/** Пакеты, нужные расчётному ядру. charset-normalizer — детекция кодировок. */
const REQUIRED_PACKAGES = ["numpy", "scipy", "pandas", "statsmodels", "charset-normalizer"];

/** Версия Pyodide — фиксируем явно для воспроизводимости результатов. */
const PYODIDE_VERSION = "0.26.4";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

interface Task {
  id: number;
  priority: number;
  code: string;
  globals: Record<string, unknown>;
}

const scope = self as unknown as DedicatedWorkerGlobalScope;
const queue: Task[] = [];
let pyodide: PyodideInterface | null = null;
let running = false;

const post = (msg: WorkerResponse) => scope.postMessage(msg);
const progress = (stage: LoadStage, message: string) =>
  post({ type: "progress", stage, message });

/** Загрузить рантайм, пакеты и ядро (core.zip с того же сайта). */
async function load(coreUrl: string): Promise<void> {
  progress("loading-runtime", "Загрузка Python-рантайма…");
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_CDN}pyodide.mjs`);
  const py: PyodideInterface = await loadPyodide({ indexURL: PYODIDE_CDN });

  progress("loading-packages", "Установка numpy, scipy, statsmodels…");
  await py.loadPackage(REQUIRED_PACKAGES);

  progress("loading-core", "Загрузка расчётного ядра…");
  const response = await fetch(coreUrl);
  if (!response.ok) throw new Error(`Не удалось загрузить ядро: HTTP ${response.status}`);
  py.unpackArchive(await response.arrayBuffer(), "zip");
  await py.runPythonAsync(`
import sys
if "" not in sys.path:
    sys.path.insert(0, "")
import core.stats.descriptive  # проверка, что ядро импортируется
`);
  pyodide = py;
  progress("ready", "Готово");
}

/** Результат Python в структурно клонируемое значение (bytes → Uint8Array). */
function toPlain(value: unknown): unknown {
  const proxy = value as { toJs?: () => unknown; destroy?: () => void } | null;
  if (!proxy || typeof proxy.toJs !== "function") return value;
  const plain = proxy.toJs();
  proxy.destroy?.();
  return plain;
}

async function execute(task: Task): Promise<void> {
  const py = pyodide!;
  try {
    const raw = await py.runPythonAsync(task.code, { globals: py.toPy(task.globals) });
    if (task.id >= 0) post({ type: "result", id: task.id, ok: true, value: toPlain(raw) });
  } catch (error) {
    if (task.id >= 0) {
      post({ type: "result", id: task.id, ok: false,
             error: error instanceof Error ? error.message : String(error) });
    }
  }
}

/** Следующая задача с наименьшим приоритетом; по одной, без перекрытия. */
async function pump(): Promise<void> {
  if (running || !pyodide || !queue.length) return;
  running = true;
  queue.sort((a, b) => a.priority - b.priority || a.id - b.id);
  const task = queue.shift()!;
  await execute(task);
  running = false;
  // Через макрозадачу: пришедшие тем временем сообщения встанут в очередь раньше.
  setTimeout(pump, 0);
}

let warmId = -1; // прогрев — без ответа, id < 0

scope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  if (msg.type === "init") {
    try {
      await load(msg.coreUrl);
      pump();
    } catch (error) {
      post({ type: "init-error", error: error instanceof Error ? error.message : String(error) });
    }
  } else if (msg.type === "run") {
    queue.push({ id: msg.id, priority: msg.priority, code: msg.code, globals: msg.globals });
    pump();
  } else if (msg.type === "warm") {
    for (const mod of msg.modules) {
      queue.push({ id: warmId--, priority: 2, code: `import ${mod}`, globals: {} });
    }
    pump();
  }
};
