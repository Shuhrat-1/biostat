/**
 * Клиент Web Worker'а с Pyodide (worker.ts): загрузка, очередь вызовов, отмена.
 *
 * Python исполняется в Worker'е, поэтому страница не замирает на долгом
 * расчёте. Отмена — остановка Worker'а и запуск нового: Pyodide и пакеты
 * берутся из кеша браузера (секунды), разобранная таблица восстанавливается
 * повтором последнего parseFile (вызов с restore: true).
 */

import {
  PRIORITY,
  type LoadStage,
  type Priority,
  type WorkerRequest,
  type WorkerResponse,
} from "./protocol";

export type { LoadStage } from "./protocol";

export interface LoadProgress {
  stage: LoadStage;
  message: string;
}

type ProgressCallback = (progress: LoadProgress) => void;

/** Расчёт остановлен пользователем — не ошибка, показывать как ошибку не нужно. */
export class CancelledError extends Error {
  constructor() {
    super("cancelled");
    this.name = "CancelledError";
  }
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

interface Call {
  code: string;
  globals: Record<string, unknown>;
}

const listeners = new Set<ProgressCallback>();
const pending = new Map<number, Pending>();
let worker: Worker | null = null;
let ready: Promise<void> | null = null;
let lastProgress: LoadProgress = { stage: "idle", message: "Инициализация…" };
let nextId = 1;
let restoreCall: Call | null = null;
let warmModules: string[] = [];

function emit(progress: LoadProgress): void {
  lastProgress = progress;
  listeners.forEach((fn) => fn(progress));
}

function send(msg: WorkerRequest): void {
  worker!.postMessage(msg);
}

/** Новый Worker; промис готовности — до стадии ready или ошибки загрузки. */
function startWorker(): Promise<void> {
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  return new Promise<void>((resolve, reject) => {
    worker!.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data;
      if (msg.type === "progress") {
        emit({ stage: msg.stage, message: msg.message });
        if (msg.stage === "ready") resolve();
      } else if (msg.type === "init-error") {
        emit({ stage: "error", message: msg.error });
        reject(new Error(msg.error));
      } else {
        settle(msg);
      }
    };
    send({ type: "init", coreUrl: new URL("/core/core.zip", location.href).href });
  });
}

function settle(msg: Extract<WorkerResponse, { type: "result" }>): void {
  const call = pending.get(msg.id);
  if (!call) return;
  pending.delete(msg.id);
  if (msg.ok) call.resolve(msg.value);
  else call.reject(new Error(msg.error));
}

/** После загрузки: вернуть таблицу (если была) и прогреть импорты фоном. */
async function afterReady(): Promise<void> {
  if (restoreCall) await post(restoreCall, PRIORITY.user);
  if (warmModules.length) send({ type: "warm", modules: warmModules });
}

/**
 * Запустить Worker и загрузить Python (один раз; повторные вызовы
 * подписываются на ту же загрузку). onProgress остаётся подписанным и
 * получает стадии и при перезапуске после отмены.
 */
export function initRuntime(onProgress?: ProgressCallback): Promise<void> {
  if (onProgress) {
    listeners.add(onProgress);
    onProgress(lastProgress);
  }
  if (!ready) {
    ready = startWorker().then(afterReady);
    ready.catch(() => (ready = null)); // повторная попытка после сбоя
  }
  return ready;
}

function post(call: Call, priority: Priority): Promise<unknown> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type: "run", id, priority, code: call.code, globals: call.globals });
  });
}

/**
 * Выполнить Python в Worker'е; результат — значение последнего выражения
 * (bytes → Uint8Array). restore: этот вызов задаёт состояние Python
 * (разобранную таблицу) и повторяется после перезапуска Worker'а.
 */
export async function runPython<T>(
  code: string,
  globals: Record<string, unknown> = {},
  options: { priority?: Priority; restore?: boolean } = {},
): Promise<T> {
  await initRuntime();
  const call = { code, globals };
  const value = await post(call, options.priority ?? PRIORITY.user);
  if (options.restore) restoreCall = call;
  return value as T;
}

/** Остановить текущий расчёт: все ждущие вызовы → CancelledError, Worker заново. */
export function cancelRun(): Promise<void> {
  worker?.terminate();
  pending.forEach((call) => call.reject(new CancelledError()));
  pending.clear();
  ready = startWorker().then(afterReady);
  return ready;
}

/** Модули ядра для фонового импорта после загрузки (первый расчёт без ожидания). */
export function setWarmModules(modules: string[]): void {
  warmModules = [...new Set(modules)];
}

/** Признак готовности рантайма — для условного рендера UI. */
export function isReady(): boolean {
  return lastProgress.stage === "ready";
}
