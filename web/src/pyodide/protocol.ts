/** Сообщения между основным потоком (runtime.ts) и Worker'ом с Pyodide (worker.ts). */

export type LoadStage =
  | "idle"
  | "loading-runtime"
  | "loading-packages"
  | "loading-core"
  | "ready"
  | "error";

/** Приоритет задачи: расчёт пользователя, подсказки карты, фоновый прогрев. */
export const PRIORITY = { user: 0, hint: 1 } as const;
export type Priority = (typeof PRIORITY)[keyof typeof PRIORITY];

export type WorkerRequest =
  | { type: "init"; coreUrl: string }
  | { type: "run"; id: number; priority: Priority; code: string;
      globals: Record<string, unknown> }
  | { type: "warm"; modules: string[] };

export type WorkerResponse =
  | { type: "progress"; stage: LoadStage; message: string }
  | { type: "init-error"; error: string }
  | { type: "result"; id: number; ok: true; value: unknown }
  | { type: "result"; id: number; ok: false; error: string };
