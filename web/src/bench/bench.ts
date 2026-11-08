/// <reference types="vite/client" />
/**
 * Замер методов в браузере (шаг 1 задачи Web Worker): те же данные и
 * случаи, что `tools/bench_methods.py` в CPython, но через сам мост.
 * Страница для разработки: `web/bench.html`, в сборку не входит.
 */

import benchData from "../../../tools/bench_data.py?raw";
import { initRuntime } from "../pyodide/runtime";
import {
  getPlotData,
  parseFile,
  recognizeStructure,
  runMethod,
  type ColumnSelection,
} from "../pyodide/bridge";

type Case = [table: string, columns: ColumnSelection, kwargs: Record<string, unknown>];
interface Timing { first: number; best: number }
interface Row { run?: Timing; plots?: number; error?: string }
type Results = Record<string, Record<number, Row>>;

const params = new URLSearchParams(location.search);
const SIZES = (params.get("sizes") ?? "100,1000,5000").split(",").map(Number);
const ONLY = params.get("only")?.split(",") ?? null;
const log = (msg: string) => { document.getElementById("log")!.textContent += msg + "\n"; };

/** Первый вызов и лучший из повторов (суммарно < budget мс, до 5 раз). */
async function timed(fn: () => Promise<unknown>, budget = 300): Promise<Timing> {
  let t0 = performance.now();
  await fn();
  const first = performance.now() - t0;
  if (first > 2000) return { first, best: first };
  const times: number[] = [];
  while (times.length < 5 && times.reduce((a, b) => a + b, 0) < budget) {
    t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  return { first, best: Math.min(...times) };
}

/** Загрузка рантайма по стадиям, мс. */
async function startup(): Promise<Record<string, number>> {
  const marks: Record<string, number> = {};
  const t0 = performance.now();
  let last = t0;
  await initRuntime(({ stage }) => {
    const now = performance.now();
    marks[stage] = now - last;
    last = now;
  });
  marks.total = performance.now() - t0;
  return marks;
}

/** Модуль данных в ФС Pyodide; случаи и роли из него. */
async function loadCases(): Promise<{ cases: Record<string, Case>; roles: ColumnSelection }> {
  const py = await initRuntime();
  py.FS.writeFile("/home/pyodide/bench_data.py", benchData);
  return JSON.parse(py.runPython("import bench_data\nbench_data.cases_json()"));
}

async function tableBytes(name: string, n: number): Promise<Uint8Array> {
  const py = await initRuntime();
  const proxy = py.runPython(`bench_data.table_bytes(${JSON.stringify(name)}, ${n})`);
  const bytes = proxy.toJs() as Uint8Array;
  proxy.destroy();
  return bytes;
}

/** Один метод на текущей таблице: run (первый + лучший) и графики. */
async function benchMethod(method: string, c: Case): Promise<Row> {
  try {
    const run = await timed(() => runMethod(method, c[1], c[2]));
    const plots = (await timed(() => getPlotData(method, c[1]))).best;
    return { run, plots };
  } catch (error) {
    return { error: String(error).slice(0, 200) };
  }
}

/** Все размеры: разбор таблицы, распознаватели, методы этой таблицы. */
async function benchAll(cases: Record<string, Case>, roles: ColumnSelection): Promise<Results> {
  const out: Results = {};
  const put = (m: string, n: number, row: Row) => ((out[m] ??= {})[n] = row);
  const methods = Object.keys(cases).filter((m) => !ONLY || ONLY.includes(m));
  for (const n of SIZES) {
    for (const table of new Set(methods.map((m) => cases[m][0]))) {
      const bytes = await tableBytes(table, n);
      put(`parse:${table}`, n, { run: await timed(() => parseFile(bytes)) });
      if (table === "main") {
        for (const cls of ["structural", "computed"] as const)
          put(`recognize_${cls}`, n, { run: await timed(() => recognizeStructure(roles, cls)) });
      }
      for (const m of methods.filter((x) => cases[x][0] === table)) {
        log(`${n} строк: ${m}…`);
        await new Promise((r) => setTimeout(r)); // дать странице отрисовать лог
        put(m, n, await benchMethod(m, cases[m]));
      }
    }
  }
  return out;
}

const total = (r?: Row) => (r?.run ? r.run.best + (r.plots ?? 0) : NaN);

function render(results: Results): void {
  const head = `<tr><th>метод</th>${SIZES.map((n) => `<th>${n} стр., мс</th>`).join("")}<th>1-й вызов, мс</th></tr>`;
  const worst = (m: string) => Math.max(...SIZES.map((n) => total(results[m][n]) || 0));
  const rows = Object.keys(results).sort((a, b) => worst(b) - worst(a)).map((m) => {
    const cells = SIZES.map((n) => {
      const r = results[m][n];
      const ms = total(r);
      const cls = ms > 1000 ? "slow" : ms > 200 ? "mid" : "";
      return `<td class="${cls}">${r?.error ? "ошибка" : Number.isNaN(ms) ? "—" : ms.toFixed(0)}</td>`;
    });
    const first = results[m][SIZES[0]]?.run?.first;
    return `<tr><td>${m}</td>${cells.join("")}<td>${first?.toFixed(0) ?? "—"}</td></tr>`;
  });
  document.getElementById("out")!.innerHTML = head + rows.join("");
}

document.getElementById("start")!.addEventListener("click", async () => {
  log("Загрузка Pyodide…");
  const start = await startup();
  log(`Старт рантайма, мс: ${JSON.stringify(start, (_, v) => (typeof v === "number" ? Math.round(v) : v))}`);
  const { cases, roles } = await loadCases();
  const results = await benchAll(cases, roles);
  render(results);
  const blob = new Blob([JSON.stringify({
    startup: start, sizes: SIZES, results, userAgent: navigator.userAgent,
    cores: navigator.hardwareConcurrency,
  }, null, 1)], { type: "application/json" });
  const save = document.getElementById("save") as HTMLAnchorElement;
  save.href = URL.createObjectURL(blob);
  save.hidden = false;
  log("Готово.");
});
