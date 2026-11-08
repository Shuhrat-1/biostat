"""Замер времени методов BioStat через шаблоны моста (шаг 1 задачи Web Worker).

Исполняет в CPython тот же Python-код, что мост (`web/src/pyodide/bridge.ts`)
отдаёт Pyodide: parseFile, recognizeStructure, runMethod, getPlotData.
Данные и случаи — `tools/bench_data.py` (общие со страницей `web/bench.html`). Каждый метод идёт в отдельном
процессе с таймаутом: зависший или слишком долгий метод не роняет замер.

В Pyodide (WebAssembly) расчёт обычно в 3–5 раз медленнее CPython — оценка
браузера в отчёте = CPython × PYODIDE_FACTOR; точные числа даст замер в
браузере (шаг 2).

Запуск из корня репозитория:
    python tools/bench_methods.py                      # 100, 1000, 5000 строк
    python tools/bench_methods.py --sizes 100 1000 --only glmm_analysis mixed_model
    python tools/bench_methods.py --json bench.json --timeout 60
"""

from __future__ import annotations

import argparse
import builtins
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_data import CASES, RCBD_ROLES, table_bytes  # noqa: E402

PYODIDE_FACTOR = 4.0  # [предл.] середина 3–5×; уточнить замером в браузере
BRIDGE = (ROOT / "web/src/pyodide/bridge.ts").read_text(encoding="utf-8")
MODULES = dict(re.findall(r'^\s+(\w+): "(core\.stats\.\w+)",', BRIDGE, re.M))
SETS = {name: set(re.findall(r'"(\w+)"', body)) for name, body in
        re.findall(r"const (\w+_METHODS) = new Set\(\[(.*?)\]\)", BRIDGE, re.S)}


# ── Шаблоны моста ────────────────────────────────────────────────────────


def template(fn: str) -> str:
    """Python-код функции моста ``fn``: шаблон `const code` или аргумент runJson."""
    body = BRIDGE[BRIDGE.index(f"export async function {fn}"):]
    end = body.find("\nexport ", 1)
    body = body[:end] if end > 0 else body
    anchor = "const code = `" if "const code = `" in body else "runJson<"
    start = body.index("`", body.index(anchor)) + 1
    return body[start:body.index("`", start)]


def kind_of(method: str) -> str:
    """Тот же `kind`, что вычисляет мост по множествам *_METHODS."""
    if method == "describe_all":
        return "summary"
    if method == "sampling_scheme_power":
        return "planning"
    for name, kind in (("BLOCKED_METHODS", "blocked"), ("STRUCTURED_METHODS", "structured"),
                       ("CATEGORICAL_METHODS", "categorical"), ("MULTI_METHODS", "multi"),
                       ("MULTI_GROUP_METHODS", "multi_group")):
        if method in SETS.get(name, ()):
            return kind
    if method in SETS.get("GROUPED_METHODS", ()):
        return "two_groups" if method in ("two_sample_t", "mann_whitney") else "many_groups"
    if method in SETS.get("XY_METHODS", ()):
        return "xy"
    if method in SETS.get("PAIRED_METHODS", ()):
        return "paired"
    return "single"


# ── Исполнение ───────────────────────────────────────────────────────────

RECOGNIZE = {"recognize_structural": "structural", "recognize_computed": "computed"}
def timed(fn, budget: float = 0.3, tries: int = 5) -> tuple[float, float]:
    """(первый вызов, лучший из повторов). Модуль метода уже импортирован;
    первый вызов ловит ленивые импорты внутри (statsmodels и т. п.).
    Медленный первый вызов (> 2 с) не повторяем."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > 2:
        return first, first
    times = []
    while len(times) < tries and sum(times) < budget:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return first, min(times)


def exec_parse(data: bytes) -> None:
    code = template("parseFile").replace("${JSON.stringify(lang)}", '"ru"')
    exec(code, {"__overrides_json": "{}", "__file_bytes": data})


def exec_method(method: str, cols: dict, kwargs: dict) -> None:
    code = (template("runMethod").replace("${modulePath}", MODULES[method])
            .replace("${method}", method)
            .replace("${JSON.stringify(kind)}", json.dumps(kind_of(method))))
    exec(code, {"__columns_json": json.dumps(cols), "__kwargs_json": json.dumps(kwargs)})


def result_errors() -> list[str]:
    """Коды предупреждений уровня error последнего результата: быстрый, но
    отказной расчёт (метод отказал по входу) не должен сойти за быстрый."""
    last = getattr(builtins, "__biostat_last", None)
    if last is None:
        return []
    return [w.code for w in last.warnings if getattr(w.severity, "value", w.severity) == "error"]


def exec_plots(method: str, cols: dict) -> None:
    lines = template("getPlotData").replace(
        "${JSON.stringify(method)}", json.dumps(method)).strip().splitlines()
    scope = {"__columns_json": json.dumps(cols)}
    exec("\n".join(lines[:-1]), scope)
    eval(lines[-1], scope)


def exec_recognize(cls: str) -> None:
    exec(template("recognizeStructure"),
         {"__roles_json": json.dumps(RCBD_ROLES), "__rec_cls": cls})


def measure(method: str, n: int) -> dict:
    """Секунды CPython на n строк: run (+ cold — первый вызов), plots."""
    from core.validation.parser import parse_file

    name, cols, kwargs = CASES.get(method, ("main", {}, {}))
    data = table_bytes(name, n)
    builtins.__biostat_table = parse_file(data)
    out: dict = {"rows": builtins.__biostat_table.n_rows}
    if method == "parse":
        job = lambda: exec_parse(data)  # noqa: E731
    elif method in RECOGNIZE:
        job = lambda: exec_recognize(RECOGNIZE[method])  # noqa: E731
    else:
        job = lambda: exec_method(method, cols, kwargs)  # noqa: E731
    out["cold"], out["run"] = timed(job)
    if method in CASES:
        out["result_errors"] = result_errors()
        out["plots"] = timed(lambda: exec_plots(method, cols))[1]
    return out


def warm_import(method: str) -> float:
    """Секунды на импорт модуля метода (в браузере — разово, при первом вызове)."""
    import importlib

    t0 = time.perf_counter()
    importlib.import_module("core.validation.parser")
    if method in MODULES:
        importlib.import_module(MODULES[method])
    elif method in RECOGNIZE:
        importlib.import_module("core.validation.recognizers_computed")
    return time.perf_counter() - t0


def _child(method: str, sizes: list[int], queue: mp.Queue) -> None:
    imported = warm_import(method)
    for n in sizes:
        try:
            queue.put((n, {**measure(method, n), "import": imported}))
        except Exception as exc:  # noqa: BLE001 — замер, а не ядро
            queue.put((n, {"error": f"{type(exc).__name__}: {exc}"[:200]}))
            return


def bench(method: str, sizes: list[int], timeout: float) -> dict[int, dict]:
    """Замер метода по размерам в отдельном процессе; таймаут — на каждый размер."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(method, sizes, queue), daemon=True)
    proc.start()
    out: dict[int, dict] = {}
    for n in sizes:
        res = _wait(queue, proc, timeout + 30)  # +30 с на импорт ядра и данные
        out[n] = res
        if "run" not in res:
            break
    proc.terminate()
    return out


def _wait(queue: mp.Queue, proc: mp.Process, limit: float) -> dict:
    """Результат очередного размера, {"timeout"} или {"error": "crashed"}."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            return queue.get(timeout=0.5)[1]
        except Exception:  # noqa: BLE001 — queue.Empty
            if not proc.is_alive():
                return {"error": f"процесс упал (код {proc.exitcode})"}
    return {"timeout": limit - 30}


# ── Отчёт ────────────────────────────────────────────────────────────────

BANDS = ((0.2, "мгновенно"), (1.0, "заметно"), (10.0, "нужен Worker"),
         (float("inf"), "алгоритм / Worker"))


def band(seconds: float) -> str:
    """Класс по оценке браузера (CPython × PYODIDE_FACTOR)."""
    browser = seconds * PYODIDE_FACTOR
    return next((label for limit, label in BANDS if browser < limit), BANDS[-1][1])


def cell(res: dict) -> str:
    if res.get("result_errors"):
        return "отказ: " + ", ".join(res["result_errors"])
    if "run" in res:
        return f"{res['run'] + res.get('plots', 0):.3f}"
    if "timeout" in res:
        return f"> {res['timeout']:.0f}"
    return "ошибка"


def report(results: dict[str, dict[int, dict]], sizes: list[int]) -> str:
    """Markdown: run + plots (с, лучший повтор) по размерам, холодный старт, класс."""
    head = ("| метод | " + " | ".join(f"{n} стр." for n in sizes)
            + " | импорт + 1-й вызов | ≈ браузер (макс.) | класс |")
    lines = [head, "|" + "---|" * (len(sizes) + 4)]
    for method, by_n in sorted(results.items(), key=lambda kv: -_worst(kv[1])):
        worst = _worst(by_n)
        first = next(iter(by_n.values()), {})
        lines.append(f"| {method} | " + " | ".join(cell(by_n.get(n, {})) for n in sizes)
                     + f" | {first.get('import', 0) + first.get('cold', 0):.2f} | "
                     + f"{worst * PYODIDE_FACTOR:.2f} | {band(worst)} |")
    errors = [f"- {m}: {r['error']}" for m, by_n in results.items()
              for r in by_n.values() if "error" in r]
    return "\n".join(lines + (["", "Ошибки:"] + errors if errors else []))


def _worst(by_n: dict[int, dict]) -> float:
    """Худшее время (с); таймаут и ошибка — бесконечность / таймаут."""
    vals = [r["run"] + r.get("plots", 0) if "run" in r else r.get("timeout", float("inf"))
            for r in by_n.values()]
    return max(vals, default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    ap.add_argument("--only", nargs="+", default=None, help="только эти методы")
    ap.add_argument("--timeout", type=float, default=120, help="с на один размер")
    ap.add_argument("--json", type=Path, default=None, help="сохранить сырые числа")
    args = ap.parse_args()
    methods = args.only or ["parse", *RECOGNIZE, *CASES]
    results = {}
    for method in methods:
        results[method] = bench(method, args.sizes, args.timeout)
        print(f"{method:28s}", {n: cell(r) for n, r in results[method].items()}, flush=True)
    print("\n" + report(results, args.sizes))
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
