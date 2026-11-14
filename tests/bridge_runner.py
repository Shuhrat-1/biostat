"""Запуск методов через шаблоны моста (`web/src/pyodide/bridge.ts`), как в браузере.

Проверяет весь путь таблица → роли → метод, а не только функцию ядра.
Синтетика и набор ролей на каждый метод — `tools/bench_data.py`.
"""

from __future__ import annotations

import builtins
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bench_methods  # noqa: E402
from bench_data import CASES, bibd_table, latin_table, main_table  # noqa: E402

from core.validation.parser import parse_file  # noqa: E402

RESPONSES = ("y", "ylin", "ycurve", "ytrig", "pre", "post")
_TABLES = {"main": main_table, "latin": lambda n: latin_table(), "bibd": bibd_table}


def table(method: str, n: int = 120) -> dict[str, list]:
    """Синтетическая таблица метода как колонки одинаковой длины."""
    cols = _TABLES[CASES[method][0]](n)
    size = max(len(v) for v in cols.values() if hasattr(v, "__len__") and not isinstance(v, str))
    return {k: list(v) if hasattr(v, "__len__") and not isinstance(v, str) else [v] * size
            for k, v in cols.items()}


def to_csv(cols: dict[str, list]) -> bytes:
    """CSV; None и NaN — пустая ячейка."""
    def cell(v: Any) -> str:
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)

    rows = zip(*cols.values())
    return (",".join(cols) + "\n" + "\n".join(",".join(map(cell, r)) for r in rows) + "\n").encode()


def run(method: str, cols: dict[str, list], roles: dict | None = None,
        kwargs: dict | None = None) -> dict[str, Any]:
    """Результат метода (словарь, как его получает фронтенд)."""
    builtins.__biostat_table = parse_file(to_csv(cols))
    builtins.__biostat_last = None
    bench_methods.exec_method(method, CASES[method][1] if roles is None else roles,
                              CASES[method][2] if kwargs is None else kwargs)
    return builtins.__biostat_last.to_dict()


def scaled(cols: dict[str, list], factor: float) -> dict[str, list]:
    """Все отклики умножены на factor (смена единиц измерения)."""
    return {k: [x * factor for x in v] if k in RESPONSES else v for k, v in cols.items()}


def errors(result: dict) -> list[str]:
    return [w["code"] for w in result["warnings"] if w["severity"] == "error"]


def codes(result: dict) -> list[str]:
    return [w["code"] for w in result["warnings"]]


def p_values(obj: Any, key: str = "") -> list[float]:
    """Все P результата (ключи p, p_value, p_adjusted …), отсортированные."""
    out: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += p_values(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += p_values(v, key)
    elif isinstance(obj, float) and (key == "p" or key.startswith("p_")) and not math.isnan(obj):
        out.append(obj)
    return sorted(out)


def plots(method: str, roles: dict | None = None) -> dict[str, Any]:
    """Данные графиков для последнего результата `run` (шаблон getPlotData)."""
    lines = bench_methods.template("getPlotData").replace(
        "${JSON.stringify(method)}", json.dumps(method)).strip().splitlines()
    scope = {"__columns_json": json.dumps(CASES[method][1] if roles is None else roles)}
    exec("\n".join(lines[:-1]), scope)  # noqa: S102 — шаблон моста из репозитория
    return json.loads(eval(lines[-1], scope))  # noqa: S307
