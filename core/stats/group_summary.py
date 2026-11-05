"""Описательная статистика по группам (Witte & Witte, Statistics, 11th ed., гл. 3–4).

``describe_by_group`` — для каждой группы и для всех данных вместе те же
метрики, что ``describe_all`` (среднее, медиана, SD и дисперсия с делителем
n − 1, SE, квартили, IQR, min/max, размах, CV, асимметрия), плюс сумма
квадратов SS (формула 4.3–4.4, стр. 69). Среднее и медиана показываются рядом
без вердикта (при асимметрии Witte советует сообщать обе, стр. 53).

Мода — только для дискретных или повторяющихся значений (все значения целые
или различных значений не больше половины n): для непрерывных измерений,
где значения не повторяются, она бессмысленна (стр. 48, 55). Если все значения
встречаются по одному разу, моды нет.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from core.report.schema import Severity, StatResult
from core.stats.ancova import _to_float
from core.stats.descriptive import _column_metrics
from core.stats.factorial import _natural_order

_TOTAL = "Итого"


def _frame(values: list[float], group: list[str]) -> tuple[pd.DataFrame, int]:
    """Строки с числом и непустой меткой группы."""
    frame = pd.DataFrame({"y": _to_float(values), "g": pd.Series(group, dtype=object)})
    frame["g"] = frame["g"].replace("", np.nan)
    clean = frame.dropna().reset_index(drop=True)
    clean["g"] = clean["g"].astype(str)
    return clean, len(frame) - len(clean)


def mode_metrics(arr: np.ndarray) -> dict:
    """Мода(ы) и их частота для дискретных/повторяющихся значений; иначе пусто."""
    levels, counts = np.unique(arr, return_counts=True)
    discrete = bool(np.all(arr == np.round(arr)))
    if not (discrete or levels.size <= arr.size / 2) or counts.max() < 2:
        return {}
    return {"mode": [float(v) for v in levels[counts == counts.max()]],
            "mode_f": float(counts.max())}


def group_metrics(arr: np.ndarray) -> dict[str, float]:
    """Метрики describe_all + сумма квадратов SS."""
    with warnings.catch_warnings():  # постоянная группа: асимметрия не определена (NaN)
        warnings.simplefilter("ignore", RuntimeWarning)
        metrics = _column_metrics(arr)
    metrics["ss"] = float(((arr - arr.mean()) ** 2).sum())
    return metrics


def describe_by_group(
    values: list[float],
    group: list[str],
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """n, среднее, SD, SE, медиана, квартили, IQR, min/max, CV по группам и в целом (Witte гл. 3–4)."""
    frame, dropped = _frame(values, group)
    cn = column_names or {}
    result = StatResult(method="describe_by_group", n=len(frame),
                        parameters={"value": cn.get("value", "y"), "group": cn.get("group", "g")})
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    if frame.empty:
        result.add_warning("empty", "Нет данных после очистки", Severity.ERROR)
        return result
    levels = _natural_order(frame["g"])
    total = _TOTAL if _TOTAL not in levels else f"{_TOTAL} (все группы)"
    arrays = {lev: frame.loc[frame["g"] == lev, "y"].to_numpy() for lev in levels}
    arrays[total] = frame["y"].to_numpy()
    summary = {k: group_metrics(a) for k, a in arrays.items()}
    modes = {k: m for k, a in arrays.items() if (m := mode_metrics(a))}
    result.parameters.update({"summary": summary, "columns": [*levels, total],
                              "groups": levels, "total_label": total, "modes": modes})
    result.statistics.update({"n_groups": float(len(levels)), "n": float(len(frame))})
    if any(m["n"] < 2 for m in summary.values()):
        result.add_warning("group_n_too_small", "В части групп n < 2: SD, SE и форма не "
                           "считаются.", Severity.WARNING)
    return result
