"""Таблица частот одной колонки (Witte & Witte, Statistics, 11th ed., гл. 2–4).

``frequency_table`` — частоты, относительные частоты и накопленные проценты:

- качественные (номинальные) данные: f и доли по классам, мода (стр. 31, 55);
  накопленные частоты не считаются (стр. 32);
- порядковые качественные (порядок уровней задан): накопленные проценты как
  приближённые процентильные ранги, медианный класс — первый класс с
  накопленным процентом ≥ 50 (стр. 31–32, 55);
- количественные с < 20 возможными значениями — несгруппированная таблица со
  всеми значениями от минимума до максимума с шагом единицы измерения
  (стр. 23); иначе — сгруппированная по правилам стр. 25–27: около 10 классов
  равной ширины, удобный интервал, нижняя граница кратна интервалу, пустые
  классы перечислены, реальные границы ±½ единицы измерения (стр. 26).

Удобный интервал — ближайший к размах/10 из 1, 2, 5 × 10ᵏ (книга допускает
1…10, «особенно 5 и 10»). ДИ долей — интервал Уилсона [предл.: Wilson 1927;
statsmodels proportion_confint(method="wilson")]; в Witte формулы нет.
"""

from __future__ import annotations

import math
import re

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import _to_float

_KINDS = ("auto", "categorical", "numeric")
_UNGROUPED_MAX = 20  # возможных значений меньше ~20 → без группировки (стр. 23)
_NICE = (1.0, 2.0, 5.0)
_MANY_CLASSES = 30
_MAX_DECIMALS = 6
_MAX_ROWS = 200


# --- Разбор колонки ---------------------------------------------------------------


def _clean(response: list, count: list | None) -> tuple[list, np.ndarray, int]:
    """Непустые значения и их веса (частоты строк); пропуск в любой колонке — строка вон."""
    weights = np.ones(len(response)) if count is None else _to_float(count)
    keep = [i for i, v in enumerate(response)
            if v not in ("", None) and not (isinstance(v, float) and math.isnan(v))
            and not math.isnan(weights[i])]
    return [response[i] for i in keep], weights[keep], len(response) - len(keep)


def _numeric(values: list) -> np.ndarray | None:
    """Все значения — числа (десятичная запятая допустима), иначе None."""
    x = _to_float(values)
    return None if np.isnan(x).any() else x


def _decimals(value: object) -> int:
    """Знаков после запятой в записи значения (для строк — как записано: 150.0 → 1)."""
    if isinstance(value, str):
        match = re.search(r"[.,](\d+)$", value.strip())
        return min(len(match.group(1)), _MAX_DECIMALS) if match else 0
    for d in range(_MAX_DECIMALS + 1):
        if abs(float(value) * 10 ** d - round(float(value) * 10 ** d)) < 1e-9:
            return d
    return _MAX_DECIMALS


def infer_unit(values: list) -> float:
    """Единица измерения — наименьшая разность, с которой записаны значения (стр. 24)."""
    return 10.0 ** -max(_decimals(v) for v in values)


def convenient_width(raw: float) -> float:
    """Ближайший к raw удобный интервал 1, 2, 5 × 10ᵏ (стр. 25, правило 5; стр. 27)."""
    k = math.floor(math.log10(raw))
    candidates = [c * 10.0 ** e for e in (k - 1, k, k + 1) for c in _NICE]
    return min(candidates, key=lambda c: (abs(c - raw), -c))


# --- Классы -----------------------------------------------------------------------


def _fmt(value: float, unit: float) -> str:
    """Граница класса с числом знаков единицы измерения (130.0–139.9 при единице 0.1)."""
    digits = max(0, -round(math.log10(unit)))
    return f"{value:.{digits}f}"


def numeric_classes(x: np.ndarray, unit: float, width: float) -> list[dict]:
    """Классы от кратной width границы ниже минимума до класса с максимумом (стр. 27)."""
    digits = max(0, -round(math.log10(unit))) + 2
    lower = math.floor(round(x.min() / width, 9)) * width
    classes = []
    while lower <= x.max() + 1e-9 * max(1.0, abs(lower)):
        upper = lower + width - unit
        label = _fmt(lower, unit) if width == unit else f"{_fmt(lower, unit)}–{_fmt(upper, unit)}"
        bounds = {"lower": lower, "upper": upper, "real_lower": lower - unit / 2,
                  "real_upper": upper + unit / 2, "midpoint": (lower + upper) / 2}
        classes.append({"class": label, **{k: round(v, digits) for k, v in bounds.items()}})
        lower = round(lower + width, digits)
    return classes


def _numeric_counts(x: np.ndarray, w: np.ndarray, classes: list[dict]) -> list[float]:
    """Частоты классов по реальным границам: каждое наблюдение ровно в одном классе."""
    edges = np.array([c["real_lower"] for c in classes] + [classes[-1]["real_upper"]])
    idx = np.searchsorted(edges, x, side="right") - 1
    return [float(w[idx == j].sum()) for j in range(len(classes))]


def _grouping(x: np.ndarray, unit: float, n_classes: int, class_width: float) -> tuple[str, float]:
    """Вид таблицы и ширина класса: без группировки при < 20 значениях, иначе по правилам."""
    possible = round((x.max() - x.min()) / unit) + 1
    if class_width <= 0 and possible < _UNGROUPED_MAX:
        return "ungrouped", unit
    width = class_width if class_width > 0 else convenient_width((x.max() - x.min()) / n_classes)
    return "grouped", max(round(width / unit) * unit, unit)


# --- Сводки -----------------------------------------------------------------------


def _finish(rows: list[dict], ordered: bool, alpha: float) -> None:
    """Доли с ДИ Уилсона и (для упорядоченных) накопленные частоты и проценты."""
    n = sum(r["f"] for r in rows)
    cumulative = 0.0
    for r in rows:
        lo, hi = proportion_confint(r["f"], n, alpha=alpha, method="wilson")
        r.update({"proportion": r["f"] / n, "percent": 100 * r["f"] / n,
                  "ci_lower": float(lo), "ci_upper": float(hi)})
        if ordered:
            cumulative += r["f"]
            r.update({"cumulative_f": cumulative, "cumulative_percent": 100 * cumulative / n})


def summaries(rows: list[dict], ordered: bool) -> dict:
    """Мода (все классы с максимальной f), медианный класс, разброс качественных (стр. 78)."""
    top = max(r["f"] for r in rows)
    observed = [r["class"] for r in rows if r["f"] > 0]
    out = {"mode": [r["class"] for r in rows if r["f"] == top], "mode_f": top,
           "max_share": top / sum(r["f"] for r in rows), "observed_classes": len(observed)}
    if ordered:
        out["median_class"] = next(r["class"] for r in rows
                                   if r["cumulative_percent"] >= 50 - 1e-9)
        out.update({"lowest_class": observed[0], "highest_class": observed[-1]})
    return out


def _order(text: str) -> list[str]:
    """Порядок уровней от меньшего к большему: через «;» (или через запятую, если «;» нет)."""
    sep = ";" if ";" in text else ","
    return [p.strip() for p in text.split(sep) if p.strip()]


# --- Метод каталога -----------------------------------------------------------------


def frequency_table(
    response: list,
    count: list | None = None,
    kind: str = "auto",
    level_order: str = "",
    n_classes: int = 10,
    class_width: float = 0.0,
    unit: float = 0.0,
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Частоты, доли, накопленные проценты, мода и медианный класс одной колонки (Witte гл. 2–4)."""
    if kind not in _KINDS or not 0 < alpha < 1 or not 2 <= int(n_classes) <= 50:
        raise ValueError(f"kind ∈ {_KINDS}, 0 < alpha < 1, n_classes 2–50")
    values, weights, dropped = _clean(response, count)
    name = (column_names or {}).get("response", "X")
    result = StatResult(method="frequency_table", n=len(values),
                        parameters={"alpha": alpha, "column": name})
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    if not _check(values, weights, result):
        return result
    x = _numeric(values) if kind != "categorical" else None
    if kind == "numeric" and x is None:
        result.add_warning("freq_not_numeric", "Колонка содержит нечисловые значения.",
                           Severity.ERROR)
        return result
    if x is not None and not level_order.strip():
        rows = _numeric_table(values, x, weights, (int(n_classes), class_width, unit), result)
    else:
        rows = _categorical_table(values, weights, _order(level_order), result)
    if rows:
        _report(rows, result)
    return result


def _check(values: list, weights: np.ndarray, result: StatResult) -> bool:
    """Есть данные; частоты неотрицательны и не все нулевые."""
    if not values or (weights < 0).any() or weights.sum() <= 0:
        result.add_warning("freq_empty", "Нет значений или частоты отрицательны / нулевые.",
                           Severity.ERROR)
        return False
    return True


def _numeric_table(values: list, x: np.ndarray, w: np.ndarray, options: tuple,
                   result: StatResult) -> list[dict]:
    """Количественная колонка: несгруппированная или сгруппированная таблица (стр. 23–27)."""
    n_classes, class_width, unit = options
    unit = unit if unit > 0 else infer_unit(values)
    table_kind, width = _grouping(x, unit, n_classes, class_width)
    if (x.max() - x.min()) / width > _MAX_ROWS:
        result.add_warning("freq_too_many_classes", f"Больше {_MAX_ROWS} классов: увеличьте "
                           "ширину класса или единицу измерения.", Severity.ERROR,
                           params={"k": _MAX_ROWS})
        return []
    classes = numeric_classes(x, unit, width)
    for c, f in zip(classes, _numeric_counts(x, w, classes)):
        c["f"] = f
    result.parameters.update({"kind": table_kind, "ordered": True, "unit": unit,
                              "class_width": width})
    _finish(classes, True, result.parameters["alpha"])
    return classes


def _categorical_table(values: list, w: np.ndarray, order: list[str],
                       result: StatResult) -> list[dict]:
    """Категории: в заданном порядке (порядковые) или по убыванию частоты (номинальные)."""
    labels = [str(v).strip() for v in values]
    counts: dict[str, float] = {}
    for lab, wt in zip(labels, w):
        counts[lab] = counts.get(lab, 0.0) + float(wt)
    unknown = sorted(set(counts) - set(order)) if order else []
    if unknown:
        result.add_warning("freq_unknown_levels", "Значения вне заданного порядка уровней: "
                           + ", ".join(unknown[:10]), Severity.ERROR,
                           params={"levels": ", ".join(unknown[:10])})
        return []
    keys = order or sorted(counts, key=lambda k: (-counts[k], k))
    rows = [{"class": k, "f": counts.get(k, 0.0)} for k in keys]
    result.parameters.update({"kind": "categorical", "ordered": bool(order)})
    if not order:
        result.add_warning("freq_nominal", "Порядок уровней не задан: данные считаются "
                           "номинальными, накопленные частоты и медиана не считаются.",
                           Severity.INFO)
    if len(rows) > _MANY_CLASSES:
        result.add_warning("freq_many_classes", f"Категорий {len(rows)}: колонка похожа на "
                           "идентификаторы или количественную.", params={"k": len(rows)})
    _finish(rows, bool(order), result.parameters["alpha"])
    return rows


def _report(rows: list[dict], result: StatResult) -> None:
    """Таблица, сводки и оценки долей с ДИ Уилсона."""
    p = result.parameters
    sums = summaries(rows, p["ordered"])
    p.update({"table": rows, **sums})
    result.statistics.update({"n": float(sum(r["f"] for r in rows)),
                              "n_classes": float(len(rows)), "mode_f": sums["mode_f"],
                              "max_share": sums["max_share"]})
    for r in rows:
        result.estimates.append(Estimate(name=r["class"], value=r["proportion"],
                                         ci_lower=r["ci_lower"], ci_upper=r["ci_upper"],
                                         ci_level=1 - p["alpha"]))
