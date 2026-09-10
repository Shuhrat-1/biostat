"""Описательная статистика и доверительные интервалы (гл. 2 книги).

Все функции возвращают ``StatResult`` из единой модели отчёта.
Конвенция: выборочная дисперсия/SD с делением на (n-1), как в книге.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult

# Минимум наблюдений для осмысленной оценки разброса.
_MIN_N_SPREAD = 2


def _clean(data: np.ndarray | list[float]) -> np.ndarray:
    """Привести вход к 1-D массиву float без NaN.

    Возвращает очищенный массив. Пустой вход даёт пустой массив —
    вызывающая функция сама решает, как реагировать.
    """
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _column_metrics(arr: np.ndarray) -> dict[str, float]:
    """Полный набор описательных метрик для одной очищенной колонки.

    Все оценки разброса выборочные (ddof=1), чтобы std² = variance.
    Значения не округляются — округление на стороне интерфейса, иначе
    накапливается ошибка при дальнейшем использовании.
    """
    n = arr.size
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    metrics: dict[str, float] = {
        "n": float(n),
        "mean": mean,
        "median": median,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.ptp(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        # Доля и число нулей — полезно для разреженных данных.
        "zero_count": float(np.count_nonzero(arr == 0)),
        "zero_pct": float(np.mean(arr == 0) * 100),
        # MAD — медианное абсолютное отклонение, устойчивая мера разброса.
        "mad": float(np.median(np.abs(arr - median))),
        # Относительный IQR к медиане, %.
        "relative_iqr_pct": float(iqr / median * 100) if median != 0 else float("nan"),
    }

    if n >= _MIN_N_SPREAD:
        variance = float(np.var(arr, ddof=1))
        std = float(np.sqrt(variance))
        metrics.update(
            {
                "variance": variance,
                "std": std,
                "sem": float(stats.sem(arr)),
                "cv_pct": float(std / mean * 100) if mean != 0 else float("nan"),
                "skewness": float(stats.skew(arr)),
                "kurtosis": float(stats.kurtosis(arr)),  # excess (Fisher)
                # Выбросы по правилу 1.5*IQR (Тьюки).
                "outliers_iqr": float(
                    np.count_nonzero(
                        (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)
                    )
                ),
                # Выбросы по |z| > 3.
                "outliers_zscore": float(
                    np.count_nonzero(np.abs((arr - mean) / std) > 3)
                )
                if std > 0
                else 0.0,
            }
        )
    return metrics


def describe_all(
    columns: dict[str, np.ndarray | list[float]],
) -> StatResult:
    """Сводная описательная статистика по нескольким числовым колонкам.

    Возвращает результат, где ``parameters["summary"]`` — таблица метрик:
    для каждой колонки словарь метрик. Интерфейс рисует это как таблицу
    с метриками в строках и колонками датасета в столбцах.

    Пропущенные значения удаляются поколоночно; их число попадает в
    метрику ``missing``.
    """
    result = StatResult(method="describe_all")
    if not columns:
        result.add_warning("empty", "Нет числовых колонок", Severity.ERROR)
        return result

    summary: dict[str, dict[str, float]] = {}
    empty_columns: list[str] = []

    for name, data in columns.items():
        raw = np.asarray(data, dtype=float).ravel()
        arr = raw[~np.isnan(raw)]
        n_missing = raw.size - arr.size
        if arr.size == 0:
            empty_columns.append(name)
            summary[name] = {"missing": float(n_missing), "n": 0.0}
            continue
        metrics = _column_metrics(arr)
        metrics["missing"] = float(n_missing)
        summary[name] = metrics

    result.parameters["summary"] = summary
    result.parameters["columns"] = list(columns.keys())
    result.n = max((int(m.get("n", 0)) for m in summary.values()), default=0)

    if empty_columns:
        result.add_warning(
            "empty_columns",
            f"Колонки без данных: {', '.join(empty_columns)}",
            Severity.WARNING,
        )
    return result


def describe(data: np.ndarray | list[float]) -> StatResult:
    """Полная описательная статистика выборки.

    Считает центр, разброс, форму и квартили. Пропущенные (NaN)
    значения удаляются; их число фиксируется в предупреждении.
    """
    raw = np.asarray(data, dtype=float).ravel()
    arr = _clean(raw)
    n = arr.size
    n_missing = raw.size - n

    result = StatResult(method="describe", n=n)
    if n_missing:
        result.add_warning(
            "missing_dropped",
            f"Удалено пропущенных значений: {n_missing}",
            Severity.INFO,
        )
    if n == 0:
        result.add_warning("empty", "Нет данных после очистки", Severity.ERROR)
        return result

    result.statistics = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.ptp(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "iqr": float(stats.iqr(arr)),
    }
    if n >= _MIN_N_SPREAD:
        variance = float(np.var(arr, ddof=1))
        result.statistics.update(
            {
                "variance": variance,
                "std": float(np.sqrt(variance)),
                "sem": float(stats.sem(arr)),
                "cv": float(np.std(arr, ddof=1) / np.mean(arr))
                if np.mean(arr) != 0
                else float("nan"),
                "skewness": float(stats.skew(arr)),
                "kurtosis": float(stats.kurtosis(arr)),  # excess (Fisher)
            }
        )
    else:
        result.add_warning(
            "n_too_small",
            "n < 2: разброс и форма не вычисляются",
            Severity.WARNING,
        )
    return result


def mean_ci(
    data: np.ndarray | list[float], level: float = 0.95
) -> StatResult:
    """Доверительный интервал для среднего (t-распределение).

    Использует t-распределение с (n-1) степенями свободы — корректно
    для малых выборок, что типично для биологических экспериментов.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level должен быть в (0, 1)")

    arr = _clean(data)
    n = arr.size
    result = StatResult(
        method="mean_ci", n=n, parameters={"level": level}
    )
    if n < _MIN_N_SPREAD:
        result.add_warning(
            "n_too_small", "n < 2: CI не вычисляется", Severity.ERROR
        )
        return result

    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    half = sem * stats.t.ppf((1 + level) / 2, df=n - 1)
    result.estimates.append(
        Estimate(
            name="mean",
            value=mean,
            ci_lower=mean - half,
            ci_upper=mean + half,
            ci_level=level,
            std_error=sem,
        )
    )
    result.statistics = {"mean": mean, "sem": sem, "df": n - 1}
    return result
