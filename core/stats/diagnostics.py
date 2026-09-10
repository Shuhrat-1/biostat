"""Диагностика допущений модели через анализ остатков (гл. 5 книги).

Приоритетный модуль: именно проверка допущений отличает серьёзный
инструмент от калькулятора p-value. Проходит через всю прикладную
статистику книги.

Разделение ответственности: ядро НЕ рисует графики (это работа фронтенда
через Plotly), а отдаёт готовые числовые ряды. Четыре графика с обложки
книги: residual vs fitted, Q-Q plot, гистограмма остатков, scale-location.

Модуль принимает уже посчитанный результат регрессии — не пересчитывает
модель. Остатки берутся из ``StatResult.parameters``, куда их положил
модуль regression.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

from core.report.schema import Severity, StatResult

_ASSUMPTION_ALPHA = 0.05
_SHAPIRO_MIN = 3
_SHAPIRO_MAX = 5000
# Границы «нормы» для Дарбина-Уотсона: ~2 нет автокорреляции, <1 или >3 тревога.
_DW_LOW = 1.0
_DW_HIGH = 3.0


def _require_residuals(source: StatResult) -> tuple[np.ndarray, np.ndarray]:
    """Извлечь остатки и предсказанные значения из результата регрессии.

    Бросает ValueError, если источник не содержит сохранённых остатков.
    """
    p = source.parameters
    if "residuals" not in p or "fitted" not in p:
        raise ValueError(
            "Источник не содержит остатков. Передай результат "
            "simple_linear_regression."
        )
    return (
        np.asarray(p["residuals"], dtype=float),
        np.asarray(p["fitted"], dtype=float),
    )


def diagnose(source: StatResult) -> StatResult:
    """Полная диагностика допущений регрессии по её остаткам.

    Проверяет нормальность (Шапиро-Уилк), независимость (Дарбин-Уотсон)
    и готовит данные для четырёх диагностических графиков. Каждое
    нарушение фиксируется предупреждением.
    """
    resid, fitted = _require_residuals(source)
    n = resid.size
    result = StatResult(
        method="diagnose",
        n=n,
        parameters={"source_method": source.method},
    )
    if n < _SHAPIRO_MIN:
        result.add_warning("n_too_small", "Слишком мало остатков", Severity.ERROR)
        return result

    _test_normality(resid, result)
    _test_independence(resid, result)
    result.parameters["plots"] = _build_plot_data(resid, fitted)
    return result


def _test_normality(resid: np.ndarray, result: StatResult) -> None:
    """Тест Шапиро-Уилка на нормальность остатков."""
    if _SHAPIRO_MIN <= resid.size <= _SHAPIRO_MAX:
        p = float(stats.shapiro(resid).pvalue)
        result.statistics["shapiro_p"] = p
        if p < _ASSUMPTION_ALPHA:
            result.add_warning(
                "normality_violated",
                f"Остатки не прошли тест на нормальность (p={p:.4f}). "
                f"Рассмотри трансформацию отклика (гл. 6) или "
                f"непараметрический метод.",
                Severity.WARNING,
            )


def _test_independence(resid: np.ndarray, result: StatResult) -> None:
    """Тест Дарбина-Уотсона на автокорреляцию остатков."""
    dw = float(durbin_watson(resid))
    result.statistics["durbin_watson"] = dw
    if dw < _DW_LOW or dw > _DW_HIGH:
        direction = "положительная" if dw < _DW_LOW else "отрицательная"
        result.add_warning(
            "autocorrelation",
            f"Дарбин-Уотсон = {dw:.3f}: возможна {direction} автокорреляция "
            f"остатков (ожидается ~2). Проверь порядок наблюдений.",
            Severity.WARNING,
        )


def _build_plot_data(resid: np.ndarray, fitted: np.ndarray) -> dict:
    """Собрать числовые ряды для четырёх диагностических графиков.

    Ядро отдаёт данные, фронтенд рисует. Стандартизация остатков — для
    сопоставимости шкал между графиками.
    """
    std = resid.std(ddof=1)
    std_resid = resid / std if std > 0 else np.zeros_like(resid)

    # Q-Q: теоретические (osm) и упорядоченные выборочные (osr) квантили.
    osm, osr = stats.probplot(resid, dist="norm", fit=False)

    return {
        # 1. Residual vs Fitted — линейность и случайность разброса.
        "residual_vs_fitted": {
            "fitted": fitted.tolist(),
            "residuals": resid.tolist(),
        },
        # 2. Q-Q plot — нормальность.
        "qq": {
            "theoretical_quantiles": np.asarray(osm).tolist(),
            "sample_quantiles": np.asarray(osr).tolist(),
        },
        # 3. Гистограмма остатков — форма распределения.
        "histogram": {"residuals": resid.tolist()},
        # 4. Scale-Location — гомоскедастичность (корень |станд. остатка|).
        "scale_location": {
            "fitted": fitted.tolist(),
            "sqrt_abs_std_residuals": np.sqrt(np.abs(std_resid)).tolist(),
        },
    }


def normality_test(data: np.ndarray | list[float]) -> StatResult:
    """Отдельный тест нормальности произвольной выборки (Шапиро-Уилк).

    Полезен вне контекста регрессии — например, проверить сырые данные
    перед выбором параметрического или непараметрического теста.
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    result = StatResult(method="normality_test", n=arr.size)
    if not _SHAPIRO_MIN <= arr.size <= _SHAPIRO_MAX:
        result.add_warning(
            "n_out_of_range",
            f"Шапиро-Уилк осмыслен при {_SHAPIRO_MIN} ≤ n ≤ {_SHAPIRO_MAX}",
            Severity.ERROR,
        )
        return result

    res = stats.shapiro(arr)
    result.p_value = float(res.pvalue)
    result.statistics = {
        "W": float(res.statistic),
        "skewness": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr)),
    }
    if res.pvalue < _ASSUMPTION_ALPHA:
        result.add_warning(
            "normality_violated",
            f"Данные не прошли тест на нормальность (p={res.pvalue:.4f}).",
            Severity.WARNING,
        )
    return result
