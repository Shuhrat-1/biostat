"""Корреляция двух числовых переменных (гл. 2.5 книги).

Коэффициент Пирсона и t-тест H0: ρ = 0 через scipy.stats.pearsonr
(проверенный эталон для r, p и доверительного интервала по Фишеру).
Книга: t = r·√(N−2)/√(1−r²) на N−2 df (стр. 37–38); тест точен при
двумерной нормальности, иначе приближённый (стр. 38).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.hypothesis import _check_normality
from core.stats.regression import _clean_xy

_MIN_N = 3  # при n = 2 r = ±1 всегда, df = 0
_ALTERNATIVES = ("two-sided", "less", "greater")


def _guard(cx: np.ndarray, cy: np.ndarray, result: StatResult) -> bool:
    """Проверить n и вариацию; при проблеме записать ошибку, вернуть False."""
    if cx.size < _MIN_N:
        result.add_warning("n_too_small", "Нужно ≥ 3 пар", Severity.ERROR)
        return False
    if np.ptp(cx) == 0 or np.ptp(cy) == 0:
        result.add_warning(
            "no_variation_corr",
            "Одна из переменных не варьирует — корреляция не определена",
            Severity.ERROR,
        )
        return False
    return True


def pearson_correlation(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alternative: str = "two-sided",
    ci_level: float = 0.95,
) -> StatResult:
    """Коэффициент корреляции Пирсона r и t-тест H0: ρ = 0 (гл. 2.5).

    Пары с пропуском в любой переменной удаляются. Доверительный
    интервал r — преобразование Фишера (scipy), в книге не приводится.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level должен быть в (0, 1)")

    cx, cy, n_dropped = _clean_xy(x, y)
    n = cx.size
    result = StatResult(
        method="pearson_correlation",
        n=n,
        parameters={"alternative": alternative, "ci_level": ci_level},
    )
    if n_dropped:
        result.add_warning(
            "missing_dropped", f"Удалено пар с пропусками: {n_dropped}", Severity.INFO
        )
    if not _guard(cx, cy, result):
        return result

    res = stats.pearsonr(cx, cy, alternative=alternative)
    ci = res.confidence_interval(confidence_level=ci_level)
    _fill_result(result, cx, cy, float(res.statistic), (float(ci.low), float(ci.high)))
    result.p_value = float(res.pvalue)
    _check_normality(cx, result, "x")
    _check_normality(cy, result, "y")
    result.add_warning(
        "correlation_linear_only",
        "r измеряет только линейную связь; r ≈ 0 не исключает нелинейной "
        "зависимости, а корреляция не означает причинности. Смотри диаграмму "
        "рассеяния.",
        Severity.INFO,
    )
    return result


def _fill_result(
    result: StatResult,
    cx: np.ndarray,
    cy: np.ndarray,
    r: float,
    ci: tuple[float, float],
) -> None:
    """Записать r с ДИ, t-статистику, df и ковариацию в результат."""
    n = cx.size
    ci_level = result.parameters["ci_level"]
    result.estimates.append(
        Estimate(name="r", value=r, ci_lower=ci[0], ci_upper=ci[1], ci_level=ci_level)
    )
    result.statistics = {
        "r": r,
        "r_squared": r**2,
        "df": n - 2,
        "covariance": float(np.cov(cx, cy, ddof=1)[0, 1]),
    }
    # t = r·√(N−2)/√(1−r²) (стр. 37). При |r| = 1 t бесконечна, а inf
    # ломает JSON.parse на фронте, поэтому t не пишем, а предупреждаем.
    if abs(r) < 1.0:
        result.statistics["t"] = r * float(np.sqrt((n - 2) / (1 - r**2)))
    else:
        result.add_warning(
            "perfect_correlation",
            "Точки лежат на одной прямой (|r| = 1): t не определена",
            Severity.INFO,
        )
    result.parameters["x"] = cx.tolist()
    result.parameters["y"] = cy.tolist()
