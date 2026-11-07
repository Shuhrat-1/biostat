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


_SPEARMAN_SMALL_N = 10  # ниже — P по t-приближению грубый


def spearman_correlation(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alternative: str = "two-sided",
    ci_level: float = 0.95,
) -> StatResult:
    """Ранговая корреляция Спирмена ρ — r Пирсона по рангам (Witte, стр. 119).

    Ранги — средние при совпадениях; P — t на n − 2 df (scipy.spearmanr).
    ДИ — Фишер с SE = √(1.06/(n − 3)) [предл.: Fieller, Hartley & Pearson 1957].
    ΣD² и формула 1 − 6ΣD²/(n(n² − 1)) — для сверки с ручным счётом (точна без
    совпадений). Для диаграммы рассеяния в parameters x, y — ранги.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level должен быть в (0, 1)")
    cx, cy, n_dropped = _clean_xy(x, y)
    result = StatResult(method="spearman_correlation", n=cx.size,
                        parameters={"alternative": alternative, "ci_level": ci_level})
    if n_dropped:
        result.add_warning("missing_dropped", f"Удалено пар с пропусками: {n_dropped}",
                           Severity.INFO)
    if not _guard(cx, cy, result):
        return result
    rx, ry = stats.rankdata(cx), stats.rankdata(cy)
    res = stats.spearmanr(cx, cy, alternative=alternative)
    rho = float(res.statistic)
    result.p_value = float(res.pvalue)
    _fill_spearman(result, rx, ry, rho)
    _spearman_notes(result, cx.size, bool(np.unique(cx).size < cx.size
                                          or np.unique(cy).size < cy.size))
    return result


def _fill_spearman(result: StatResult, rx: np.ndarray, ry: np.ndarray, rho: float) -> None:
    """ρ с ДИ Фишера (n ≥ 4), t, df, ΣD², формула без совпадений; ранги для графика."""
    n, level = rx.size, result.parameters["ci_level"]
    if abs(abs(rho) - 1) < 1e-12:  # ранги совпали: убрать ошибку округления scipy
        rho = float(np.sign(rho))
    d2 = float(np.sum((rx - ry) ** 2))
    result.statistics = {"rho": rho, "rho_squared": rho**2, "df": n - 2, "sum_d_squared": d2,
                         "rho_no_ties_formula": 1 - 6 * d2 / (n * (n**2 - 1))}
    if abs(rho) < 1.0:
        result.statistics["t"] = rho * float(np.sqrt((n - 2) / (1 - rho**2)))
    lo = hi = None
    if n >= 4 and abs(rho) < 1.0:
        half = stats.norm.ppf(0.5 + level / 2) * np.sqrt(1.06 / (n - 3))
        lo, hi = float(np.tanh(np.arctanh(rho) - half)), float(np.tanh(np.arctanh(rho) + half))
        result.statistics.update({"ci_lower": lo, "ci_upper": hi})
    result.estimates.append(Estimate(name="rho", value=rho, ci_lower=lo, ci_upper=hi,
                                     ci_level=level))
    result.parameters.update({"x": rx.tolist(), "y": ry.tolist(), "scatter_scale": "ranks"})


def _spearman_notes(result: StatResult, n: int, ties: bool) -> None:
    """Малое n, совпадения, смысл ρ (монотонная связь)."""
    if n < _SPEARMAN_SMALL_N:
        result.add_warning(
            "spearman_small_n",
            f"Пар меньше {_SPEARMAN_SMALL_N}: P по t-приближению грубый.",
            Severity.INFO, params={"n": n})
    if ties:
        result.add_warning(
            "spearman_ties",
            "Есть совпадения: ρ посчитан как r Пирсона по средним рангам; формула "
            "1 − 6ΣD²/(n(n² − 1)) в этом случае лишь приближение.", Severity.INFO)
    result.add_warning(
        "correlation_monotonic_only",
        "ρ измеряет монотонную связь (растёт ли y вместе с x), не обязательно линейную; "
        "корреляция не означает причинности.", Severity.INFO)
