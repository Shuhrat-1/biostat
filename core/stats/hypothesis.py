"""Проверка гипотез: t-тесты и непараметрические аналоги (гл. 2 книги).

Каждый тест проверяет свои допущения и предупреждает при нарушении,
подсказывая непараметрическую альтернативу. Все функции возвращают
``StatResult`` из единой модели отчёта.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from core.report.schema import Estimate, Severity, StatResult

# Порог p для тестов на нарушение допущений (нормальность, равенство дисперсий).
_ASSUMPTION_ALPHA = 0.05
# Диапазон n, в котором тест Шапиро-Уилка осмыслен.
_SHAPIRO_MIN = 3
_SHAPIRO_MAX = 5000
_ALTERNATIVES = ("two-sided", "less", "greater")


def _clean(data: np.ndarray | list[float]) -> np.ndarray:
    """Привести вход к 1-D массиву float без NaN."""
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def levene_p_value(*groups: np.ndarray) -> float | None:
    """p теста Левене или None, если тест вырожден.

    При группах из 2 наблюдений модули отклонений от медианы внутри
    группы равны, W = inf и p = 0 — ложный сигнал неравных дисперсий.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.levene(*groups)
    if not (np.isfinite(res.statistic) and np.isfinite(res.pvalue)):
        return None
    return float(res.pvalue)


def _choose_equal_var(
    arr_a: np.ndarray, arr_b: np.ndarray, equal_var: bool | None, result: StatResult
) -> bool:
    """Выбрать Стьюдента или Уэлча по Левене; при вырожденном Левене — Стьюдент.

    Объединённая дисперсия — вариант книги по умолчанию (гл. 2.4.2).
    """
    levene_p = levene_p_value(arr_a, arr_b)
    if levene_p is None:
        result.add_warning(
            "levene_undefined",
            "Тест Левене не определён (слишком мало наблюдений в группах); "
            "равенство дисперсий не проверено.",
            Severity.INFO,
        )
        return True if equal_var is None else equal_var
    result.statistics["levene_p"] = levene_p
    if equal_var is None:
        return levene_p >= _ASSUMPTION_ALPHA
    if equal_var and levene_p < _ASSUMPTION_ALPHA:
        result.add_warning(
            "unequal_variance",
            f"Тест Левене указывает на неравенство дисперсий "
            f"(p={levene_p:.4f}), но задан equal_var=True. "
            f"Рассмотри поправку Уэлча.",
            Severity.WARNING,
        )
    return equal_var


def _mean_diff_ci(res, confidence: float) -> tuple[float | None, float | None]:
    """ДИ разности средних того же варианта и направления, что и t.

    При одностороннем alternative одна граница бесконечна → None.
    """
    ci = res.confidence_interval(confidence_level=confidence)
    lo, hi = float(ci.low), float(ci.high)
    return (lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None)


def hedges_correction(df: float) -> float:
    """Точный множитель J Хеджеса: g = J · d [предл.: Hedges 1981].

    J = Γ(df/2) / (√(df/2) · Γ((df − 1)/2)); ≈ 1 − 3/(4·df − 1).
    """
    log_j = special.gammaln(df / 2) - special.gammaln((df - 1) / 2)
    return float(np.exp(log_j) / np.sqrt(df / 2))


def _cohens_d_pooled(arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[float, float]:
    """d Коэна по объединённому SD и g Хеджеса [Witte, стр. 262, ф. 14.5].

    Объединённое SD — всегда, независимо от варианта t (Стьюдент/Уэлч).
    """
    df = arr_a.size + arr_b.size - 2
    ss = np.sum((arr_a - arr_a.mean()) ** 2) + np.sum((arr_b - arr_b.mean()) ** 2)
    sd_pooled = float(np.sqrt(ss / df))
    if sd_pooled == 0:
        return float("nan"), float("nan")
    d = float((arr_a.mean() - arr_b.mean()) / sd_pooled)
    return d, d * hedges_correction(df)


def _check_normality(
    arr: np.ndarray, result: StatResult, label: str
) -> None:
    """Проверить нормальность (Шапиро-Уилк) и записать предупреждение."""
    if not _SHAPIRO_MIN <= arr.size <= _SHAPIRO_MAX:
        return
    p = float(stats.shapiro(arr).pvalue)
    result.statistics[f"shapiro_p_{label}"] = p
    if p < _ASSUMPTION_ALPHA:
        result.add_warning(
            "normality_violated",
            f"{label}: тест Шапиро-Уилка отклонил нормальность "
            f"(p={p:.4f}). Рассмотри непараметрический аналог.",
            Severity.WARNING,
        )


def one_sample_t(
    data: np.ndarray | list[float],
    popmean: float,
    alternative: str = "two-sided",
) -> StatResult:
    """Одновыборочный t-тест: среднее выборки против заданного значения."""
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")

    arr = _clean(data)
    result = StatResult(
        method="one_sample_t",
        n=arr.size,
        parameters={"popmean": popmean, "alternative": alternative},
    )
    if arr.size < 2:
        result.add_warning("n_too_small", "n < 2: тест невозможен", Severity.ERROR)
        return result

    res = stats.ttest_1samp(arr, popmean, alternative=alternative)
    mean = float(np.mean(arr))
    result.p_value = float(res.pvalue)
    result.statistics = {
        "t": float(res.statistic),
        "df": arr.size - 1,
        "mean": mean,
        "mean_diff": mean - popmean,
    }
    result.estimates.append(Estimate(name="mean", value=mean))
    _check_normality(arr, result, "sample")
    return result


def two_sample_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    equal_var: bool | None = None,
    alternative: str = "two-sided",
    confidence: float = 0.95,
) -> StatResult:
    """Двухвыборочный t-тест: сравнение средних двух групп.

    ``equal_var=None`` — автоматический выбор по тесту Левене:
    равные дисперсии → классический t-тест, иначе → поправка Уэлча.
    Явное значение переопределяет автоматику.

    Размер эффекта (Witte, гл. 14.8–14.9): ДИ разности того же варианта
    и направления, что и t; d Коэна по объединённому SD; g Хеджеса.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")
    if not 0 < confidence < 1:
        raise ValueError("confidence ∈ (0, 1)")

    arr_a, arr_b = _clean(a), _clean(b)
    result = StatResult(
        method="two_sample_t",
        n=arr_a.size + arr_b.size,
        parameters={"alternative": alternative},
    )
    if arr_a.size < 2 or arr_b.size < 2:
        result.add_warning(
            "n_too_small", "Каждая группа должна иметь n ≥ 2", Severity.ERROR
        )
        return result

    # Проверка равенства дисперсий (Левене) — определяет выбор варианта.
    equal_var = _choose_equal_var(arr_a, arr_b, equal_var, result)

    res = stats.ttest_ind(
        arr_a, arr_b, equal_var=equal_var, alternative=alternative
    )
    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    result.p_value = float(res.pvalue)
    result.parameters["equal_var"] = equal_var
    result.parameters["variant"] = "student" if equal_var else "welch"
    result.statistics.update(
        {
            "t": float(res.statistic),
            "df": float(res.df) if hasattr(res, "df") else arr_a.size + arr_b.size - 2,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_a - mean_b,
        }
    )
    _add_two_sample_effect(result, res, arr_a, arr_b, confidence)
    _check_normality(arr_a, result, "group_a")
    _check_normality(arr_b, result, "group_b")
    return result


def _add_two_sample_effect(
    result: StatResult,
    res,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    confidence: float,
) -> None:
    """Записать ДИ разности, d Коэна и g Хеджеса новыми ключами."""
    diff = float(arr_a.mean() - arr_b.mean())
    se = float(diff / res.statistic) if res.statistic else float("nan")
    lo, hi = _mean_diff_ci(res, confidence)
    d, g = _cohens_d_pooled(arr_a, arr_b)
    result.statistics.update(
        {
            "se_diff": se,
            "ci_level": confidence,
            "ci_lower": lo,
            "ci_upper": hi,
            "cohens_d": d,
            "hedges_g": g,
        }
    )
    result.estimates.append(
        Estimate(
            name="mean_diff", value=diff, ci_lower=lo, ci_upper=hi,
            ci_level=confidence, std_error=se,
        )
    )
    result.estimates.append(Estimate(name="cohens_d", value=d))
    result.estimates.append(Estimate(name="hedges_g", value=g))


def paired_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
) -> StatResult:
    """Парный t-тест: разности связанных измерений (до/после)."""
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")

    arr_a = np.asarray(a, dtype=float).ravel()
    arr_b = np.asarray(b, dtype=float).ravel()
    if arr_a.size != arr_b.size:
        raise ValueError("Парный тест требует выборки равной длины")

    # Удаляем пары, где хотя бы одно значение — NaN.
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    diff = arr_a - arr_b

    result = StatResult(
        method="paired_t",
        n=diff.size,
        parameters={"alternative": alternative},
    )
    if diff.size < 2:
        result.add_warning("n_too_small", "n < 2: тест невозможен", Severity.ERROR)
        return result

    res = stats.ttest_rel(arr_a, arr_b, alternative=alternative)
    mean_diff = float(np.mean(diff))
    result.p_value = float(res.pvalue)
    result.statistics = {
        "t": float(res.statistic),
        "df": diff.size - 1,
        "mean_diff": mean_diff,
    }
    result.estimates.append(Estimate(name="mean_diff", value=mean_diff))
    _check_normality(diff, result, "differences")
    return result


def mann_whitney(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
) -> StatResult:
    """Тест Манна-Уитни U — непараметрический аналог двухвыборочного t.

    Не требует нормальности; сравнивает распределения двух независимых групп.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")

    arr_a, arr_b = _clean(a), _clean(b)
    result = StatResult(
        method="mann_whitney",
        n=arr_a.size + arr_b.size,
        parameters={"alternative": alternative},
    )
    if arr_a.size < 1 or arr_b.size < 1:
        result.add_warning("empty_group", "Обе группы должны быть непусты", Severity.ERROR)
        return result

    res = stats.mannwhitneyu(arr_a, arr_b, alternative=alternative)
    result.p_value = float(res.pvalue)
    result.statistics = {
        "U": float(res.statistic),
        "median_a": float(np.median(arr_a)),
        "median_b": float(np.median(arr_b)),
    }
    return result


def wilcoxon(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
) -> StatResult:
    """Знаково-ранговый тест Уилкоксона — непараметрический аналог парного t."""
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")

    arr_a = np.asarray(a, dtype=float).ravel()
    arr_b = np.asarray(b, dtype=float).ravel()
    if arr_a.size != arr_b.size:
        raise ValueError("Тест Уилкоксона требует выборки равной длины")

    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[mask], arr_b[mask]

    result = StatResult(
        method="wilcoxon",
        n=arr_a.size,
        parameters={"alternative": alternative},
    )
    if arr_a.size < 1:
        result.add_warning("empty", "Нет данных после очистки", Severity.ERROR)
        return result

    res = stats.wilcoxon(arr_a, arr_b, alternative=alternative)
    result.p_value = float(res.pvalue)
    result.statistics = {
        "W": float(res.statistic),
        "median_diff": float(np.median(arr_a - arr_b)),
    }
    return result
