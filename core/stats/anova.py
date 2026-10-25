"""Однофакторный дисперсионный анализ и множественные сравнения (гл. 3, 8).

Однофакторный ANOVA не зависит от конвенции sum of squares (Type I/II/III
совпадают для одного фактора) — расхождений между пакетами нет. Проблема
конвенций всплывёт позже, на факторных схемах.

Пост-хок: Tukey HSD (контроль familywise error для всех попарных сравнений).
Все функции возвращают ``StatResult`` из единой модели отчёта.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from core.report.schema import Estimate, Severity, StatResult

_ASSUMPTION_ALPHA = 0.05
_SHAPIRO_MIN = 3
_SHAPIRO_MAX = 5000


def _clean_group(data: np.ndarray | list[float]) -> np.ndarray:
    """Привести группу к 1-D массиву float без NaN."""
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _pooled_residuals(groups: list[np.ndarray]) -> np.ndarray:
    """Остатки внутри групп (значение минус среднее своей группы)."""
    return np.concatenate([g - g.mean() for g in groups])


def one_way_anova(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
) -> StatResult:
    """Однофакторный ANOVA: сравнение средних нескольких групп.

    Проверяет гомогенность дисперсий (Левене) и нормальность остатков
    (Шапиро-Уилк), предупреждая при нарушении. Заполняет полную таблицу
    ANOVA (SS, df, MS, F) в statistics.
    """
    clean = [_clean_group(g) for g in groups]
    k = len(clean)
    if k < 2:
        result = StatResult(method="one_way_anova")
        result.add_warning("too_few_groups", "Нужно ≥ 2 групп", Severity.ERROR)
        return result

    if labels is None:
        labels = [f"g{i + 1}" for i in range(k)]
    elif len(labels) != k:
        raise ValueError("Число меток не совпадает с числом групп")

    sizes = [g.size for g in clean]
    n_total = int(sum(sizes))
    result = StatResult(
        method="one_way_anova",
        n=n_total,
        parameters={"labels": labels, "group_sizes": sizes},
    )
    if any(s < 2 for s in sizes):
        result.add_warning(
            "n_too_small", "Каждая группа должна иметь n ≥ 2", Severity.ERROR
        )
        return result

    # F-тест.
    f_res = stats.f_oneway(*clean)
    df_between = k - 1
    df_within = n_total - k

    # Таблица ANOVA вручную для полной прозрачности.
    grand_mean = np.concatenate(clean).mean()
    ss_between = float(sum(g.size * (g.mean() - grand_mean) ** 2 for g in clean))
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in clean))
    ss_total = ss_between + ss_within
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    result.p_value = float(f_res.pvalue)
    result.statistics = {
        "F": float(f_res.statistic),
        "df_between": df_between,
        "df_within": df_within,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "eta_squared": ss_between / ss_total if ss_total > 0 else float("nan"),
    }
    for lab, g in zip(labels, clean):
        result.estimates.append(Estimate(name=f"mean_{lab}", value=float(g.mean())))

    _check_anova_assumptions(clean, result)
    return result


def _check_anova_assumptions(groups: list[np.ndarray], result: StatResult) -> None:
    """Проверить гомогенность дисперсий и нормальность остатков."""
    levene_p = float(stats.levene(*groups).pvalue)
    result.statistics["levene_p"] = levene_p
    if levene_p < _ASSUMPTION_ALPHA:
        result.add_warning(
            "variance_heterogeneity",
            f"Тест Левене указывает на неравенство дисперсий (p={levene_p:.4f}). "
            f"Рассмотри тест Уэлча ANOVA или Краскела-Уоллиса.",
            Severity.WARNING,
        )

    residuals = _pooled_residuals(groups)
    if _SHAPIRO_MIN <= residuals.size <= _SHAPIRO_MAX:
        shapiro_p = float(stats.shapiro(residuals).pvalue)
        result.statistics["shapiro_p_residuals"] = shapiro_p
        if shapiro_p < _ASSUMPTION_ALPHA:
            result.add_warning(
                "normality_violated",
                f"Остатки не прошли тест на нормальность (p={shapiro_p:.4f}). "
                f"Рассмотри Краскела-Уоллиса.",
                Severity.WARNING,
            )


def tukey_hsd(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Пост-хок Tukey HSD: все попарные сравнения с контролем FWER.

    Применяется после значимого ANOVA. Каждое сравнение даёт разность
    средних, скорректированный p и доверительный интервал.

    Конвенция знака (как в statsmodels): ``mean_diff = mean(group2) -
    mean(group1)``. Положительное значение означает, что group2 больше.
    """
    clean = [_clean_group(g) for g in groups]
    k = len(clean)
    if k < 2:
        result = StatResult(method="tukey_hsd", parameters={"alpha": alpha})
        result.add_warning("too_few_groups", "Нужно ≥ 2 групп", Severity.ERROR)
        return result

    if labels is None:
        labels = [f"g{i + 1}" for i in range(k)]
    elif len(labels) != k:
        raise ValueError("Число меток не совпадает с числом групп")

    values = np.concatenate(clean)
    group_col = np.concatenate([[lab] * g.size for lab, g in zip(labels, clean)])
    res = pairwise_tukeyhsd(values, group_col, alpha=alpha)

    result = StatResult(
        method="tukey_hsd",
        n=int(values.size),
        parameters={"alpha": alpha, "labels": labels},
    )
    comparisons = []
    for i, (g1, g2) in enumerate(_pairs(res.groupsunique.tolist())):
        comparisons.append(
            {
                "group1": g1,
                "group2": g2,
                "mean_diff": float(res.meandiffs[i]),
                "p_adj": float(res.pvalues[i]),
                "ci_lower": float(res.confint[i, 0]),
                "ci_upper": float(res.confint[i, 1]),
                "reject": bool(res.reject[i]),
            }
        )
    result.parameters["comparisons"] = comparisons
    result.statistics["n_comparisons"] = len(comparisons)
    result.statistics["n_significant"] = sum(c["reject"] for c in comparisons)
    return result


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    """Все неупорядоченные пары в том же порядке, что даёт statsmodels."""
    return [
        (items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


def kruskal_wallis(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
) -> StatResult:
    """Краскел-Уоллис — непараметрический аналог однофакторного ANOVA.

    Не требует нормальности и гомогенности дисперсий; сравнивает
    распределения по рангам.
    """
    clean = [_clean_group(g) for g in groups]
    k = len(clean)
    result = StatResult(method="kruskal_wallis")
    if k < 2:
        result.add_warning("too_few_groups", "Нужно ≥ 2 групп", Severity.ERROR)
        return result
    if any(g.size < 1 for g in clean):
        result.add_warning("empty_group", "Все группы должны быть непусты", Severity.ERROR)
        return result

    if labels is None:
        labels = [f"g{i + 1}" for i in range(k)]
    res = stats.kruskal(*clean)
    result.n = int(sum(g.size for g in clean))
    result.p_value = float(res.pvalue)
    result.parameters["labels"] = labels
    result.statistics = {"H": float(res.statistic), "df": k - 1}
    for lab, g in zip(labels, clean):
        result.estimates.append(
            Estimate(name=f"median_{lab}", value=float(np.median(g)))
        )
    return result


def treatment_means(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> StatResult:
    """Средние групп с SEM и ДИ по объединённой дисперсии, SED и LSD (гл. 4.4).

    Фон s² = ResMS однофакторного ANOVA на N − t df (стр. 84–87):
    SEM_j = √(s²/n_j), SED_jk = √(s²(1/n_j + 1/n_k)), LSD = t[α/2] × SED.
    Попарные p — без поправки на множественность (unprotected LSD).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должен быть в (0, 1)")
    clean = [_clean_group(g) for g in groups]
    labels = _resolve_labels(labels, len(clean))
    result = StatResult(
        method="treatment_means",
        n=int(sum(g.size for g in clean)),
        parameters={"alpha": alpha, "labels": labels},
    )
    if not _means_guard(clean, result):
        return result

    s2, df = _pooled_variance(clean)
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    result.statistics = {"residual_ms": s2, "df_resid": df, "t_crit": t_crit}
    _add_mean_estimates(clean, labels, s2, t_crit, 1 - alpha, result)
    result.parameters["comparisons"] = _lsd_comparisons(clean, labels, s2, df, t_crit)
    _add_common_lsd(clean, s2, t_crit, result)
    _check_anova_assumptions(clean, result)
    return result


def _add_common_lsd(
    groups: list[np.ndarray], s2: float, t_crit: float, result: StatResult
) -> None:
    """Общие SED и LSD при равной повторности; предупреждение о множественности."""
    sizes = {g.size for g in groups}
    if len(sizes) == 1:  # равная повторность → один SED на все пары (стр. 86)
        sed = float(np.sqrt(2 * s2 / sizes.pop()))
        result.statistics.update({"sed": sed, "lsd": t_crit * sed})
    if len(groups) > 2:
        result.add_warning(
            "unadjusted_comparisons",
            "Попарные p и ДИ разностей без поправки на множественность; "
            "для всех попарных сравнений используй Tukey HSD.",
            Severity.INFO,
        )


def _resolve_labels(labels: list[str] | None, k: int) -> list[str]:
    """Метки групп по умолчанию g1…gk; проверить длину заданных."""
    if labels is None:
        return [f"g{i + 1}" for i in range(k)]
    if len(labels) != k:
        raise ValueError("Число меток не совпадает с числом групп")
    return list(labels)


def _means_guard(groups: list[np.ndarray], result: StatResult) -> bool:
    """≥ 2 групп, n_j ≥ 2 и ненулевая остаточная дисперсия, иначе ошибка."""
    if len(groups) < 2:
        result.add_warning("too_few_groups", "Нужно ≥ 2 групп", Severity.ERROR)
        return False
    if any(g.size < 2 for g in groups):
        result.add_warning(
            "n_too_small", "Каждая группа должна иметь n ≥ 2", Severity.ERROR
        )
        return False
    if _pooled_variance(groups)[0] == 0:
        result.add_warning(
            "no_residual_variation",
            "Внутри групп нет вариации — SEM и SED равны нулю, ДИ не определены",
            Severity.ERROR,
        )
        return False
    return True


def _pooled_variance(groups: list[np.ndarray]) -> tuple[float, int]:
    """ResMS = Σ(y − ȳ_j)² / (N − t) и его df (стр. 80–81)."""
    df = int(sum(g.size for g in groups)) - len(groups)
    ss = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    return ss / df, df


def _add_mean_estimates(
    groups: list[np.ndarray],
    labels: list[str],
    s2: float,
    t_crit: float,
    level: float,
    result: StatResult,
) -> None:
    """Среднее каждой группы с SEM и ДИ на df остатка (стр. 85)."""
    for lab, g in zip(labels, groups):
        mean = float(g.mean())
        sem = float(np.sqrt(s2 / g.size))
        result.estimates.append(
            Estimate(
                name=f"mean_{lab}",
                value=mean,
                ci_lower=mean - t_crit * sem,
                ci_upper=mean + t_crit * sem,
                ci_level=level,
                std_error=sem,
            )
        )


def _lsd_comparisons(
    groups: list[np.ndarray],
    labels: list[str],
    s2: float,
    df: int,
    t_crit: float,
) -> list[dict[str, float | str]]:
    """Все пары: разность, SED, LSD, ДИ и p t-теста на df остатка (стр. 86–87).

    Знак как в tukey_hsd: mean_diff = mean(group2) − mean(group1).
    """
    rows = []
    for i, j in ((i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))):
        diff = float(groups[j].mean() - groups[i].mean())
        sed = float(np.sqrt(s2 * (1 / groups[i].size + 1 / groups[j].size)))
        t_stat = diff / sed
        rows.append(
            {
                "group1": labels[i],
                "group2": labels[j],
                "mean_diff": diff,
                "sed": sed,
                "lsd": t_crit * sed,
                "ci_lower": diff - t_crit * sed,
                "ci_upper": diff + t_crit * sed,
                "t": t_stat,
                "p_unadjusted": float(2 * stats.t.sf(abs(t_stat), df)),
            }
        )
    return rows
