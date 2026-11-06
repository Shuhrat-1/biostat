"""Однофакторный анализ: дополнения к anova.py по Witte (гл. 16).

- ``welch_anova`` — F Уэлча для неравных дисперсий [Witte, стр. 316;
  предл.: Welch 1951]. В Welham не рассматривается.

Все функции возвращают ``StatResult``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult


def _clean(data: np.ndarray | list[float]) -> np.ndarray:
    """1D float без NaN."""
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _labels(labels: list[str] | None, k: int) -> list[str]:
    """Метки групп по умолчанию g1…gk; проверка длины."""
    if labels is None:
        return [f"g{i + 1}" for i in range(k)]
    if len(labels) != k:
        raise ValueError("Число меток не совпадает с числом групп")
    return list(labels)


def welch_statistic(
    means: np.ndarray, variances: np.ndarray, sizes: np.ndarray
) -> tuple[float, float, float]:
    """F Уэлча и его df (df1 = k − 1, df2 по Уэлчу) [предл.: Welch 1951].

    w_j = n_j / s²_j; F = [Σ w_j (x̄_j − x̄_w)² / (k − 1)] / [1 + 2(k − 2)·Λ / (k² − 1)],
    Λ = Σ (1 − w_j/Σw)² / (n_j − 1); df2 = (k² − 1) / (3Λ).
    """
    k = means.size
    w = sizes / variances
    mean_w = float(np.sum(w * means) / np.sum(w))
    lam = float(np.sum((1 - w / w.sum()) ** 2 / (sizes - 1)))
    numerator = float(np.sum(w * (means - mean_w) ** 2)) / (k - 1)
    f = numerator / (1 + 2 * (k - 2) * lam / (k**2 - 1))
    return f, float(k - 1), float((k**2 - 1) / (3 * lam))


def _welch_guard(clean: list[np.ndarray], result: StatResult) -> bool:
    """Ошибки входа: < 2 групп, n < 2, нулевая дисперсия в группе."""
    if len(clean) < 2:
        result.add_warning("too_few_groups", "Нужно ≥ 2 групп", Severity.ERROR)
        return False
    if any(g.size < 2 for g in clean):
        result.add_warning("n_too_small", "Каждая группа должна иметь n ≥ 2", Severity.ERROR)
        return False
    if any(np.var(g, ddof=1) == 0 for g in clean):
        result.add_warning(
            "welch_zero_variance",
            "В одной из групп все значения одинаковы: веса Уэлча не определены.",
            Severity.ERROR,
        )
        return False
    return True


def welch_anova(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
) -> StatResult:
    """ANOVA Уэлча: сравнение средних нескольких групп без равенства дисперсий.

    Альтернатива обычному F при явном неравенстве дисперсий, особенно при
    неравных n (Witte, гл. 16.13, стр. 316: «более сложная версия F,
    рассчитанная на неравные дисперсии»).
    """
    clean = [_clean(g) for g in groups]
    names = _labels(labels, len(clean))
    sizes = [int(g.size) for g in clean]
    result = StatResult(
        method="welch_anova",
        n=int(sum(sizes)),
        parameters={"labels": names, "group_sizes": sizes},
    )
    if not _welch_guard(clean, result):
        return result

    means = np.array([g.mean() for g in clean])
    variances = np.array([g.var(ddof=1) for g in clean])
    f, df1, df2 = welch_statistic(means, variances, np.array(sizes, dtype=float))
    result.p_value = float(stats.f.sf(f, df1, df2))
    result.statistics = {"F": f, "df_between": df1, "df_within": df2}
    for lab, m, v in zip(names, means, variances):
        result.estimates.append(Estimate(name=f"mean_{lab}", value=float(m)))
        result.statistics[f"sd_{lab}"] = float(np.sqrt(v))
    return result
