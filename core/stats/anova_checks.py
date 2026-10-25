"""Проверка допущений однофакторного ANOVA (гл. 5 книги).

Остатки CRD в определении книги (стр. 94–96), тест Бартлетта на
гомогенность дисперсий (стр. 105–107) и перестановочный F-тест для
случая ненормальных отклонений (стр. 102–103). Все функции принимают
группы так же, как one_way_anova, и возвращают ``StatResult``.
"""

from __future__ import annotations

from math import factorial, prod

import numpy as np
from scipy import stats

from core.report.schema import Severity, StatResult
from core.stats.anova import _clean_group, _pooled_variance, _resolve_labels
from core.stats.hypothesis import levene_p_value

_ALPHA = 0.05
_OUTLIER_CUTOFF = 2.0  # |r| > 2 — кандидаты в выбросы (стр. 95)
_MAX_PERMUTATIONS = 99_999


def _guard_groups(groups: list[np.ndarray], result: StatResult) -> bool:
    """≥ 2 групп, n_j ≥ 2 и ненулевая дисперсия внутри групп, иначе ошибка."""
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
            "no_residual_variation", "Внутри групп нет вариации", Severity.ERROR
        )
        return False
    return True


def bartlett_test(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
) -> StatResult:
    """Тест Бартлетта: H0 — дисперсии всех групп равны (гл. 5.3).

    X² ~ χ²_{t−1}, тест односторонний. Чувствителен к ненормальности:
    значимый результат может отражать её, а не неравные дисперсии.
    """
    clean = [_clean_group(g) for g in groups]
    labels = _resolve_labels(labels, len(clean))
    result = StatResult(
        method="bartlett_test",
        n=int(sum(g.size for g in clean)),
        parameters={"labels": labels},
    )
    if not _guard_groups(clean, result):
        return result
    if any(np.var(g, ddof=1) == 0 for g in clean):
        result.add_warning(
            "zero_group_variance",
            "В одной из групп все значения одинаковы — log(0), тест неприменим",
            Severity.ERROR,
        )
        return result

    res = stats.bartlett(*clean)
    result.p_value = float(res.pvalue)
    result.statistics = {
        "chi2": float(res.statistic),
        "df": len(clean) - 1,
        "pooled_variance": _pooled_variance(clean)[0],
    }
    result.parameters["group_variances"] = {
        lab: float(np.var(g, ddof=1)) for lab, g in zip(labels, clean)
    }
    if res.pvalue < _ALPHA:
        result.add_warning(
            "bartlett_unequal_variances",
            "Дисперсии групп различаются (Бартлетт). Если дисперсия растёт со "
            "средним — трансформация отклика (гл. 6); тест чувствителен и к "
            "ненормальности (гл. 18).",
            Severity.WARNING,
        )
    return result


def anova_residuals(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
) -> StatResult:
    """Простые и стандартизованные остатки однофакторной модели (гл. 5.1).

    r_jk = e_jk / √(s²(n_j − 1)/n_j), s² = ResMS (стр. 96). Отдаёт ряды
    для графиков: остатки против подогнанных, |r| против подогнанных,
    нормальный график с позициями (i − 0.375)/(n + 0.25) (стр. 101).
    """
    clean = [_clean_group(g) for g in groups]
    labels = _resolve_labels(labels, len(clean))
    result = StatResult(
        method="anova_residuals",
        n=int(sum(g.size for g in clean)),
        parameters={"labels": labels},
    )
    if not _guard_groups(clean, result):
        return result

    s2, df = _pooled_variance(clean)
    rows = _residual_rows(clean, labels, s2)
    std = np.array([row["std_residual"] for row in rows])
    result.statistics = {
        "residual_ms": s2,
        "df_resid": df,
        "max_abs_std_residual": float(np.abs(std).max()),
        "n_abs_std_residual_gt_2": int((np.abs(std) > _OUTLIER_CUTOFF).sum()),
    }
    result.parameters["residuals"] = rows
    result.parameters["plots"] = _residual_plots(rows)
    _flag_outliers(rows, result)
    return result


def _residual_rows(
    groups: list[np.ndarray], labels: list[str], s2: float
) -> list[dict[str, float | str | int]]:
    """Строка на наблюдение: группа, номер в группе, y, ŷ, e, r."""
    rows = []
    for lab, g in zip(labels, groups):
        se = float(np.sqrt(s2 * (g.size - 1) / g.size))
        for k, y in enumerate(g):
            resid = float(y - g.mean())
            rows.append(
                {
                    "group": lab,
                    "index_in_group": k + 1,
                    "value": float(y),
                    "fitted": float(g.mean()),
                    "residual": resid,
                    "std_residual": resid / se,
                }
            )
    return rows


def _residual_plots(rows: list[dict[str, float | str | int]]) -> dict:
    """Ряды для фитированного, абсолютного и нормального графиков (стр. 96–101)."""
    fitted = [float(r["fitted"]) for r in rows]
    std = np.array([float(r["std_residual"]) for r in rows])
    n = std.size
    blom = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    return {
        "residual_vs_fitted": {"fitted": fitted, "residuals": std.tolist()},
        "abs_residual_vs_fitted": {
            "fitted": fitted,
            "abs_residuals": np.abs(std).tolist(),
        },
        "qq": {
            "theoretical_quantiles": stats.norm.ppf(blom).tolist(),
            "sample_quantiles": np.sort(std).tolist(),
        },
        "std_residuals": std.tolist(),
    }


def _flag_outliers(rows: list[dict[str, float | str | int]], result: StatResult) -> None:
    """Пометить |r| > 2 как кандидатов в выбросы, без удаления (стр. 108–110)."""
    flagged = [r for r in rows if abs(float(r["std_residual"])) > _OUTLIER_CUTOFF]
    result.parameters["potential_outliers"] = flagged
    if flagged:
        result.add_warning(
            "potential_outliers",
            f"Наблюдений с |стандартизованным остатком| > 2: {len(flagged)} "
            f"(при нормальности ожидается около 5 %). Проверь их по первичным "
            f"записям; удалять без найденной причины нельзя.",
            Severity.INFO,
            params={"count": len(flagged), "expected": round(0.05 * len(rows), 1)},
        )


def anova_permutation_test(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
    n_permutations: int = 999,
    seed: int | None = None,
) -> StatResult:
    """Перестановочный F-тест однофакторного ANOVA (гл. 5.2.4).

    Альтернатива F-распределению при ненормальных отклонениях; требует
    примерно однородных дисперсий. p = доля перестановок с F не меньше
    наблюдаемого; при малом числе перестановок перебор полный.
    """
    n_permutations = int(n_permutations)  # из формы может прийти float
    if not 1 <= n_permutations <= _MAX_PERMUTATIONS:
        raise ValueError(f"n_permutations ∈ [1, {_MAX_PERMUTATIONS}]")
    clean = [_clean_group(g) for g in groups]
    labels = _resolve_labels(labels, len(clean))
    seed = int(np.random.SeedSequence().entropy % 2**32) if seed is None else seed
    result = StatResult(
        method="anova_permutation_test",
        n=int(sum(g.size for g in clean)),
        seed=seed,
        parameters={"labels": labels, "n_permutations": n_permutations},
    )
    if not _guard_groups(clean, result):
        return result

    f_res = stats.f_oneway(*clean)
    result.p_value = _permutation_p(clean, n_permutations, seed)
    result.statistics = {
        "F": float(f_res.statistic),
        "f_test_p": float(f_res.pvalue),
        "df_between": len(clean) - 1,
        "df_within": _pooled_variance(clean)[1],
    }
    result.parameters["exact"] = _n_arrangements(clean) <= n_permutations
    _check_permutation_variances(clean, result)
    return result


def _permutation_p(groups: list[np.ndarray], n_permutations: int, seed: int) -> float:
    """p перестановочного теста по F (scipy; random_state есть и в scipy 1.12)."""
    res = stats.permutation_test(
        groups,
        _f_statistic,
        permutation_type="independent",
        n_resamples=n_permutations,
        alternative="greater",
        random_state=np.random.default_rng(seed),
    )
    return float(res.pvalue)


def _f_statistic(*samples: np.ndarray) -> float:
    """F однофакторного ANOVA для перестановочного распределения."""
    return float(stats.f_oneway(*samples).statistic)


def _n_arrangements(groups: list[np.ndarray]) -> int:
    """Число различимых раскладок N!/(Π n_j!) — полный перебор, если не больше B."""
    return factorial(sum(g.size for g in groups)) // prod(factorial(g.size) for g in groups)


def _check_permutation_variances(groups: list[np.ndarray], result: StatResult) -> None:
    """Перестановочный тест требует сопоставимых дисперсий (стр. 102).

    Вырожденный Левене (группы по 2 наблюдения) пропускается.
    """
    levene_p = levene_p_value(*groups)
    if levene_p is None:
        return
    result.statistics["levene_p"] = levene_p
    if levene_p < _ALPHA:
        result.add_warning(
            "permutation_unequal_variances",
            f"Дисперсии групп неоднородны (Левене p={levene_p:.4f}); "
            f"перестановочный тест в книге предполагается при однородных "
            f"дисперсиях.",
            Severity.WARNING,
        )
