"""Однофакторный анализ: дополнения к anova.py по Witte (гл. 16).

- ``welch_anova`` — F Уэлча для неравных дисперсий [Witte, стр. 316;
  предл.: Welch 1951]. В Welham не рассматривается.
- ``scheffe_test`` — метод Шеффе: все пары и любые контрасты с контролем
  ошибки I рода для всего семейства контрастов [Witte, стр. 314;
  предл.: Scheffé 1953]. Синтаксис контрастов — как у treatment_contrasts.

Все функции возвращают ``StatResult``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.contrasts import parse_contrasts
from core.stats.guards import note_single_groups


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


def _scheffe_row(coefs: np.ndarray, fit: dict, alpha: float) -> dict:
    """Оценка контраста, SE, F_S = (ψ̂/SE)²/(k − 1), P по F(k − 1, N − k), ДИ ± S·SE."""
    psi = float(coefs @ fit["means"])
    se = float(np.sqrt(fit["ms"] * np.sum(coefs**2 / fit["n"])))
    f_s = (psi / se) ** 2 / fit["df1"]
    p = float(stats.f.sf(f_s, fit["df1"], fit["df2"]))
    half = fit["S"] * se
    return {"estimate": psi, "se": se, "F_scheffe": f_s, "p_adj": p,
            "ci_lower": psi - half, "ci_upper": psi + half, "reject": bool(p < alpha)}


def _scheffe_fit(clean: list[np.ndarray], alpha: float) -> dict:
    """Средние, n, MS_within, df и критическое S = √((k − 1)·F_crit)."""
    n = np.array([g.size for g in clean], dtype=float)
    df1, df2 = len(clean) - 1, int(n.sum()) - len(clean)
    ms = float(sum(((g - g.mean()) ** 2).sum() for g in clean)) / df2
    s_crit = float(np.sqrt(df1 * stats.f.ppf(1 - alpha, df1, df2)))
    return {"means": np.array([g.mean() for g in clean]), "n": n, "ms": ms,
            "df1": df1, "df2": df2, "S": s_crit}


def _scheffe_pairs(names: list[str], fit: dict, alpha: float) -> list[dict]:
    """Все пары; знак как у tukey_hsd: mean(group2) − mean(group1)."""
    sd = np.sqrt(fit["ms"])
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            coefs = np.zeros(len(names))
            coefs[i], coefs[j] = -1.0, 1.0
            row = _scheffe_row(coefs, fit, alpha)
            rows.append({"group1": names[i], "group2": names[j], "mean_diff": row.pop("estimate"),
                         **row, "cohens_d": float((fit["means"][j] - fit["means"][i]) / sd)})
    return rows


def _scheffe_guard(clean: list[np.ndarray], result: StatResult) -> bool:
    """Ошибки входа: < 2 групп, пустая группа, нет df остатка, MS_within = 0."""
    if len(clean) < 2:
        result.add_warning("too_few_groups", "Нужно ≥ 2 групп", Severity.ERROR)
        return False
    if any(g.size < 1 for g in clean):
        result.add_warning("empty_group", "Все группы должны быть непусты", Severity.ERROR)
        return False
    if sum(g.size for g in clean) - len(clean) < 1:
        result.add_warning("n_too_small", "Нет степеней свободы остатка: нужны повторности",
                           Severity.ERROR)
        return False
    if all(np.ptp(g) == 0 for g in clean):
        result.add_warning("scheffe_zero_variance",
                           "Внутри групп нет разброса: MS_within = 0, сравнения не определены.",
                           Severity.ERROR)
        return False
    return True


def scheffe_test(
    *groups: np.ndarray | list[float],
    labels: list[str] | None = None,
    alpha: float = 0.05,
    contrasts: str = "",
) -> StatResult:
    """Метод Шеффе: попарные сравнения и сложные контрасты (Witte, гл. 16.10).

    Консервативнее Тьюки для пар, но допускает любые контрасты, например
    «0 vs 24 + 48», не теряя контроля ошибки I рода на всём семействе.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")
    clean = [_clean(g) for g in groups]
    names = _labels(labels, len(clean))
    result = StatResult(method="scheffe_test", n=int(sum(g.size for g in clean)),
                        parameters={"alpha": alpha, "labels": names})
    if not _scheffe_guard(clean, result):
        return result
    note_single_groups(result, [g.size for g in clean], names)
    fit = _scheffe_fit(clean, alpha)
    comparisons = _scheffe_pairs(names, fit, alpha)
    result.parameters["comparisons"] = comparisons
    if contrasts.strip():
        try:
            parsed = parse_contrasts(contrasts, names)
        except ValueError as exc:
            result.add_warning("contrast_invalid", str(exc), Severity.ERROR)
            return result
        result.parameters["contrasts"] = _scheffe_contrasts(parsed, names, fit, alpha, result)
    result.statistics.update({
        "critical_S": fit["S"], "ms_within": fit["ms"], "df_between": fit["df1"],
        "df_within": fit["df2"], "n_comparisons": len(comparisons),
        "n_significant": sum(c["reject"] for c in comparisons),
    })
    return result


def _scheffe_contrasts(parsed: list[tuple[str, np.ndarray]], names: list[str], fit: dict,
                       alpha: float, result: StatResult) -> list[dict]:
    """Строки контрастов; оценки с ДИ Шеффе — и в estimates."""
    rows = []
    for name, coefs in parsed:
        est = _scheffe_row(coefs, fit, alpha)
        # Имена полей — как у treatment_contrasts, чтобы таблица контрастов читалась так же.
        row = {"term": name, "coefficients": dict(zip(names, coefs.tolist())),
               "estimate": est["estimate"], "se": est["se"], "F": est["F_scheffe"],
               "p_adjusted": est["p_adj"], "ci_lower": est["ci_lower"],
               "ci_upper": est["ci_upper"], "significant": est["reject"]}
        rows.append(row)
        result.estimates.append(Estimate(
            name=name, value=row["estimate"], ci_lower=row["ci_lower"],
            ci_upper=row["ci_upper"], ci_level=1 - alpha, std_error=row["se"]))
    return rows
