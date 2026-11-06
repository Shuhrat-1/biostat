"""Доли: одна доля, разность двух долей, Мак-Немар (пакет W3).

Witte даёт ДИ процента по Вальду (гл. 12.7, стр. 228) и χ² для частот (гл. 19);
отдельных методов долей в книге нет — всё ниже [предл.]:
- ``one_proportion``: оценка, ДИ Уилсона и Клоппера–Пирсона, точный
  биномиальный P и score-тест [предл.: Wilson 1927; Clopper & Pearson 1934;
  statsmodels proportion_confint, proportions_ztest];
- ``two_proportions``: разность долей независимых групп, ДИ Ньюкомба
  [предл.: Newcombe 1998; statsmodels confint_proportions_2indep], z-тест с
  объединённой долей (z² = χ² 2 × 2 без Йейтса).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from core.report.schema import Severity, StatResult

_ALTERNATIVES = ("two-sided", "less", "greater")
_SUCCESS_HINTS = ("1", "yes", "y", "true", "да", "sim", "s", "success", "+")


def _to_float(values: list) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            out[i] = float(str(v).replace(",", "."))
        except (TypeError, ValueError):
            pass
    return out


def _label(v: object) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s in ("nan", "None") else s


def _natural(levels: list[str]) -> list[str]:
    """Числовые коды по значению, остальные по алфавиту."""
    def key(level: str) -> tuple[int, float, str]:
        try:
            return (0, float(level.replace(",", ".")), level)
        except ValueError:
            return (1, 0.0, level)
    return sorted(set(levels), key=key)


def success_level(levels: list[str], success: str, result: StatResult) -> str | None:
    """Уровень «успех»: заданный; иначе 1/yes/да/…; иначе первый по порядку (info)."""
    if success.strip():
        if success.strip() not in levels:
            result.add_warning("prop_unknown_success",
                               f"Уровня «{success.strip()}» нет в данных; есть: {', '.join(levels[:10])}",
                               Severity.ERROR, params={"level": success.strip()})
            return None
        return success.strip()
    hinted = [lv for lv in levels if lv.lower() in _SUCCESS_HINTS]
    chosen = hinted[0] if hinted else levels[0]
    result.add_warning("prop_success_chosen", f"«Успех» — уровень «{chosen}» (задайте явно в "
                       "параметре, если нужен другой).", Severity.INFO, params={"level": chosen})
    return chosen


def _check(alpha: float, alternative: str) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative ∈ {_ALTERNATIVES}")


# --- Одна доля ------------------------------------------------------------------------


def _one_counts(response: list, count: list | None) -> dict[str, float]:
    w = np.ones(len(response)) if count is None else _to_float(count)
    out: dict[str, float] = {}
    for v, wt in zip(response, w):
        if (lab := _label(v)) and not math.isnan(wt):
            out[lab] = out.get(lab, 0.0) + float(wt)
    return out


def proportion_summary(x: int, n: int, alpha: float) -> dict[str, float]:
    """p̂, SE, ДИ Уилсона и Клоппера–Пирсона, полуширина Вальда (Witte, стр. 228)."""
    p = x / n
    wl, wh = proportion_confint(x, n, alpha=alpha, method="wilson")
    el, eh = proportion_confint(x, n, alpha=alpha, method="beta")
    se = math.sqrt(p * (1 - p) / n)
    return {"successes": float(x), "n_total": float(n), "proportion": p, "se": se,
            "ci_lower": float(wl), "ci_upper": float(wh),
            "exact_ci_lower": float(el), "exact_ci_upper": float(eh),
            "margin_of_error_wald": float(stats.norm.ppf(1 - alpha / 2) * se)}


def _one_tests(x: int, n: int, p0: float, alternative: str) -> dict[str, float]:
    """Точный биномиальный P и score-тест (SE по p₀; z² = χ² согласия с 2 категориями)."""
    sm_alt = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}[alternative]
    z, p_z = proportions_ztest(x, n, value=p0, alternative=sm_alt, prop_var=p0)
    return {"p0": p0, "z": float(z), "p_score": float(p_z),
            "binomial_p": float(stats.binomtest(x, n, p0, alternative=alternative).pvalue)}


def one_proportion(
    response: list,
    count: list | None = None,
    success: str = "",
    p0: float = 0.5,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Доля «успехов» в одной выборке: оценка, ДИ, тест H0: p = p₀ [предл.]."""
    _check(alpha, alternative)
    if not 0 < p0 < 1:
        raise ValueError("p0 ∈ (0, 1)")
    counts = _one_counts(response, count)
    result = StatResult(method="one_proportion",
                        parameters={"alpha": alpha, "alternative": alternative,
                                    "column": (column_names or {}).get("response", "")})
    if not counts or any(v < 0 for v in counts.values()) or not np.allclose(
            list(counts.values()), np.rint(list(counts.values()))):
        result.add_warning("prop_bad_counts", "Нужны целые неотрицательные частоты.", Severity.ERROR)
        return result
    level = success_level(_natural(list(counts)), success, result)
    if level is None:
        return result
    x, n = int(np.rint(counts[level])), int(np.rint(sum(counts.values())))
    result.n = n
    result.parameters.update({"success": level, "levels": _natural(list(counts))})
    result.statistics = {**proportion_summary(x, n, alpha), **_one_tests(x, n, p0, alternative)}
    result.p_value = result.statistics["binomial_p"]
    return result


# --- Две независимые доли -----------------------------------------------------------


def _two_counts(response: list, group: list, count: list | None
                ) -> tuple[dict[str, dict[str, float]], list[str]]:
    """{группа: {уровень: частота}} и все уровни отклика."""
    w = np.ones(len(response)) if count is None else _to_float(count)
    table: dict[str, dict[str, float]] = {}
    levels: list[str] = []
    for r, g, wt in zip(response, group, w):
        r, g = _label(r), _label(g)
        if not r or not g or math.isnan(wt):
            continue
        table.setdefault(g, {})
        table[g][r] = table[g].get(r, 0.0) + float(wt)
        levels.append(r)
    return table, _natural(levels)


def _two_fit(x: np.ndarray, n: np.ndarray, alpha: float, alternative: str) -> dict:
    """p₁, p₂, разность p₂ − p₁ с ДИ Ньюкомба, z (объединённая доля), P."""
    p = x / n
    lo, hi = confint_proportions_2indep(x[1], n[1], x[0], n[0], method="newcomb",
                                        compare="diff", alpha=alpha)
    # statsmodels проверяет p₁ − p₂; наша разность — p₂ − p₁, поэтому направление обратное.
    alt = {"two-sided": "two-sided", "greater": "smaller", "less": "larger"}[alternative]
    z, p_z = proportions_ztest(x, n, alternative=alt)
    return {"p1": float(p[0]), "p2": float(p[1]), "diff": float(p[1] - p[0]),
            "ci_lower": float(lo), "ci_upper": float(hi), "z": float(-z), "p": float(p_z),
            "chi2_equivalent": float(z**2)}


def two_proportions(
    response: list,
    group: list,
    count: list | None = None,
    success: str = "",
    alternative: str = "two-sided",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Разность долей «успехов» двух независимых групп (group2 − group1) [предл.].

    ``alternative``: greater — доля во второй группе больше, less — меньше.
    """
    _check(alpha, alternative)
    table, levels = _two_counts(response, group, count)
    result = StatResult(method="two_proportions",
                        parameters={"alpha": alpha, "alternative": alternative})
    if len(table) != 2:
        result.add_warning("prop_two_groups", f"Нужно ровно 2 группы, найдено {len(table)}.",
                           Severity.ERROR, params={"k": len(table)})
        return result
    level = success_level(levels, success, result) if levels else None
    if level is None:
        return result
    groups = _natural(list(table))
    x = np.array([table[g].get(level, 0.0) for g in groups])
    n = np.array([sum(table[g].values()) for g in groups])
    if np.any(n <= 0):
        result.add_warning("prop_bad_counts", "Нужны целые неотрицательные частоты.", Severity.ERROR)
        return result
    fit = _two_fit(x, n, alpha, alternative)
    result.n = int(n.sum())
    result.p_value = fit["p"]
    result.statistics = {**fit, "x1": float(x[0]), "n1": float(n[0]), "x2": float(x[1]),
                         "n2": float(n[1])}
    result.parameters.update(_two_parameters(groups, level, x, n, fit, alpha))
    return result


def _two_parameters(groups: list[str], level: str, x: np.ndarray, n: np.ndarray, fit: dict,
                    alpha: float) -> dict:
    """Доли групп с ДИ Уилсона и строка сравнения для графика разностей."""
    rows = []
    for g, xi, ni in zip(groups, x, n):
        lo, hi = proportion_confint(xi, ni, alpha=alpha, method="wilson")
        rows.append({"label": g, "x": float(xi), "n": float(ni), "mean": float(xi / ni),
                     "ci_lower": float(lo), "ci_upper": float(hi)})
    return {"success": level, "groups": groups, "proportions": rows,
            "comparisons": [{"group1": groups[0], "group2": groups[1], "mean_diff": fit["diff"],
                             "ci_lower": fit["ci_lower"], "ci_upper": fit["ci_upper"],
                             "p_adj": fit["p"]}]}
