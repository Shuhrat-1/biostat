"""Доли: одна доля, разность двух долей, Мак-Немар (пакет W3).

Witte даёт ДИ процента по Вальду (гл. 12.7, стр. 228) и χ² для частот (гл. 19);
отдельных методов долей в книге нет — всё ниже [предл.]:
- ``one_proportion``: оценка, ДИ Уилсона и Клоппера–Пирсона, точный
  биномиальный P и score-тест [предл.: Wilson 1927; Clopper & Pearson 1934;
  statsmodels proportion_confint, proportions_ztest].
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

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
