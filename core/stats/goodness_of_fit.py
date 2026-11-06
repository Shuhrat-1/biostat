"""Одномерный χ²-тест согласия (Witte, гл. 19.1–19.5; пакет W3).

f_e = p₀·n (ф. 19.1), χ² = Σ(f_o − f_e)²/f_e (ф. 19.2), df = c − 1 (ф. 19.3).
Гипотетические доли задаются текстом «уровень=доля; …» (можно отношениями:
«жёлтые=9; зелёные=3; …» — нормируются); без них — все доли равны.

Дополнительно [предл.]: G² (отношение правдоподобия), w Коэна = √(χ²/n),
ДИ Уилсона наблюдаемых долей, стандартизованные и скорректированные остатки,
при двух категориях — точный биномиальный P. φ²c = χ²/(n(c − 1)) — как Witte
в ответе к Review 19.10c (стр. 454).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from core.report.schema import Severity, StatResult

_MIN_EXPECTED = 5.0  # «консервативное правило» Witte, стр. 380


def _to_float(values: list) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            out[i] = float(str(v).replace(",", "."))
        except (TypeError, ValueError):
            pass
    return out


def _counts(response: list, count: list | None) -> dict[str, float]:
    """Частоты категорий в порядке первого появления; пропуски — вон."""
    w = np.ones(len(response)) if count is None else _to_float(count)
    out: dict[str, float] = {}
    for v, wt in zip(response, w):
        label = "" if v is None else str(v).strip()
        if label in ("", "nan", "None") or math.isnan(wt):
            continue
        out[label] = out.get(label, 0.0) + float(wt)
    return out


def parse_proportions(text: str) -> dict[str, float]:
    """«A=0.4; B=0.6» → {A: 0.4, B: 0.6}; ValueError при ошибке. Нормировка — снаружи."""
    out: dict[str, float] = {}
    for item in (p.strip() for p in text.split(";") if p.strip()):
        if "=" not in item:
            raise ValueError(f"Ожидалось «уровень=доля»: «{item}»")
        level, value = item.rsplit("=", 1)
        share = float(value.strip().replace(",", "."))
        if not share > 0:
            raise ValueError(f"Доля уровня «{level.strip()}» должна быть > 0")
        out[level.strip()] = share
    if len(out) < 2:
        raise ValueError("Нужно ≥ 2 уровней с долями")
    return out


def _hypothesis(observed: dict[str, float], proportions: str,
                result: StatResult) -> dict[str, float] | None:
    """Доли H0 по уровням (нормированные); None при ошибке (предупреждение добавлено)."""
    if not proportions.strip():
        return {k: 1 / len(observed) for k in observed}
    try:
        shares = parse_proportions(proportions)
    except ValueError as exc:
        result.add_warning("gof_bad_proportions", str(exc), Severity.ERROR)
        return None
    unknown = [k for k in observed if k not in shares]
    if unknown:
        result.add_warning("gof_unknown_levels", "Нет доли H0 для уровней: " + ", ".join(unknown[:10]),
                           Severity.ERROR, params={"levels": ", ".join(unknown[:10])})
        return None
    total = sum(shares.values())
    if abs(total - 1) > 1e-6:
        result.add_warning("gof_normalized", f"Доли H0 в сумме дают {total:g}: нормированы к 1.",
                           Severity.INFO, params={"total": round(total, 6)})
    return {k: v / total for k, v in shares.items()}


def _cells(obs: np.ndarray, p0: np.ndarray, labels: list[str], alpha: float) -> list[dict]:
    """f_o, f_e, доли, ДИ Уилсона, стандартизованный и скорректированный остатки."""
    n = obs.sum()
    exp = p0 * n
    lo, hi = proportion_confint(obs, n, alpha=alpha, method="wilson")
    return [{"category": lab, "observed": float(o), "expected": float(e), "p_h0": float(p),
             "p_observed": float(o / n), "ci_lower": float(a), "ci_upper": float(b),
             "std_residual": float((o - e) / np.sqrt(e)),
             "adjusted_residual": float((o - e) / np.sqrt(e * (1 - p)))}
            for lab, o, e, p, a, b in zip(labels, obs, exp, p0, np.atleast_1d(lo), np.atleast_1d(hi))]


def _statistics(obs: np.ndarray, p0: np.ndarray) -> dict[str, float]:
    """χ², G², df, P, φ²c, w; при c = 2 — точный биномиальный P."""
    n, c = obs.sum(), obs.size
    exp = p0 * n
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    nz = obs > 0
    g2 = float(2 * np.sum(obs[nz] * np.log(obs[nz] / exp[nz])))
    out = {"chi2": chi2, "df": c - 1, "p": float(stats.chi2.sf(chi2, c - 1)),
           "g2": g2, "p_g2": float(stats.chi2.sf(g2, c - 1)), "n_total": float(n),
           "phi_c_squared": float(chi2 / (n * (c - 1))), "cohens_w": math.sqrt(chi2 / n)}
    if c == 2 and np.allclose(obs, np.round(obs)):
        out["binomial_p"] = float(stats.binomtest(int(np.rint(obs[0])), int(np.rint(n)), p0[0]).pvalue)
    return out


def chi2_goodness_of_fit(
    response: list,
    count: list | None = None,
    proportions: str = "",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """χ² согласия одной номинальной переменной с долями H0 (Witte, гл. 19.1–19.5)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")
    observed = _counts(response, count)
    result = StatResult(method="chi2_goodness_of_fit",
                        parameters={"alpha": alpha,
                                    "column": (column_names or {}).get("response", "")})
    if len(observed) < 1 or sum(observed.values()) <= 0 or min(observed.values()) < 0:
        result.add_warning("gof_empty", "Нет наблюдений или частоты отрицательны.", Severity.ERROR)
        return result
    p_h0 = _hypothesis(observed, proportions, result)
    if p_h0 is None:
        return result
    if len(p_h0) < 2:
        result.add_warning("gof_one_category", "Нужно ≥ 2 категорий.", Severity.ERROR)
        return result
    labels = list(p_h0)
    obs = np.array([observed.get(k, 0.0) for k in labels])
    p0 = np.array([p_h0[k] for k in labels])
    result.n = int(np.rint(obs.sum()))
    result.statistics = _statistics(obs, p0)
    result.p_value = result.statistics["p"]
    result.parameters["cells"] = _cells(obs, p0, labels, alpha)
    _small_expected(p0 * obs.sum(), labels, result)
    return result


def _small_expected(exp: np.ndarray, labels: list[str], result: StatResult) -> None:
    """Witte (стр. 380): все f_e ≥ 5; иначе объединить категории или увеличить n."""
    small = [lab for lab, e in zip(labels, exp) if e < _MIN_EXPECTED]
    if small:
        result.add_warning(
            "gof_small_expected",
            f"Ожидаемые частоты < 5: {', '.join(small[:10])}. χ² ненадёжен — объедините "
            f"категории по смыслу или увеличьте выборку; при двух категориях — точный "
            f"биномиальный P.",
            params={"levels": ", ".join(small[:10])})
