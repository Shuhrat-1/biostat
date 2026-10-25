"""Сравнения по таблице предсказанных средних (гл. 8.8 книги).

``mean_comparisons`` — один фактор вариантов (CRD или RCBD), набор
сравнений: все пары или каждый вариант против контроля. Процедуры:

- ``lsd`` — t-тесты без поправки (8.8.3.1); вместе с ними — наблюдаемый FDR;
- ``bonferroni`` — α/m на сравнение (8.8.1);
- ``fdr`` — Бенджамини–Хохберг (8.8.2);
- ``tukey`` — одновременные ДИ Тьюки, q[t]/√2 × SED (8.8.3.3);
- ``alsd`` — скорректированный LSD Хсу по книге: q[t−1]/√2 × SED (8.8.3.1);
- ``dunnett`` — против контроля, критическое значение распределения Даннета (8.8.4).

Квантиль Даннета считается детерминированно: двойной интеграл по общей
нормальной компоненте и по s (Dunnett, 1955), неравная повторность допустима.
Тесты множественного размаха (Ньюман–Кейлс, Дункан) не реализованы: книга
не рекомендует их из-за неопределённой экспериментной ошибки (стр. 199).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

from core.report.schema import Estimate, Severity, StatResult
from core.stats.contrasts import means_table, one_factor_fit
from core.stats.factorial import _design_error, _settings

PROCEDURES = ("lsd", "bonferroni", "fdr", "tukey", "alsd", "dunnett")
ALTERNATIVES = ("two-sided", "greater", "less")

# Узлы квадратур для распределения Даннета: x ~ N(0, 1), s ~ χ_df/√df.
_X_NODES, _X_WEIGHTS = np.polynomial.hermite_e.hermegauss(96)
_X_WEIGHTS = _X_WEIGHTS / np.sqrt(2 * np.pi)
_U_NODES, _U_WEIGHTS = np.polynomial.legendre.leggauss(96)
_U_NODES, _U_WEIGHTS = (_U_NODES + 1) / 2, _U_WEIGHTS / 2


# --- Распределение Даннета ----------------------------------------------------


def dunnett_cdf(c: float, ratios: np.ndarray, df: float, two_sided: bool = True) -> float:
    """P(все |T_i| ≤ c) (или T_i ≤ c) для сравнений с контролем.

    ``ratios`` — n_i / n_0 для каждого варианта; при равной повторности — единицы.
    """
    s = stats.chi.ppf(_U_NODES, df) / np.sqrt(df)
    r = np.asarray(ratios, dtype=float)[None, None, :]
    centre = _X_NODES[None, :, None] * np.sqrt(r)
    half = c * s[:, None, None] * np.sqrt(1 + r)
    prob = stats.norm.cdf(centre + half)
    if two_sided:
        prob = prob - stats.norm.cdf(centre - half)
    return float(_U_WEIGHTS @ (np.prod(prob, axis=2) @ _X_WEIGHTS))


def dunnett_quantile(alpha: float, ratios: np.ndarray, df: float,
                     two_sided: bool = True) -> float:
    """Критическое значение d: P(max статистики ≤ d) = 1 − alpha (стр. 201)."""
    return float(optimize.brentq(
        lambda c: dunnett_cdf(c, ratios, df, two_sided) - (1 - alpha), 0.0, 50.0, xtol=1e-8))


# --- Набор сравнений ----------------------------------------------------------


def _pairs(levels: list[str], control: str) -> list[tuple[int, int]]:
    """Пары (i, j) для разности μ_j − μ_i: все пары или каждый против контроля."""
    if not control:
        return list(combinations(range(len(levels)), 2))
    lookup = {lv.lower(): i for i, lv in enumerate(levels)}
    if control.strip().lower() not in lookup:
        raise ValueError(f"Контроль «{control}» не найден; уровни: {', '.join(levels)}")
    c = lookup[control.strip().lower()]
    return [(c, j) for j in range(len(levels)) if j != c]


def _check_procedure(procedure: str, control: str, alternative: str, n_levels: int) -> None:
    """Совместимость процедуры, набора сравнений и альтернативы."""
    if procedure not in PROCEDURES:
        raise ValueError(f"procedure ∈ {PROCEDURES}")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative ∈ {ALTERNATIVES}")
    if procedure == "dunnett" and not control:
        raise ValueError("Для метода Даннета укажи контрольный уровень")
    if procedure in ("tukey", "alsd") and alternative != "two-sided":
        raise ValueError("Тьюки и aLSD — только двусторонние")
    if alternative != "two-sided" and not control:
        raise ValueError("Односторонняя альтернатива — только для сравнений с контролем")
    if procedure == "alsd" and n_levels < 3:
        raise ValueError("aLSD требует ≥ 3 уровней (q для t − 1 групп)")


# --- Критические значения и поправленные P ------------------------------------


def _critical(procedure: str, alpha: float, m: int, fit: dict, alternative: str,
              ratios: np.ndarray) -> float:
    """Множитель SED для полуширины ДИ (на шкале t)."""
    df, t_levels = fit["df"], len(fit["levels"])
    tails = 2 if alternative == "two-sided" else 1
    if procedure == "bonferroni":
        return float(stats.t.ppf(1 - alpha / (tails * m), df))
    if procedure == "tukey":
        return float(stats.studentized_range.ppf(1 - alpha, t_levels, df) / np.sqrt(2))
    if procedure == "alsd":
        return float(stats.studentized_range.ppf(1 - alpha, t_levels - 1, df) / np.sqrt(2))
    if procedure == "dunnett":
        return dunnett_quantile(alpha, ratios, df, alternative == "two-sided")
    return float(stats.t.ppf(1 - alpha / tails, df))  # lsd, fdr


def _raw_p(t_stat: np.ndarray, df: float, alternative: str) -> np.ndarray:
    """P t-теста на df остатка с учётом направления альтернативы."""
    if alternative == "greater":
        return stats.t.sf(t_stat, df)
    if alternative == "less":
        return stats.t.cdf(t_stat, df)
    return 2 * stats.t.sf(np.abs(t_stat), df)


def _adjusted_p(procedure: str, t_stat: np.ndarray, raw: np.ndarray, fit: dict,
                alternative: str, ratios: np.ndarray) -> np.ndarray:
    """P с поправкой выбранной процедуры (для lsd — без поправки)."""
    m, df, t_levels = len(raw), fit["df"], len(fit["levels"])
    if procedure == "bonferroni":
        return np.minimum(raw * m, 1.0)
    if procedure == "fdr":
        return multipletests(raw, method="fdr_bh")[1]
    if procedure in ("tukey", "alsd"):
        k = t_levels if procedure == "tukey" else t_levels - 1
        return stats.studentized_range.sf(np.abs(t_stat) * np.sqrt(2), k, df)
    if procedure == "dunnett":
        oriented = t_stat if alternative != "less" else -t_stat
        stat = np.abs(t_stat) if alternative == "two-sided" else oriented
        return np.array([1 - dunnett_cdf(v, ratios, df, alternative == "two-sided")
                         for v in stat])
    return raw


# --- Метод --------------------------------------------------------------------


def _comparison_rows(pairs: list[tuple[int, int]], fit: dict, procedure: str,
                     alternative: str, alpha: float) -> list[dict]:
    """Разности μ_j − μ_i, SED, t, сырой и поправленный P, ДИ, значимость."""
    i_idx, j_idx = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    n = fit["n"]
    diff = fit["means"][j_idx] - fit["means"][i_idx]
    sed = np.sqrt(fit["s2"] * (1 / n[i_idx] + 1 / n[j_idx]))
    t_stat = diff / sed
    ratios = n[j_idx] / n[i_idx]
    crit = _critical(procedure, alpha, len(pairs), fit, alternative, ratios)
    raw = _raw_p(t_stat, fit["df"], alternative)
    adjusted = _adjusted_p(procedure, t_stat, raw, fit, alternative, ratios)
    inclusive = procedure == "fdr"  # правило книги: m·P(k)/k ≤ FDR
    rows = []
    for k, (i, j) in enumerate(pairs):
        rows.append({
            "group1": fit["levels"][i], "group2": fit["levels"][j],
            "mean_diff": float(diff[k]), "sed": float(sed[k]), "t": float(t_stat[k]),
            "p_unadjusted": float(raw[k]), "p_adjusted": float(adjusted[k]),
            "half_width": float(crit * sed[k]),
            "significant": bool(adjusted[k] <= alpha if inclusive else adjusted[k] < alpha),
        } | _interval(diff[k], crit * sed[k], alternative))
    return rows


def _interval(diff: float, half: float, alternative: str) -> dict:
    """ДИ разности; односторонний — с одной границей (другая None)."""
    return {"ci_lower": None if alternative == "less" else float(diff - half),
            "ci_upper": None if alternative == "greater" else float(diff + half)}


def _observed_fdr(rows: list[dict], alpha: float) -> float | None:
    """Наблюдаемый FDR = m·α / s по сырым P (стр. 197); None, если значимых нет."""
    s = sum(r["p_unadjusted"] < alpha for r in rows)
    return len(rows) * alpha / s if s else None


def mean_comparisons(
    values: list[float],
    group: list[str],
    block: list[str] | None = None,
    procedure: str = "tukey",
    control: str = "",
    alternative: str = "two-sided",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Попарные сравнения или сравнения с контролем с выбранной поправкой (8.8)."""
    settings = _settings(alpha, transform_kind, offset)
    result, fit = one_factor_fit(values, group, block, settings, "mean_comparisons",
                                 column_names)
    if fit is None:
        return result
    try:
        _check_procedure(procedure, control, alternative, len(fit["levels"]))
        pairs = _pairs(fit["levels"], control)
    except ValueError as exc:
        _design_error(result, "comparison_invalid", str(exc))
        return result
    rows = _comparison_rows(pairs, fit, procedure, alternative, alpha)
    _fill_result(rows, fit, procedure, settings, result)
    result.parameters.update({"procedure": procedure, "control": control or None,
                              "alternative": alternative})
    return result


def _fill_result(rows: list[dict], fit: dict, procedure: str, settings: dict,
                 result: StatResult) -> None:
    """Таблица сравнений, средние в estimates, сводка и подсказки."""
    alpha = settings["alpha"]
    means = means_table(fit, alpha)
    for row in means:
        result.estimates.append(Estimate(
            name=row["level"], value=row["mean"], ci_lower=row["ci_lower"],
            ci_upper=row["ci_upper"], ci_level=1 - alpha, std_error=row["se"]))
    result.parameters.update({"comparisons": rows, "means": means})
    result.statistics.update({
        "n_comparisons": len(rows),
        "n_significant": sum(r["significant"] for r in rows),
        "n_significant_unadjusted": sum(r["p_unadjusted"] < alpha for r in rows),
    })
    observed = _observed_fdr(rows, alpha)
    if observed is not None:
        result.statistics["observed_fdr"] = observed
    if procedure == "lsd" and len(rows) > 1:
        result.add_warning(
            "unadjusted_comparisons",
            "Сравнения без поправки на множественность: экспериментная ошибка I рода "
            "растёт с числом сравнений. Смотри наблюдаемый FDR.",
            Severity.INFO)
