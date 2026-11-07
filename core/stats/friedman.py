"""Критерий Фридмана — ранговый аналог ANOVA повторных измерений.

Нет ни у Witte (Fig. 21.1 даёт ранговые аналоги только для независимых групп и
пар), ни у Welham — [предл.: Friedman 1937; Conover 1999, гл. 5.8].

Ранги — внутри субъекта (средние при совпадениях). R_j — сумма рангов условия.
Q = 12/(n·k·(k+1))·ΣR_j² − 3n(k+1), поправка на совпадения
C = 1 − Σ(t³ − t)/(n·k·(k² − 1)), χ² = Q/C на k − 1 df (как scipy).
W Кендалла = χ²/(n(k − 1)) — размер эффекта (0…1). F Имана–Давенпорта
(n − 1)χ²/(n(k − 1) − χ²) на k − 1 и (k − 1)(n − 1) df точнее χ² при малых n
[предл.: Iman & Davenport 1980].

Пары условий — знаково-ранговый Уилкоксон по разностям, P по Холму; сдвиг
Ходжеса–Лемана с ДИ по Бонферрони (как парный блок rm_anova). Длинный формат:
значение, условие, субъект; неполные субъекты исключаются.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from core.report.schema import Severity, StatResult
from core.stats.hodges_lehmann import hl_one_sample
from core.stats.repeated import _matrix

_SMALL_N = 10  # ниже — χ²-приближение грубое, смотреть F Имана–Давенпорта


def friedman_statistic(y: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Q, поправка на совпадения, χ², df, P, W Кендалла, F Имана–Давенпорта; суммы рангов."""
    n, k = y.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, y)
    r_sums = ranks.sum(axis=0)
    q = 12 / (n * k * (k + 1)) * float(np.sum(r_sums**2)) - 3 * n * (k + 1)
    ties = sum(float(np.sum(c**3 - c)) for c in
               (np.unique(row, return_counts=True)[1] for row in y))
    corr = 1 - ties / (n * k * (k**2 - 1))
    chi2 = q / corr if corr > 0 else float("nan")
    out = {"chi2": chi2, "Q_uncorrected": q, "tie_correction": corr, "df": k - 1,
           "p": float(stats.chi2.sf(chi2, k - 1)), "kendalls_w": chi2 / (n * (k - 1))}
    denom = n * (k - 1) - chi2
    if n > 1 and denom > 0:
        f = (n - 1) * chi2 / denom
        df2 = (k - 1) * (n - 1)
        out.update({"F_iman_davenport": f, "df2_iman_davenport": df2,
                    "p_iman_davenport": float(stats.f.sf(f, k - 1, df2))})
    return out, r_sums


def _signed_rank_p(d: np.ndarray) -> float:
    """Двусторонний P Уилкоксона; все разности нулевые — P = 1."""
    if np.all(d == 0):
        return 1.0
    with warnings.catch_warnings():  # scipy: нули/совпадения → нормальное приближение
        warnings.simplefilter("ignore")
        return float(stats.wilcoxon(d).pvalue)


def _pair_rows(y: np.ndarray, levels: list[str], alpha: float) -> list[dict]:
    """Уилкоксон для каждой пары, P по Холму, сдвиг ХЛ с ДИ по Бонферрони."""
    k = y.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    rows = []
    for i, j in pairs:
        d = y[:, j] - y[:, i]
        hl = hl_one_sample(d, alpha / len(pairs))
        rows.append({"group1": levels[i], "group2": levels[j], "p": _signed_rank_p(d),
                     "median_diff": float(np.median(d)), "hl_shift": hl["hl_shift"],
                     "ci_lower": hl["hl_ci_lower"], "ci_upper": hl["hl_ci_upper"],
                     "ci_method": hl["hl_ci_method"]})
    p_holm = multipletests([r["p"] for r in rows], alpha=alpha, method="holm")[1]
    for row, p_adj in zip(rows, p_holm):
        row.update({"p_adj": float(p_adj), "reject": bool(p_adj < alpha)})
    return rows


def _guard(n: int, k: int, dropped: int, y: np.ndarray, result: StatResult) -> bool:
    """Ошибки: < 2 условий или субъектов, все значения равны; предупреждения."""
    if dropped:
        result.add_warning(
            "rm_incomplete_subjects",
            f"Исключено субъектов с пропусками: {dropped}. При пропусках используйте "
            f"mixed_model (Welham, гл. 16) — он не требует полных данных.",
            Severity.WARNING, params={"dropped": dropped})
    if k < 2:
        result.add_warning("too_few_groups", "Нужно ≥ 2 условий", Severity.ERROR)
        return False
    if n < 2:
        result.add_warning("n_too_small", "Нужно ≥ 2 субъектов с полными данными",
                           Severity.ERROR)
        return False
    if np.all(y == y[:, :1]):
        result.add_warning("friedman_all_tied",
                           "У каждого субъекта все условия равны: ранги не различаются.",
                           Severity.ERROR)
        return False
    return True


def _notes(n: int, k: int, result: StatResult) -> None:
    """k = 2 — это знаковый тест; малое n — χ² грубое."""
    if k == 2:
        result.add_warning(
            "friedman_two_conditions",
            "Два условия: критерий Фридмана равносилен знаковому тесту; мощнее "
            "знаково-ранговый тест Уилкоксона (wilcoxon).", Severity.INFO)
    if n < _SMALL_N:
        result.add_warning(
            "friedman_small_n",
            f"Субъектов меньше {_SMALL_N}: χ²-приближение грубое, опирайтесь на F "
            f"Имана–Давенпорта.", Severity.INFO, params={"n": n})


def friedman(
    values: list[float],
    condition: list[str],
    subject: list[str],
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Критерий Фридмана: k условий у одних и тех же субъектов, ранги внутри субъекта."""
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")
    wide, levels, dropped = _matrix(values, condition, subject)
    y = wide.to_numpy(dtype=float)
    n, k = y.shape if y.size else (0, len(levels))
    result = StatResult(method="friedman", n=int(y.size),
                        parameters={"alpha": alpha, "labels": levels,
                                    "column_names": column_names or {}})
    if not _guard(n, k, dropped, y, result):
        return result
    st, rank_sums = friedman_statistic(y)
    result.p_value = st["p"]
    result.statistics = {**st, "n_subjects": n, "n_conditions": k,
                         "n_excluded_subjects": dropped}
    result.parameters.update({
        "ranks": [{"level": lv, "rank_sum": float(rs), "mean_rank": float(rs) / n,
                   "median": float(md)}
                  for lv, rs, md in zip(levels, rank_sums, np.median(y, axis=0))],
        "comparisons": _pair_rows(y, levels, alpha),
        "profiles": {"subjects": [str(s) for s in wide.index], "conditions": levels,
                     "values": y.tolist()},
    })
    _notes(n, k, result)
    return result
