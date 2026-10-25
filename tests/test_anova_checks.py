"""Тесты проверки допущений ANOVA (гл. 5): Бартлетт, остатки, перестановки."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from core.report.schema import Severity
from core.stats.anova_checks import (
    anova_permutation_test,
    anova_residuals,
    bartlett_test,
)

# Welham et al. (2015), Table 4.1 (стр. 72): calcium pot trial (целые, как в файле).
CALCIUM = {
    "A": [58, 52, 74, 58, 79],
    "B": [80, 68, 72, 74, 85],
    "C": [49, 70, 72, 74, 71],
    "D": [47, 49, 45, 48, 38],
}


def _codes(result) -> set[str]:
    return {w.code for w in result.warnings}


# --- Бартлетт -----------------------------------------------------------------


def _bartlett_by_book(groups: list[list[float]]) -> float:
    """Формула книги (стр. 106): X² = [(N−t)log s²_p − Σ(n_j−1)log s²_j] / (1 + c)."""
    arrs = [np.asarray(g, float) for g in groups]
    t, big_n = len(arrs), sum(a.size for a in arrs)
    var = [a.var(ddof=1) for a in arrs]
    pooled = sum((a.size - 1) * v for a, v in zip(arrs, var)) / (big_n - t)
    c = (sum(1 / (a.size - 1) for a in arrs) - 1 / (big_n - t)) / (3 * (t - 1))
    num = (big_n - t) * np.log(pooled) - sum((a.size - 1) * np.log(v) for a, v in zip(arrs, var))
    return num / (1 + c)


def test_book_example_5_1e_bartlett():
    """Example 5.1E (стр. 107): s²_p = 76.35, c = 0.1042, X² = 3.633 < 7.815."""
    r = bartlett_test(*CALCIUM.values(), labels=list(CALCIUM))
    assert r.statistics["chi2"] == pytest.approx(3.633, abs=5e-4)
    assert r.statistics["df"] == 3
    assert r.statistics["pooled_variance"] == pytest.approx(76.35)
    assert r.parameters["group_variances"]["A"] == pytest.approx(135.20)
    assert r.statistics["chi2"] < stats.chi2.ppf(0.95, 3)  # 7.815
    assert "bartlett_unequal_variances" not in _codes(r)


def test_bartlett_matches_book_formula_unequal_n():
    rng = np.random.default_rng(21)
    groups = [rng.normal(0, 1, 6), rng.normal(0, 3, 9), rng.normal(0, 1.5, 4)]
    r = bartlett_test(*groups)
    assert r.statistics["chi2"] == pytest.approx(_bartlett_by_book(groups))


def test_bartlett_flags_unequal_variances():
    rng = np.random.default_rng(22)
    r = bartlett_test(rng.normal(0, 1, 30), rng.normal(0, 6, 30))
    assert r.p_value < 0.05
    assert "bartlett_unequal_variances" in _codes(r)


def test_bartlett_zero_variance_group_is_error():
    r = bartlett_test([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert r.p_value is None
    assert "zero_group_variance" in _codes(r)


# --- Остатки ------------------------------------------------------------------


def _std_resid(result, group, value):
    return next(
        row["std_residual"] for row in result.parameters["residuals"]
        if row["group"] == group and row["value"] == value
    )


def test_book_example_5_1f_outlier_candidate():
    """Example 5.1F (стр. 110): горшок C = 49 имеет стандартизованный остаток −2.33."""
    r = anova_residuals(*CALCIUM.values(), labels=list(CALCIUM))
    assert _std_resid(r, "C", 49.0) == pytest.approx(-2.33, abs=5e-3)
    assert r.statistics["n_abs_std_residual_gt_2"] == 1
    assert [row["value"] for row in r.parameters["potential_outliers"]] == [49.0]
    assert "potential_outliers" in _codes(r)


def test_residuals_sum_to_zero_within_groups():
    """Остатки внутри каждой группы в сумме дают 0 (стр. 95)."""
    r = anova_residuals(*CALCIUM.values(), labels=list(CALCIUM))
    for lab in CALCIUM:
        total = sum(row["residual"] for row in r.parameters["residuals"] if row["group"] == lab)
        assert total == pytest.approx(0.0, abs=1e-9)


def test_standardized_residuals_match_statsmodels_unequal_n():
    """Определение книги = internally studentized residuals OLS (стр. 96)."""
    rng = np.random.default_rng(23)
    groups = {"a": rng.normal(5, 1, 4), "b": rng.normal(7, 1, 8), "c": rng.normal(6, 1, 5)}
    r = anova_residuals(*groups.values(), labels=list(groups))
    frame = pd.DataFrame([(k, v) for k, vals in groups.items() for v in vals], columns=["g", "y"])
    ref = smf.ols("y ~ C(g)", data=frame).fit().get_influence().resid_studentized_internal
    ours = [row["std_residual"] for row in r.parameters["residuals"]]
    assert ours == pytest.approx(list(ref))


def test_normal_plot_uses_blom_positions():
    """i-я наименьшая — против 100(i − 0.375)/(n + 0.25)-го перцентиля (стр. 101)."""
    r = anova_residuals(*CALCIUM.values(), labels=list(CALCIUM))
    qq = r.parameters["plots"]["qq"]
    assert qq["theoretical_quantiles"][0] == pytest.approx(stats.norm.ppf(0.625 / 20.25))
    assert qq["sample_quantiles"] == sorted(qq["sample_quantiles"])


def test_residuals_json_safe():
    json.dumps(anova_residuals(*CALCIUM.values()).to_dict(), allow_nan=False)


def test_residuals_no_variation_is_error():
    r = anova_residuals([2.0, 2.0], [3.0, 3.0])
    assert any(w.code == "no_residual_variation" and w.severity == Severity.ERROR for w in r.warnings)


# --- Перестановочный тест -----------------------------------------------------


def _exact_permutation_p(groups: list[list[float]]) -> float:
    """Полный перебор раскладок для трёх групп по 2 (90 раскладок)."""
    values = np.concatenate([np.asarray(g, float) for g in groups])
    f_obs = stats.f_oneway(*groups).statistic
    idx = set(range(values.size))
    hits = total = 0
    for first in combinations(sorted(idx), 2):
        rest = sorted(idx - set(first))
        for second in combinations(rest, 2):
            third = sorted(set(rest) - set(second))
            f = stats.f_oneway(values[list(first)], values[list(second)], values[third]).statistic
            hits += f >= f_obs - 1e-12
            total += 1
    return hits / total


def test_exact_permutation_matches_enumeration():
    groups = [[1.0, 2.5], [4.0, 3.5], [6.0, 9.0]]
    r = anova_permutation_test(*groups, n_permutations=999, seed=1)
    assert r.parameters["exact"] is True
    assert r.p_value == pytest.approx(_exact_permutation_p(groups))


def test_permutation_calcium_agrees_with_f_test():
    """Calcium: F-тест P = 0.00041 (стр. 83); перестановочный тоже < 0.01."""
    r = anova_permutation_test(*CALCIUM.values(), labels=list(CALCIUM), seed=7)
    assert r.statistics["F"] == pytest.approx(10.753, abs=5e-4)
    assert r.p_value < 0.01
    assert r.parameters["exact"] is False


def test_permutation_reproducible_with_seed_and_seed_recorded():
    a = anova_permutation_test(*CALCIUM.values(), n_permutations=199, seed=42)
    b = anova_permutation_test(*CALCIUM.values(), n_permutations=199, seed=42)
    assert a.p_value == b.p_value and a.seed == 42
    auto = anova_permutation_test(*CALCIUM.values(), n_permutations=99)
    assert isinstance(auto.seed, int)


def test_permutation_warns_on_unequal_variances():
    rng = np.random.default_rng(24)
    r = anova_permutation_test(rng.normal(0, 1, 25), rng.normal(0, 8, 25), n_permutations=99, seed=1)
    assert "permutation_unequal_variances" in _codes(r)


def test_permutation_bad_count_raises():
    with pytest.raises(ValueError):
        anova_permutation_test([1, 2, 3], [4, 5, 6], n_permutations=0)


def test_permutation_json_safe():
    r = anova_permutation_test(*CALCIUM.values(), n_permutations=99, seed=3)
    json.dumps(r.to_dict(), allow_nan=False)


def test_permutation_skips_degenerate_levene_for_pairs():
    """Группы по 2: Левене вырожден (W = inf, p = 0) — не даём ложного сигнала."""
    r = anova_permutation_test([1.0, 2.5], [4.0, 3.5], [6.0, 9.0], seed=1)
    assert "levene_p" not in r.statistics
    assert "permutation_unequal_variances" not in _codes(r)
