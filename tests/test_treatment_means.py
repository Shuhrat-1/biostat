"""Тесты средних групп с SEM, SED и LSD (гл. 4.4): пример книги, statsmodels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from core.report.schema import Severity
from core.stats.anova import one_way_anova, treatment_means

# Welham et al. (2015), Table 4.1 (стр. 72), файл calcium.dat:
# длина корней (см), 4 концентрации кальция по 5 горшков, CRD.
CALCIUM = {
    "A": [58, 52, 74, 58, 79],
    "B": [80, 68, 72, 74, 85],
    "C": [49, 70, 72, 74, 71],
    "D": [47, 49, 45, 48, 38],
}


def _calcium_means(alpha: float = 0.05):
    return treatment_means(*CALCIUM.values(), labels=list(CALCIUM), alpha=alpha)


def _est(result, name):
    return next(e for e in result.estimates if e.name == name)


def _pair(result, g1, g2):
    return next(
        c for c in result.parameters["comparisons"]
        if c["group1"] == g1 and c["group2"] == g2
    )


def test_book_example_4_1d_anova_table():
    """Table 4.5 (стр. 83): существующий one_way_anova воспроизводит книгу."""
    r = one_way_anova(*CALCIUM.values(), labels=list(CALCIUM))
    s = r.statistics
    assert s["ss_between"] == pytest.approx(2462.95)
    assert s["ss_within"] == pytest.approx(1221.60)
    assert s["ss_total"] == pytest.approx(3684.55)
    assert s["ms_between"] == pytest.approx(820.98, abs=5e-3)
    assert s["ms_within"] == pytest.approx(76.35)
    assert s["F"] == pytest.approx(10.753, abs=5e-4)
    assert (s["df_between"], s["df_within"]) == (3, 16)
    assert r.p_value == pytest.approx(0.00041, abs=5e-6)


def test_book_example_4_1e_sem_and_ci():
    """Example 4.1E (стр. 85): SEM = 3.91, 95 % ДИ для A = (55.9, 72.5)."""
    r = _calcium_means()
    a = _est(r, "mean_A")
    assert a.value == pytest.approx(64.2)
    assert a.std_error == pytest.approx(3.91, abs=5e-3)
    assert (a.ci_lower, a.ci_upper) == pytest.approx((55.9, 72.5), abs=0.05)
    assert r.statistics["residual_ms"] == pytest.approx(76.35)
    assert r.statistics["df_resid"] == 16
    assert r.statistics["t_crit"] == pytest.approx(2.120, abs=5e-4)


def test_book_example_4_1f_sed_lsd_difference():
    """Example 4.1F (стр. 87): SED = 5.53, LSD = 11.72, B − A = 11.6, ДИ (−0.1, 23.3)."""
    r = _calcium_means()
    assert r.statistics["sed"] == pytest.approx(5.53, abs=5e-3)
    assert r.statistics["lsd"] == pytest.approx(11.72, abs=5e-3)
    ba = _pair(r, "A", "B")
    assert ba["mean_diff"] == pytest.approx(11.6)
    assert (ba["ci_lower"], ba["ci_upper"]) == pytest.approx((-0.1, 23.3), abs=0.05)
    assert ba["p_unadjusted"] > 0.05  # разность чуть меньше LSD (стр. 87)


def _ols_reference(groups: dict[str, list[float]]):
    """Независимый эталон: OLS y ~ C(g) − 1 даёт средние и общий s²."""
    frame = pd.DataFrame(
        [(lab, v) for lab, vals in groups.items() for v in vals], columns=["g", "y"]
    )
    return smf.ols("y ~ C(g) - 1", data=frame).fit()


def test_unequal_replication_matches_statsmodels():
    """При неравной повторности SEM, ДИ средних и p разностей = OLS-контрасты."""
    rng = np.random.default_rng(11)
    groups = {"a": rng.normal(10, 2, 4), "b": rng.normal(12, 2, 7), "c": rng.normal(9, 2, 5)}
    r = treatment_means(*groups.values(), labels=list(groups), alpha=0.1)
    ref = _ols_reference(groups)
    conf = ref.conf_int(alpha=0.1)
    for lab in groups:
        est, key = _est(r, f"mean_{lab}"), f"C(g)[{lab}]"
        assert est.value == pytest.approx(ref.params[key])
        assert est.std_error == pytest.approx(ref.bse[key])
        assert (est.ci_lower, est.ci_upper) == pytest.approx(tuple(conf.loc[key]))
    tt = ref.t_test("C(g)[c] - C(g)[a] = 0")
    ac = _pair(r, "a", "c")
    assert ac["sed"] == pytest.approx(float(np.ravel(tt.sd)[0]))
    assert ac["p_unadjusted"] == pytest.approx(float(np.ravel(tt.pvalue)[0]))
    assert "sed" not in r.statistics  # общего SED нет при неравной повторности


def test_two_groups_equal_pooled_t_test():
    """Две группы: разность и p совпадают с t-тестом Стьюдента (стр. 86)."""
    from scipy import stats

    a, b = [7, 9, 6, 12, 4, 6, 9], [12, 8, 12, 9, 8, 16, 7]  # Example 2.1C
    pair = treatment_means(a, b).parameters["comparisons"][0]
    assert pair["t"] == pytest.approx(stats.ttest_ind(b, a).statistic)
    assert pair["p_unadjusted"] == pytest.approx(0.109, abs=5e-4)


def test_multiplicity_warning_only_for_three_plus_groups():
    assert "unadjusted_comparisons" in {w.code for w in _calcium_means().warnings}
    two = treatment_means([1, 2, 3], [2, 4, 5])
    assert "unadjusted_comparisons" not in {w.code for w in two.warnings}


def test_number_of_comparisons():
    """t групп → t(t − 1)/2 пар: 4 группы → 6 (стр. 198)."""
    assert len(_calcium_means().parameters["comparisons"]) == 6


def test_result_is_json_safe():
    json.dumps(_calcium_means().to_dict(), allow_nan=False)


def test_no_within_group_variation_is_error():
    r = treatment_means([5, 5, 5], [7, 7, 7])
    assert any(
        w.code == "no_residual_variation" and w.severity == Severity.ERROR
        for w in r.warnings
    )
    assert r.estimates == []


def test_group_too_small_is_error():
    r = treatment_means([1, 2, 3], [4])
    assert "n_too_small" in {w.code for w in r.warnings}


def test_single_group_is_error():
    assert "too_few_groups" in {w.code for w in treatment_means([1, 2, 3]).warnings}


def test_nan_values_dropped():
    r = treatment_means([1, 2, np.nan, 3], [4, 5, 6])
    assert r.n == 6


def test_bad_alpha_raises():
    with pytest.raises(ValueError):
        treatment_means([1, 2, 3], [4, 5, 6], alpha=1.5)


def test_label_mismatch_raises():
    with pytest.raises(ValueError):
        treatment_means([1, 2, 3], [4, 5, 6], labels=["only_one"])
