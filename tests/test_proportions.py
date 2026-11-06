"""Доли (пакет W3, все методы [предл.]): one_proportion, two_proportions, mcnemar_test.

Эталоны — statsmodels / scipy и числа Witte там, где книга их даёт (12.7, гл. 19).
Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from core.stats.categorical import contingency_test
from core.stats.goodness_of_fit import chi2_goodness_of_fit
from core.stats.proportions import one_proportion, two_proportions

# --- one_proportion ---------------------------------------------------------------------


def test_margin_of_error_section_12_7():
    """Witte 12.7, стр. 228: 64 % из 1500, «±3 %» при 95 %. Точно по Вальду ±2.43 %
    (книга округляет вверх до целого процента); ДИ Уилсона [предл.: Wilson 1927] 61.5…66.4 %."""
    st = one_proportion(["yes", "no"], [960, 540], success="yes").statistics
    assert st["proportion"] == pytest.approx(0.64)
    assert st["margin_of_error_wald"] == pytest.approx(1.959964 * (0.64 * 0.36 / 1500) ** 0.5)
    assert round(st["margin_of_error_wald"], 3) == 0.024
    assert (st["ci_lower"], st["ci_upper"]) == pytest.approx(
        proportion_confint(960, 1500, method="wilson"))


def test_margin_shrinks_with_n_section_12_7():
    """Witte, стр. 228: n ≈ 500 → ±5 %, n ≈ 100 → ±10 % (худший случай p = .5: ±4.4 % и ±9.8 %)."""
    m500 = one_proportion(["a", "b"], [250, 250], success="a").statistics["margin_of_error_wald"]
    m100 = one_proportion(["a", "b"], [50, 50], success="a").statistics["margin_of_error_wald"]
    assert round(m500, 3) == 0.044
    assert round(m100, 3) == 0.098


def test_coin_review_19_11_score_equals_chi2():
    """Witte, стр. 383 (Review 19.11): 30 орлов из 50, H0: p = .5. Score-тест: z² = χ²(1) = 2.0,
    P = .157 — то же, что chi2_goodness_of_fit; точный биномиальный P = .203."""
    res = one_proportion(["heads", "tails"], [30, 20], success="heads")
    gof = chi2_goodness_of_fit(["heads", "tails"], [30, 20])
    assert res.statistics["z"] ** 2 == pytest.approx(gof.statistics["chi2"])
    assert res.statistics["p_score"] == pytest.approx(gof.p_value)
    assert res.statistics["binomial_p"] == pytest.approx(stats.binomtest(30, 50).pvalue)
    assert res.p_value == res.statistics["binomial_p"]


def test_exact_ci_and_one_sided():
    """[предл.: Clopper & Pearson 1934] точный ДИ; односторонний биномиальный P."""
    res = one_proportion(["s"] * 7 + ["f"] * 13, success="s", p0=0.2, alternative="greater")
    st = res.statistics
    assert (st["exact_ci_lower"], st["exact_ci_upper"]) == pytest.approx(
        proportion_confint(7, 20, method="beta"))
    assert st["binomial_p"] == pytest.approx(stats.binomtest(7, 20, 0.2, alternative="greater").pvalue)


def test_success_level_choice():
    """Без явного «успеха»: 1 / yes / да — по подсказке; иначе первый по порядку (info)."""
    auto = one_proportion(["0", "1", "1"])
    assert auto.parameters["success"] == "1"
    assert auto.warnings[0].code == "prop_success_chosen"
    plain = one_proportion(["b", "a", "a"])
    assert plain.parameters["success"] == "a"
    assert one_proportion(["a", "b"], success="c").warnings[0].code == "prop_unknown_success"
    assert one_proportion(["a", "b"], [1.5, 2]).warnings[0].code == "prop_bad_counts"
    with pytest.raises(ValueError):
        one_proportion(["a", "b"], p0=1)


# --- two_proportions --------------------------------------------------------------------

TITANIC = (["lived", "perished"] * 2, ["cabin", "cabin", "steerage", "steerage"],
           [299, 280, 186, 526])


def test_titanic_z_squared_equals_chi2_review_19_14():
    """Witte, стр. 384 (Review 19.14), ответ стр. 454: χ²(1) = 88.65 без поправки Йейтса.
    z² с объединённой долей = χ² 2 × 2 (contingency_test.chi2_uncorrected). Доли выживших:
    каюты 299/579 = .516, третий класс 186/712 = .261."""
    res = two_proportions(*TITANIC, success="lived")
    st = res.statistics
    assert (st["p1"], st["p2"]) == pytest.approx((299 / 579, 186 / 712))
    assert st["diff"] == pytest.approx(186 / 712 - 299 / 579)
    ref = contingency_test(TITANIC[1], TITANIC[0], count=TITANIC[2]).statistics
    assert st["chi2_equivalent"] == pytest.approx(ref["chi2_uncorrected"])
    assert st["chi2_equivalent"] == pytest.approx(88.65, abs=5e-3)
    assert res.parameters["groups"] == ["cabin", "steerage"]


def test_newcombe_ci_matches_statsmodels():
    """[предл.: Newcombe 1998] ДИ разности (группа 2 − группа 1) = statsmodels newcomb."""
    from statsmodels.stats.proportion import confint_proportions_2indep
    st = two_proportions(*TITANIC, success="lived").statistics
    lo, hi = confint_proportions_2indep(186, 712, 299, 579, method="newcomb", compare="diff")
    assert (st["ci_lower"], st["ci_upper"]) == pytest.approx((lo, hi))


def test_campus_progress_check_19_6_and_one_sided():
    """Witte, стр. 379 (Progress Check 19.6): кампус 51/70 против остальных 69/130, χ²(1) = 7.42.
    Односторонний «less» (во второй группе меньше) — половина двустороннего P."""
    data = (["ret", "not"] * 2, ["campus", "campus", "off", "off"], [51, 19, 69, 61])
    two = two_proportions(*data, success="ret")
    assert two.statistics["chi2_equivalent"] == pytest.approx(7.42, abs=5e-3)
    one = two_proportions(*data, success="ret", alternative="less")
    assert one.p_value == pytest.approx(two.p_value / 2)


def test_two_proportions_guards():
    assert two_proportions(["a", "b", "a"], ["x", "y", "z"]).warnings[0].code == "prop_two_groups"
    res = two_proportions(["a", "b", "a", "b"], ["x", "x", "y", "y"], success="a")
    assert res.parameters["comparisons"][0]["group2"] == "y"
