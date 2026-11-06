"""welch_anova — F Уэлча для неравных дисперсий [Witte, стр. 316; предл.: Welch 1951].

Witte (гл. 16.13) называет «более сложную версию F для неравных дисперсий» без формулы;
сверка — с книжным Уэлчем для двух групп (Table 14.4, SAS) и с statsmodels.
Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest

from core.stats.anova import one_way_anova
from core.stats.oneway import welch_anova
from tests import witte_data as W


def test_two_groups_equals_welch_t_table_14_4():
    """Witte, стр. 267 (Table 14.4, SAS): t для неравных дисперсий, Satterthwaite df = 9.88,
    Pr > |t| = .0572. При k = 2 F Уэлча = t², df2 = df Саттертуэйта."""
    res = welch_anova(W.EPO_X1, W.EPO_X2)
    assert res.statistics["df_between"] == 1
    assert round(res.statistics["df_within"], 2) == 9.88
    assert round(res.p_value, 4) == 0.0572
    assert res.statistics["F"] == pytest.approx(5 ** 2 / (90 / 5 / 6 + 72 / 5 / 6))


def test_sleep_b_matches_statsmodels():
    """Witte, стр. 302 (Table 16.3), исход B: обычный F = 7.36, p = .024; дисперсии 4, 3, 4.
    F Уэлча = 5.782, df 2 и 3.980, p = .0665 (statsmodels anova_oneway, use_var="unequal"):
    при n = 3 df знаменателя падает с 6 до ≈ 4, тест становится строже."""
    res = welch_anova(*W.SLEEP_B, labels=W.SLEEP_LABELS)
    assert res.statistics["F"] == pytest.approx(5.781585, abs=1e-6)
    assert res.statistics["df_within"] == pytest.approx(3.980100, abs=1e-6)
    assert res.p_value == pytest.approx(0.066465, abs=1e-6)
    assert res.p_value > one_way_anova(*W.SLEEP_B).p_value
    assert [e.value for e in res.estimates] == [2, 5, 8]
    assert res.statistics["sd_24"] == pytest.approx(3 ** 0.5)


@pytest.mark.parametrize(
    ("groups", "f", "df2", "p"),
    [
        (W.SLEEP_UNEQUAL, 10.987097, 5.620374, 0.011429),  # Review 16.10, неравные n
        (W.WEIGHT_LOSS, 2.223177, 12.819171, 0.148290),  # Review 16.13
        (W.EYE_CONTACTS, 9.342831, 15.139921, 0.000971),  # Progress Check 16.5
    ],
)
def test_witte_ch16_data_match_statsmodels(groups, f, df2, p):
    """Witte, стр. 307–308, 319, 321: данные гл. 16; эталон — statsmodels 0.14 anova_oneway."""
    res = welch_anova(*groups)
    assert res.statistics["F"] == pytest.approx(f, abs=1e-6)
    assert res.statistics["df_within"] == pytest.approx(df2, abs=1e-6)
    assert res.p_value == pytest.approx(p, abs=1e-6)


def test_guards():
    assert welch_anova([1, 2, 3]).warnings[0].code == "too_few_groups"
    assert welch_anova([1, 2, 3], [4]).warnings[0].code == "n_too_small"
    zero = welch_anova([1, 2, 3], [5, 5, 5])
    assert zero.warnings[0].code == "welch_zero_variance"
    assert zero.p_value is None
    with pytest.raises(ValueError):
        welch_anova([1, 2], [3, 4], labels=["a"])
