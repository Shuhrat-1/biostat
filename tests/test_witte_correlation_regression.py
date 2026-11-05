"""Book-тесты Witte & Witte (11th ed., 2017), гл. 6–7, для существующих методов.

pearson_correlation, simple_linear_regression и regression_details — из Welham
(гл. 2, 12); здесь только сверка с числами Witte (страница PDF = печатная + 18).
"""

from __future__ import annotations

import pytest

from core.stats.correlation import pearson_correlation
from core.stats.linear_regression import regression_details
from core.stats.regression import simple_linear_regression
from tests import witte_data as W


def test_greeting_cards_r_table_6_3_and_spss_table_6_4():
    """Witte 6.4, стр. 118 (Table 6.3): SP = 64, SSx = SSy = 80, r = .80;
    стр. 121 (Table 6.4, SPSS): Sig. (2-tailed) = .104, N = 5."""
    res = pearson_correlation(W.CARDS_SENT, W.CARDS_RECEIVED)
    assert res.statistics["r"] == pytest.approx(0.80)
    assert res.statistics["covariance"] * 4 == pytest.approx(64)  # SP = cov · (n − 1)
    assert round(res.p_value, 3) == 0.104


def test_outliers_change_r_figure_6_6():
    """Witte 6.5, стр. 119 (Fig. 6.6): Doris (13, 14) заменена на A (33, 34) → r = .98,
    на B (13, 4) → r = .04."""
    a = pearson_correlation([33, *W.CARDS_SENT[1:]], [34, *W.CARDS_RECEIVED[1:]])
    b = pearson_correlation(W.CARDS_SENT, [4, *W.CARDS_RECEIVED[1:]])
    assert round(a.statistics["r"], 2) == 0.98 and round(b.statistics["r"], 2) == 0.04


def test_progress_check_6_5():
    """Witte Progress Check 6.5, стр. 117; ответ стр. 428: r = 4/√(4 · 9.33) = .65."""
    assert round(pearson_correlation(W.CHILDREN_F, W.CHILDREN_M).statistics["r"], 2) == 0.65


def test_least_squares_line_table_7_1():
    """Witte 7.3, стр. 131 (Table 7.1): b = .80, a = 6.40; r² = .64 (7.6, стр. 139)."""
    res = simple_linear_regression(W.CARDS_SENT, W.CARDS_RECEIVED)
    est = {e.name: e.value for e in res.estimates}
    assert est["slope"] == pytest.approx(0.80) and est["intercept"] == pytest.approx(6.40)
    assert res.statistics["r_squared"] == pytest.approx(0.64)


def test_standard_error_of_estimate_table_7_3():
    """Witte 7.4, стр. 134 (Table 7.3): s_y|x = √(SSy(1 − r²)/(n − 2)) = √9.60 = 3.10;
    SS_y|x = 28.8 (стр. 137)."""
    res = simple_linear_regression(W.CARDS_SENT, W.CARDS_RECEIVED)
    s = res.statistics["residual_std_error"]
    assert round(s, 2) == 3.10 and s ** 2 * 3 == pytest.approx(28.8)


def test_prediction_for_emma_and_its_error():
    """Witte 7.3, стр. 132: Y′(11) = .80 · 11 + 6.40 = 15.20, книга пишет «15.20 ± 3.10».

    Расхождение с Welham (стр. 301–305): SE нового наблюдения учитывает удалённость x от x̄ —
    √(s²(1 + 1/n + (x − x̄)²/SSx)) = 3.67, а не s_y|x = 3.10 (на 18 % больше). BioStat даёт
    SE по Welham; ±s_y|x Witte сам называет грубой оценкой (стр. 134).
    """
    pred = regression_details(W.CARDS_SENT, W.CARDS_RECEIVED, x_predict="11")
    row = pred.parameters["predictions"][0]
    assert row["fitted"] == pytest.approx(15.20)
    assert row["se_new"] == pytest.approx((9.60 * (1 + 1 / 5 + (11 - 7) ** 2 / 80)) ** 0.5)
    assert row["se_new"] > 3.10
