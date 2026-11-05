"""Book-тесты Witte & Witte (11th ed., 2017), гл. 13–15: t-тесты и размер эффекта.

Методы — из Welham (гл. 2); Witte добавляет ДИ разности и d Коэна новыми
ключами (пакет W2). Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest

from core.stats.hypothesis import hedges_correction, two_sample_t
from tests import witte_data as W


def test_epo_two_sample_t_table_14_1_and_sas_table_14_4():
    """Witte 14.6, стр. 252–254 (Table 14.1): s²p = 16.2, s_{X̄1−X̄2} = 2.32, t = 2.16, df = 10;
    стр. 267 (Table 14.4, SAS): Pooled t = 2.16, Pr > |t| = .0569. Точно t = 2.1517 —
    книга округляет стандартную ошибку до 2.32."""
    res = two_sample_t(W.EPO_X1, W.EPO_X2)
    assert res.parameters["variant"] == "student"  # Левене p = 1 → Стьюдент (не трогаем)
    assert res.statistics["df"] == 10
    assert res.statistics["mean_diff"] == 5
    assert res.statistics["se_diff"] ** 2 * 3 == pytest.approx(16.2)  # s²p · (1/6 + 1/6)
    assert round(res.statistics["se_diff"], 2) == 2.32
    assert round(res.statistics["t"], 2) == 2.15
    assert round(res.p_value, 4) == 0.0569


def test_epo_one_tailed_p_progress_check_14_9():
    """Witte, стр. 267 (Progress Check 14.9b; ответ стр. 438): односторонний p = .0569 / 2 = .0285."""
    res = two_sample_t(W.EPO_X1, W.EPO_X2, alternative="greater")
    assert round(res.p_value, 4) == 0.0285
    assert res.statistics["ci_upper"] is None  # односторонний ДИ: верхней границы нет


def test_epo_mean_diff_ci_formula_14_4_and_sas():
    """Witte 14.8, стр. 260 (формула 14.4): 95 % ДИ = 5 ± 2.228 · 2.32 = −0.17…10.17;
    стр. 267 (Table 14.4, SAS): Diff (1−2) 95 % CL −0.178…10.1777 (точно)."""
    res = two_sample_t(W.EPO_X1, W.EPO_X2)
    assert res.statistics["ci_lower"] == pytest.approx(-0.178, abs=5e-4)
    assert res.statistics["ci_upper"] == pytest.approx(10.1777, abs=5e-4)
    est = next(e for e in res.estimates if e.name == "mean_diff")
    assert (est.ci_lower, est.ci_upper, est.ci_level) == (
        res.statistics["ci_lower"], res.statistics["ci_upper"], 0.95,
    )


def test_epo_welch_ci_matches_satterthwaite_df():
    """Witte, стр. 267 (Table 14.4, SAS): Satterthwaite df = 9.88, Pr > |t| = .0572.
    ДИ при Уэлче — того же варианта, что и t."""
    res = two_sample_t(W.EPO_X1, W.EPO_X2, equal_var=False)
    assert round(res.statistics["df"], 2) == 9.88
    assert round(res.p_value, 4) == 0.0572
    assert res.statistics["ci_lower"] < -0.178  # шире, чем при объединённой дисперсии


def test_epo_cohens_d_formula_14_5():
    """Witte 14.9, стр. 264 (формула 14.5): d = 5 / √16.2 = 5 / 4.02 = 1.24;
    стр. 265 (Reports in the literature): «t(10) = 2.16, p < .05 and d = 1.24»."""
    res = two_sample_t(W.EPO_X1, W.EPO_X2)
    assert round(res.statistics["cohens_d"], 2) == 1.24
    # При Уэлче d остаётся по объединённому SD — как в книге.
    assert two_sample_t(W.EPO_X1, W.EPO_X2, equal_var=False).statistics["cohens_d"] == (
        pytest.approx(res.statistics["cohens_d"])
    )


def test_hedges_g_small_sample_correction():
    """[предл.: Hedges 1981] g = J · d, J(df) = Γ(df/2) / (√(df/2) Γ((df−1)/2)) ≈ 1 − 3/(4df − 1).
    EPO, df = 10: J = 0.9228, g = 1.146."""
    assert hedges_correction(10) == pytest.approx(1 - 3 / 39, abs=1e-3)
    assert hedges_correction(1000) == pytest.approx(1, abs=1e-3)
    res = two_sample_t(W.EPO_X1, W.EPO_X2)
    assert res.statistics["hedges_g"] == pytest.approx(res.statistics["cohens_d"] * 0.92275, abs=1e-4)


def test_zero_pooled_sd_gives_nan_d():
    """Без разброса d не определён: NaN (в JSON — null), а не деление на ноль."""
    res = two_sample_t([1, 1, 1], [2, 2, 2], equal_var=True)
    assert res.statistics["cohens_d"] != res.statistics["cohens_d"]


def test_confidence_validated():
    with pytest.raises(ValueError):
        two_sample_t(W.EPO_X1, W.EPO_X2, confidence=95)
