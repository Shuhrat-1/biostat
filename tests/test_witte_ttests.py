"""Book-тесты Witte & Witte (11th ed., 2017), гл. 13–15: t-тесты и размер эффекта.

Методы — из Welham (гл. 2); Witte добавляет ДИ разности и d Коэна новыми
ключами (пакет W2). Страница PDF = печатная + 18.
"""

from __future__ import annotations

import math

import pytest

from core.stats.hypothesis import (
    hedges_correction,
    one_sample_t,
    paired_t,
    two_sample_t,
)
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_zero_pooled_sd_gives_nan_d():
    """Без разброса d не определён: NaN (в JSON — null), а не деление на ноль.

    Предупреждения scipy (ttest_ind, shapiro) на константных данных ожидаемы;
    их текст и класс зависят от версии scipy, поэтому фильтр — по классу.
    """
    res = two_sample_t([1, 1, 1], [2, 2, 2], equal_var=True)
    # Аудит 02-04: без разброса внутри групп — отказ с кодом, а не пустые t и P.
    assert [w.code for w in res.warnings] == ["no_residual_variation"]
    assert "cohens_d" not in res.statistics and res.p_value is None


def test_confidence_validated():
    with pytest.raises(ValueError):
        two_sample_t(W.EPO_X1, W.EPO_X2, confidence=95)


def test_epo_paired_t_table_15_2():
    """Witte 15.4, стр. 278–280 (Table 15.2): D̄ = 5, SS_D = 14, s_D = 1.67, s_D̄ = 0.68,
    t = 7.35, df = 5, p < .001 (односторонний). Книга округляет s_D̄ до 0.68:
    точно s_D̄ = 0.683, t = 7.319."""
    res = paired_t(W.EPO_X1, W.EPO_X2, alternative="greater")
    assert res.statistics["df"] == 5
    assert res.statistics["mean_diff"] == 5
    assert res.statistics["sd_diff"] ** 2 * 5 == pytest.approx(14)  # SS_D = s²_D · (n − 1)
    assert round(res.statistics["sd_diff"], 2) == 1.67
    assert round(res.statistics["se_diff"], 2) == 0.68
    assert round(res.statistics["t"], 3) == 7.319
    assert res.p_value < 0.001


def test_epo_paired_ci_and_d_formulas_15_6_15_7():
    """Witte 15.6, стр. 282 (формула 15.6): 95 % ДИ = 5 ± 2.571 · 0.68 = 3.25…6.75
    (точно 3.244…6.756) — уже, чем −0.17…10.17 для независимых (стр. 260);
    стр. 283 (формула 15.7): d = 5 / 1.67 = 2.99 (точно 2.988)."""
    res = paired_t(W.EPO_X1, W.EPO_X2)
    assert res.statistics["ci_lower"] == pytest.approx(3.244, abs=5e-4)
    assert res.statistics["ci_upper"] == pytest.approx(6.756, abs=5e-4)
    assert round(res.statistics["cohens_dz"], 2) == 2.99
    est = next(e for e in res.estimates if e.name == "mean_diff")
    assert (est.ci_lower, est.ci_upper) == (res.statistics["ci_lower"], res.statistics["ci_upper"])


def test_vitamin_c_progress_checks_15_2_to_15_4():
    """Witte, стр. 281 (Progress Check 15.2), ответы стр. 440–441: D̄ = −1.5, s_D = 1.27,
    s_D̄ = 0.40, t = −3.75 (df 9, критическое −1.833); 15.3: ДИ −1.50 ± 2.262 · 0.40 =
    −2.40…−0.60; 15.4: d = 1.50 / 1.27 = 1.18. Точно: t = −3.737, ДИ −2.408…−0.592."""
    res = paired_t(W.VITAMIN_C, W.VITAMIN_FAKE, alternative="less")
    assert res.statistics["mean_diff"] == pytest.approx(-1.5)
    assert round(res.statistics["sd_diff"], 2) == 1.27
    assert round(res.statistics["t"], 3) == -3.737
    assert res.statistics["ci_lower"] is None  # односторонний «less»: нижней границы нет
    two = paired_t(W.VITAMIN_C, W.VITAMIN_FAKE)
    assert two.statistics["ci_lower"] == pytest.approx(-2.408, abs=5e-4)
    assert two.statistics["ci_upper"] == pytest.approx(-0.592, abs=5e-4)
    assert round(abs(two.statistics["cohens_dz"]), 2) == 1.18


def test_gpa_review_15_7_book_arithmetic_error():
    """Witte, стр. 288–289 (Review Question 15.7), ответ стр. 442: книга берёт ΣD² = .48,
    SS_D = .13, s_D = .14, t = 4.40, d = 1.57. Ошибка книги: ΣD² = .503, SS_D = .155,
    s_D = .161, t = 3.664, d = 1.39. Решение то же: t > 3.143 (α = .01, df 6)."""
    res = paired_t(W.GPA_EXERCISE, W.GPA_CONTROL, alternative="greater")
    assert round(res.statistics["mean_diff"], 2) == 0.22
    assert round(res.statistics["sd_diff"], 3) == 0.161
    assert round(res.statistics["t"], 3) == 3.664
    assert res.statistics["t"] > 3.143
    assert res.p_value < 0.01
    assert round(res.statistics["cohens_dz"], 2) == 1.39


def test_paired_confidence_validated():
    with pytest.raises(ValueError):
        paired_t(W.EPO_X1, W.EPO_X2, confidence=0)


def test_gas_mileage_one_sample_t_table_13_1():
    """Witte 13.3, стр. 238–240 (Table 13.1): X̄ = 43, s = 2.19, s_X̄ = 0.89, t = −2.25, df 5;
    H0 сохранена при α = .01 (критическое −3.365). Книга округляет s_X̄ до 0.89:
    точно s_X̄ = 0.894, t = −2.236."""
    res = one_sample_t(W.GAS_MILEAGE, 45, alternative="less")
    assert res.statistics["mean"] == 43
    assert round(res.statistics["sd"], 2) == 2.19
    assert round(res.statistics["se_mean"], 2) == 0.89
    assert round(res.statistics["t"], 3) == -2.236
    assert res.p_value > 0.01
    assert res.statistics["ci_lower"] is None  # односторонний «less»: только верхняя граница


def test_gas_mileage_ci_formula_13_3():
    """Witte 13.8, стр. 242 (формула 13.3): 95 % ДИ = 43 ± 2.571 · 0.89 = 40.71…45.29;
    точно 40.701…45.299. ДИ — для μ, а не для разности X̄ − μ_hyp."""
    res = one_sample_t(W.GAS_MILEAGE, 45)
    assert res.statistics["ci_lower"] == pytest.approx(40.701, abs=5e-4)
    assert res.statistics["ci_upper"] == pytest.approx(45.299, abs=5e-4)
    est = next(e for e in res.estimates if e.name == "mean")
    assert (est.value, est.ci_lower, est.ci_upper) == (
        43, res.statistics["ci_lower"], res.statistics["ci_upper"],
    )


def test_beef_progress_checks_13_2_to_13_4():
    """Witte, стр. 239, 241 (Progress Checks 13.2–13.4), ответы стр. 436: X̄ = 14.7, s = .82,
    s_X̄ = .26; t = (14.7 − 16) / .26 = −5.00 (df 9, критическое −1.833);
    95 % ДИ 14.7 ± 2.26 · .26 = 14.11…15.29. Точно t = −4.993."""
    res = one_sample_t(W.BEEF_WEIGHTS, 16, alternative="less")
    assert res.statistics["mean"] == pytest.approx(14.7)
    assert round(res.statistics["sd"], 2) == 0.82
    assert round(res.statistics["se_mean"], 2) == 0.26
    assert round(res.statistics["t"], 3) == -4.993
    two = one_sample_t(W.BEEF_WEIGHTS, 16)
    assert round(two.statistics["ci_lower"], 2) == 14.11
    assert round(two.statistics["ci_upper"], 2) == 15.29


def test_one_sample_d_in_sd_units():
    """[предл.: Cohen 1988] d = (X̄ − μ_hyp) / s; Witte, стр. 213: эффект 7 при σ = 15
    (≈ половина SD) — «средний» по Коэну. Бензин: d = −2 / 2.19 = −0.91."""
    res = one_sample_t(W.GAS_MILEAGE, 45)
    assert round(res.statistics["cohens_d"], 2) == -0.91


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_one_sample_zero_sd_gives_nan_d():
    """s = 0 → d не определён: NaN (в JSON — null); предупреждения scipy ожидаемы."""
    res = one_sample_t([107.0, 107.0, 107.0], 100)
    # Аудит 02-04: s = 0 — отказ с кодом (раньше P = 0 при пустом t).
    assert [w.code for w in res.warnings] == ["no_variation"]
    assert res.statistics == {} and res.p_value is None


def test_one_sample_confidence_validated():
    with pytest.raises(ValueError):
        one_sample_t(W.GAS_MILEAGE, 45, confidence=1)
