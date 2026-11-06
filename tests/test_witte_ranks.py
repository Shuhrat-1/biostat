"""Book-тесты Witte & Witte (11th ed., 2017), гл. 20: критерии для рангов.

mann_whitney, wilcoxon, kruskal_wallis — из Welham; Witte добавляет суммы рангов,
U и T в форме книги, сдвиг Ходжеса–Лемана [предл.] новыми ключами (пакет W2).
Страница PDF = печатная + 18.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.stats.hodges_lehmann import walsh_averages
from core.stats.hypothesis import mann_whitney, wilcoxon

TV_FAVORABLE = [4, 5, 5, 5, 10, 12, 20, 49]  # Table 20.2 (стр. 389), группа 1
TV_UNFAVORABLE = [0, 0, 1, 2, 14, 42, 43]  # группа 2


def test_tv_mann_whitney_table_20_2():
    """Witte 20.3, стр. 389 (Table 20.2): R₁ = 72, R₂ = 48, U₁ = 20, U₂ = 36, U = 20 > 10 (крит.,
    n₁ = 8, n₂ = 7) → H0 сохранена (стр. 391)."""
    res = mann_whitney(TV_FAVORABLE, TV_UNFAVORABLE)
    st = res.statistics
    assert (st["rank_sum_a"], st["rank_sum_b"]) == (72, 48)
    assert st["U_min"] == 20
    assert st["U"] == 36  # scipy: число пар a > b (= U₂ книги)
    assert res.p_value > 0.05


def test_therapy_progress_check_20_2():
    """Witte, стр. 392 (Progress Check 20.2), ответ стр. 455: U₁ = 31, U₂ = 5, U = 5 = крит. 5 →
    H0 отвергнута, «p = .05». Нормальное приближение scipy (совпадения 4.5): P = .045."""
    res = mann_whitney([1, 2, 4.5, 11, 3, 4.5], [9, 6, 12, 10, 7, 8])
    assert res.statistics["U_min"] == 5
    assert res.p_value == pytest.approx(0.04495, abs=1e-5)


def test_tv_hodges_lehmann_shift():
    """[предл.: Hodges & Lehmann 1963] Сдвиг группы 1 относительно 2 — медиана 56 разностей = 4;
    совпадения (0, 0; 5, 5, 5) → ДИ по нормальному приближению; содержит 0 (H0 сохранена)."""
    res = mann_whitney(TV_FAVORABLE, TV_UNFAVORABLE)
    st = res.statistics
    assert st["hl_shift"] == 4
    assert res.parameters["hl_ci_method"] == "normal"
    assert st["hl_ci_lower"] < 0 < st["hl_ci_upper"]
    est = next(e for e in res.estimates if e.name == "hl_shift")
    assert (est.value, est.ci_lower, est.ci_upper) == (4, st["hl_ci_lower"], st["hl_ci_upper"])


def test_mann_whitney_confidence_validated():
    with pytest.raises(ValueError):
        mann_whitney([1, 2], [3, 4], confidence=95)


# Table 20.3 (стр. 393): оценки времени у TV пар сиблингов; F — «благоприятно», U — «неблагоприятно».
SIB_FAVORABLE = [2, 11, 10, 6, 7, 43, 33, 5]
SIB_UNFAVORABLE = [0, 5, 12, 6, 2, 33, 25, 2]


@pytest.mark.filterwarnings("ignore::UserWarning")  # scipy 1.12: нули → нормальное приближение
def test_tv_wilcoxon_table_20_3():
    """Witte 20.4, стр. 393 (Table 20.3): разность D (пара D) = 0 отброшена, n = 7, R⁺ = 26.5,
    R⁻ = 1.5, T = 1.5 ≤ крит. 2 → H0 отвергнута (стр. 395)."""
    res = wilcoxon(SIB_FAVORABLE, SIB_UNFAVORABLE)
    st = res.statistics
    assert (st["r_plus"], st["r_minus"], st["n_nonzero"]) == (26.5, 1.5, 7)
    assert st["T_min"] == 1.5 == st["W"]
    assert st["r_plus"] + st["r_minus"] == 7 * 8 / 2  # проверка книги
    assert res.p_value < 0.05


@pytest.mark.filterwarnings("ignore::UserWarning")  # scipy 1.12: нули → нормальное приближение
def test_smoking_progress_check_20_3():
    """Witte, стр. 395–396 (Progress Check 20.3), ответ стр. 455–456: n = 8 (курильщик F без
    изменений), R⁺ = 34, R⁻ = 2, T = 2 ≤ 5 (направленный, α = .05)."""
    before = [22, 15, 10, 21, 14, 3, 11, 8, 15]
    after = [14, 7, 0, 22, 10, 3, 10, 7, 12]
    res = wilcoxon(before, after, alternative="greater")
    st = res.statistics
    assert (st["r_plus"], st["r_minus"], st["n_nonzero"], st["T_min"]) == (34, 2, 8, 2)
    assert res.p_value < 0.05


@pytest.mark.filterwarnings("ignore::UserWarning")  # scipy 1.12: нули → нормальное приближение
def test_tv_wilcoxon_hodges_lehmann():
    """[предл.: Hodges & Lehmann 1963] Центр разностей — медиана 36 средних Уолша (с нулевой
    разностью); совпадения |D| → нормальное приближение ДИ."""
    d = np.array(SIB_FAVORABLE) - np.array(SIB_UNFAVORABLE)
    res = wilcoxon(SIB_FAVORABLE, SIB_UNFAVORABLE)
    assert res.statistics["hl_shift"] == pytest.approx(np.median(walsh_averages(d)))
    assert res.parameters["hl_ci_method"] == "normal"
    assert res.statistics["hl_ci_lower"] <= res.statistics["hl_shift"] <= res.statistics["hl_ci_upper"]


def test_wilcoxon_confidence_validated():
    with pytest.raises(ValueError):
        wilcoxon([1, 2], [3, 4], confidence=0)
