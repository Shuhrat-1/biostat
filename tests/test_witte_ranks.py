"""Book-тесты Witte & Witte (11th ed., 2017), гл. 20: критерии для рангов.

mann_whitney, wilcoxon, kruskal_wallis — из Welham; Witte добавляет суммы рангов,
U и T в форме книги, сдвиг Ходжеса–Лемана [предл.] новыми ключами (пакет W2).
Страница PDF = печатная + 18.
"""

from __future__ import annotations

import pytest

from core.stats.hypothesis import mann_whitney

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
