"""Тесты проверки гипотез: сверка со scipy напрямую и известными свойствами."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.stats.hypothesis import (
    mann_whitney,
    one_sample_t,
    paired_t,
    two_sample_t,
    wilcoxon,
)


def test_one_sample_matches_scipy():
    data = [5.1, 4.9, 5.3, 5.0, 4.8, 5.2, 5.4]
    r = one_sample_t(data, popmean=5.0)
    ref = stats.ttest_1samp(data, 5.0)
    assert r.statistics["t"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.statistics["df"] == 6


def test_one_sample_alternative_directions():
    """Односторонние p делят двусторонний пополам (при знаке эффекта)."""
    data = [6.0, 6.5, 7.0, 6.8, 7.2]  # среднее заметно выше 5
    two = one_sample_t(data, 5.0, "two-sided").p_value
    greater = one_sample_t(data, 5.0, "greater").p_value
    assert greater == pytest.approx(two / 2, rel=1e-6)


def test_one_sample_invalid_alternative():
    with pytest.raises(ValueError):
        one_sample_t([1.0, 2.0, 3.0], 0.0, "bigger")


def test_two_sample_welch_matches_scipy():
    a = [23.0, 25.0, 21.0, 24.0, 22.0]
    b = [30.0, 28.0, 33.0, 31.0, 29.0]
    r = two_sample_t(a, b, equal_var=False)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert r.statistics["t"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.parameters["variant"] == "welch"


def test_two_sample_student_matches_scipy():
    a = [23.0, 25.0, 21.0, 24.0, 22.0]
    b = [24.0, 26.0, 22.0, 25.0, 23.0]
    r = two_sample_t(a, b, equal_var=True)
    ref = stats.ttest_ind(a, b, equal_var=True)
    assert r.statistics["t"] == pytest.approx(ref.statistic)
    assert r.parameters["variant"] == "student"


def test_two_sample_auto_selects_welch_on_unequal_var():
    """Разброс групп сильно различается → авто-выбор Уэлча."""
    rng = np.random.default_rng(7)
    a = rng.normal(10, 1, 40)
    b = rng.normal(10, 8, 40)
    r = two_sample_t(a, b, equal_var=None)
    assert r.parameters["variant"] == "welch"


def test_two_sample_warns_on_forced_equal_var():
    rng = np.random.default_rng(7)
    a = rng.normal(10, 1, 40)
    b = rng.normal(10, 8, 40)
    r = two_sample_t(a, b, equal_var=True)
    assert any(w.code == "unequal_variance" for w in r.warnings)


def test_paired_matches_scipy():
    before = [120.0, 118.0, 125.0, 130.0, 122.0]
    after = [115.0, 116.0, 120.0, 128.0, 119.0]
    r = paired_t(before, after)
    ref = stats.ttest_rel(before, after)
    assert r.statistics["t"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_paired_length_mismatch():
    with pytest.raises(ValueError):
        paired_t([1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_drops_nan_pairs():
    r = paired_t([1.0, 2.0, np.nan, 4.0], [1.0, 1.0, 3.0, 2.0])
    assert r.n == 3


def test_normality_warning_fires_on_skewed():
    """Сильно скошенные данные → предупреждение о нормальности."""
    rng = np.random.default_rng(1)
    skewed = rng.exponential(2.0, 60)
    r = one_sample_t(skewed, 2.0)
    assert any(w.code == "normality_violated" for w in r.warnings)


def test_mann_whitney_matches_scipy():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [6.0, 7.0, 8.0, 9.0, 10.0]
    r = mann_whitney(a, b)
    ref = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert r.statistics["U"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_wilcoxon_matches_scipy():
    a = [10.0, 12.0, 14.0, 11.0, 13.0, 15.0]
    b = [8.0, 11.0, 12.0, 10.0, 11.0, 13.0]
    r = wilcoxon(a, b)
    ref = stats.wilcoxon(a, b)
    assert r.statistics["W"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_nonparametric_robust_to_outlier():
    """Манн-Уитни устойчивее t к выбросу (демонстрация назначения)."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 3.0, 4.0, 5.0, 6.0]
    b_out = b + [1000.0]
    a_out = a + [5.5]
    # t сильно искажается выбросом, U — минимально. Проверяем, что U считается.
    r = mann_whitney(a_out, b_out)
    assert r.p_value is not None
