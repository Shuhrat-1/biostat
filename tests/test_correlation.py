"""Тесты корреляции Пирсона: инварианты, сверка со scipy, пример книги."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from core.report.schema import Severity
from core.stats.correlation import pearson_correlation

# Welham et al. (2015), Table 2.1 (стр. 23), файл willow.dat:
# длина и ширина (мм) 50 жуков Phratora vulgatissima, по номерам 1–50.
LENGTH = [
    4.60, 4.70, 4.50, 4.55, 4.75, 4.40, 4.20, 4.70, 4.55, 4.70,
    4.65, 4.50, 4.30, 4.65, 4.75, 4.65, 4.45, 4.55, 4.60, 4.55,
    4.35, 4.45, 4.55, 4.35, 4.65, 4.50, 4.45, 4.45, 4.10, 4.50,
    4.60, 4.75, 4.70, 4.35, 4.60, 4.55, 4.775, 4.60, 4.45, 4.60,
    4.65, 4.725, 4.95, 4.65, 4.60, 4.45, 4.30, 4.75, 4.525, 4.35,
]
WIDTH = [
    1.50, 1.65, 1.55, 1.65, 1.65, 1.50, 1.70, 1.55, 1.60, 1.65,
    1.55, 1.55, 1.50, 1.65, 1.65, 1.60, 1.60, 1.50, 1.70, 1.60,
    1.60, 1.60, 1.55, 1.55, 1.65, 1.55, 1.50, 1.60, 1.40, 1.50,
    1.60, 1.575, 1.65, 1.55, 1.60, 1.65, 1.55, 1.60, 1.475, 1.60,
    1.625, 1.65, 1.725, 1.65, 1.625, 1.55, 1.525, 1.60, 1.55, 1.50,
]


def _codes(result) -> set[str]:
    return {w.code for w in result.warnings}


def test_book_example_2_3b_willow_beetles():
    """Example 2.3B (стр. 38–39): r = 0.5528, t = 4.596 на 48 df, P < 0.001.

    Ковариационная матрица из книги: s_LW = 0.0059 (точно 0.00586).
    """
    r = pearson_correlation(LENGTH, WIDTH)
    assert r.n == 50
    assert r.statistics["r"] == pytest.approx(0.5528, abs=5e-5)
    assert r.statistics["t"] == pytest.approx(4.596, abs=5e-4)
    assert r.statistics["df"] == 48
    assert r.statistics["covariance"] == pytest.approx(0.00586, abs=5e-6)
    assert r.p_value < 0.001


def test_perfect_positive_line():
    """Точная возрастающая прямая: r = 1, t не пишется, JSON валиден."""
    r = pearson_correlation([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    assert r.statistics["r"] == pytest.approx(1.0)
    assert "t" not in r.statistics
    assert "perfect_correlation" in _codes(r)
    json.dumps(r.to_dict(), allow_nan=False)  # не должно падать на inf/nan


def test_perfect_negative_line():
    r = pearson_correlation([1, 2, 3, 4], [8, 6, 4, 2])
    assert r.statistics["r"] == pytest.approx(-1.0)


def test_symmetric_in_x_y():
    """r(x, y) = r(y, x)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=25)
    y = 0.5 * x + rng.normal(size=25)
    assert pearson_correlation(x, y).statistics["r"] == pytest.approx(
        pearson_correlation(y, x).statistics["r"]
    )


def test_scale_invariant():
    """Смена единиц (см → дюймы) не меняет r (стр. 36–37)."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=30)
    y = x + rng.normal(size=30)
    base = pearson_correlation(x, y).statistics["r"]
    assert pearson_correlation(x / 2.54, y * 10 + 3).statistics["r"] == pytest.approx(base)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_matches_scipy_pearsonr(alternative):
    """r, p и ДИ совпадают со scipy.stats.pearsonr."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=40)
    y = 0.3 * x + rng.normal(size=40)
    r = pearson_correlation(x, y, alternative=alternative, ci_level=0.9)
    ref = stats.pearsonr(x, y, alternative=alternative)
    ci = ref.confidence_interval(confidence_level=0.9)
    est = r.estimates[0]
    assert est.value == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert (est.ci_lower, est.ci_upper) == pytest.approx((ci.low, ci.high))


def test_t_formula_matches_book_definition():
    """t = r·√(N−2)/√(1−r²) (стр. 37) и даёт тот же p, что pearsonr."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=20)
    y = x + rng.normal(size=20)
    r = pearson_correlation(x, y)
    p_from_t = 2 * stats.t.sf(abs(r.statistics["t"]), r.statistics["df"])
    assert r.p_value == pytest.approx(p_from_t)


def test_missing_pairs_dropped():
    """Пары с NaN в любой переменной удаляются целиком."""
    r = pearson_correlation([1, 2, np.nan, 4, 5], [2, 4, 6, np.nan, 9])
    assert r.n == 3
    assert "missing_dropped" in _codes(r)


def test_too_few_pairs_is_error():
    r = pearson_correlation([1, 2], [3, 4])
    assert r.p_value is None
    assert any(w.code == "n_too_small" and w.severity == Severity.ERROR for w in r.warnings)


def test_constant_variable_is_error():
    r = pearson_correlation([1, 2, 3, 4], [5, 5, 5, 5])
    assert r.p_value is None
    assert "no_variation_corr" in _codes(r)


def test_unequal_length_raises():
    with pytest.raises(ValueError):
        pearson_correlation([1, 2, 3], [1, 2])


def test_bad_alternative_raises():
    with pytest.raises(ValueError):
        pearson_correlation([1, 2, 3], [1, 2, 4], alternative="both")


def test_nonnormal_data_warns():
    """Сильно скошенные данные → предупреждение: тест приближённый (стр. 38)."""
    rng = np.random.default_rng(7)
    x = rng.exponential(size=60) ** 3
    y = x + rng.normal(size=60)
    assert "normality_violated" in _codes(pearson_correlation(x, y))
