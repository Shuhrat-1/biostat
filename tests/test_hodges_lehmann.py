"""Оценки Ходжеса–Лемана с ДИ [предл.: Hodges & Lehmann 1963] — инварианты, полный перебор
нулевых распределений и обращение точного теста (инструкция W2)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from core.stats.hodges_lehmann import (
    hl_one_sample,
    hl_two_sample,
    mw_null_counts,
    signrank_null_counts,
    walsh_averages,
)


@pytest.mark.parametrize(("m", "n"), [(1, 1), (3, 4), (5, 5), (2, 6)])
def test_mw_null_counts_match_enumeration(m, n):
    """Распределение U по динамике = полный перебор всех C(m + n, m) разбиений рангов."""
    ranks = np.arange(m + n)
    counts = np.zeros(m * n + 1)
    for idx in itertools.combinations(range(m + n), m):
        a, b = ranks[list(idx)], np.delete(ranks, list(idx))
        counts[int((a[:, None] > b[None, :]).sum())] += 1
    assert mw_null_counts(m, n) == pytest.approx(counts)


@pytest.mark.parametrize("n", [1, 4, 7, 10])
def test_signrank_null_counts_match_enumeration(n):
    """Распределение T⁺ = перебор всех 2ⁿ наборов знаков."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    for signs in itertools.product((0, 1), repeat=n):
        counts[sum((i + 1) * s for i, s in enumerate(signs))] += 1
    assert signrank_null_counts(n) == pytest.approx(counts)


def test_estimates_are_medians_by_enumeration():
    """Оценка — медиана всех разностей aᵢ − bⱼ / средних Уолша (перебор)."""
    a, b = np.array([3.1, 7.4, 2.2, 9.0]), np.array([1.0, 4.4, 0.3])
    assert hl_two_sample(a, b)["hl_shift"] == pytest.approx(np.median([x - y for x in a for y in b]))
    walsh = [(a[i] + a[j]) / 2 for i in range(4) for j in range(i, 4)]
    assert sorted(walsh) == pytest.approx(list(walsh_averages(a)))
    assert hl_one_sample(a)["hl_shift"] == pytest.approx(np.median(walsh))


def test_shift_and_swap_invariants():
    """Сдвиг a на c → оценка и ДИ + c; перестановка выборок → знак и границы меняются местами."""
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=9), rng.normal(size=7)
    base = hl_two_sample(a, b)
    moved = hl_two_sample(a + 2.5, b)
    for key in ("hl_shift", "hl_ci_lower", "hl_ci_upper"):
        assert moved[key] == pytest.approx(base[key] + 2.5)
    swapped = hl_two_sample(b, a)
    assert swapped["hl_shift"] == pytest.approx(-base["hl_shift"])
    assert (swapped["hl_ci_lower"], swapped["hl_ci_upper"]) == pytest.approx(
        (-base["hl_ci_upper"], -base["hl_ci_lower"]))
    one = hl_one_sample(a)
    assert hl_one_sample(a + 1)["hl_shift"] == pytest.approx(one["hl_shift"] + 1)
    assert hl_one_sample(-a)["hl_ci_lower"] == pytest.approx(-one["hl_ci_upper"])


@pytest.mark.parametrize("seed", range(15))
def test_exact_ci_inverts_exact_tests(seed):
    """Точный ДИ = множество сдвигов δ, при которых точный двусторонний тест не отвергает H0:
    чуть внутри границ P > .05, чуть снаружи P ≤ .05 (scipy, method="exact")."""
    rng = np.random.default_rng(seed)
    m, n = int(rng.integers(5, 10)), int(rng.integers(5, 10))
    a, b = rng.normal(size=m).round(4), (rng.normal(size=n) + 0.5).round(4)
    eps = 1e-7
    two = hl_two_sample(a, b)
    assert two["hl_ci_method"] == "exact"
    p = lambda d: stats.mannwhitneyu(a - d, b, method="exact").pvalue
    assert min(p(two["hl_ci_lower"] + eps), p(two["hl_ci_upper"] - eps)) > 0.05
    assert max(p(two["hl_ci_lower"] - eps), p(two["hl_ci_upper"] + eps)) <= 0.05
    assert two["hl_ci_coverage"] >= 0.95
    d = (rng.normal(size=int(rng.integers(6, 12))) + 0.3).round(4)  # n ≥ 6: ДИ ограничен
    one = hl_one_sample(d)
    q = lambda x: stats.wilcoxon(d - x, method="exact").pvalue
    assert min(q(one["hl_ci_lower"] + eps), q(one["hl_ci_upper"] - eps)) > 0.05
    assert max(q(one["hl_ci_lower"] - eps), q(one["hl_ci_upper"] + eps)) <= 0.05


def test_too_small_sample_gives_unbounded_ci():
    """n = 4: даже все разности одного знака дают P = .125 > .05 — ДИ не ограничен (None)."""
    out = hl_one_sample(np.array([1.0, 2.0, 3.5, 4.2]))
    assert (out["hl_ci_lower"], out["hl_ci_upper"]) == (None, None)


def test_ties_switch_to_normal_approximation():
    """Совпадения → нормальное приближение (точное нулевое распределение неверно)."""
    assert hl_two_sample(np.array([1, 2, 2, 3.0]), np.array([0, 1, 5.0]))["hl_ci_method"] == "normal"
    assert hl_one_sample(np.array([1, -1, 2, 3, 4, 5.0]))["hl_ci_method"] == "normal"
