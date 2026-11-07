"""Корреляция Спирмена (spearman_correlation): Witte, стр. 119 — r Пирсона по рангам."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from core.stats.correlation import pearson_correlation, spearman_correlation
from tests import witte_data as W


def test_witte_cards_rho_is_pearson_on_ranks():
    """Witte, Table 6.1 (стр. 108) и стр. 119: ранги отправленных 5,4,3,2,1 и полученных
    4,5,3,2,1 открыток; ΣD² = 2, ρ = 1 − 6·2/(5·24) = 0.9 = r Пирсона по рангам."""
    res = spearman_correlation(W.CARDS_SENT, W.CARDS_RECEIVED)
    st = res.statistics
    assert st["sum_d_squared"] == 2
    assert st["rho"] == pytest.approx(0.9)
    assert st["rho_no_ties_formula"] == pytest.approx(0.9)
    ranks = pearson_correlation(stats.rankdata(W.CARDS_SENT), stats.rankdata(W.CARDS_RECEIVED))
    assert st["rho"] == pytest.approx(ranks.statistics["r"])
    assert res.parameters["x"] == [5, 4, 3, 2, 1] and res.parameters["scatter_scale"] == "ranks"
    assert "spearman_small_n" in {w.code for w in res.warnings}
    json.dumps(res.to_dict(), allow_nan=False)


def test_matches_scipy_with_ties_and_fisher_ci():
    """С совпадениями — как scipy.spearmanr; ДИ Фишера с SE = √(1.06/(n − 3))."""
    rng = np.random.default_rng(3)
    x = rng.integers(1, 6, 30).astype(float)
    y = x + rng.integers(0, 4, 30)
    res = spearman_correlation(x, y, alternative="greater")
    ref = stats.spearmanr(x, y, alternative="greater")
    assert res.statistics["rho"] == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    half = stats.norm.ppf(0.975) * np.sqrt(1.06 / 27)
    assert res.statistics["ci_upper"] == pytest.approx(np.tanh(np.arctanh(ref.statistic) + half))
    assert "spearman_ties" in {w.code for w in res.warnings}


def test_monotonic_nonlinear_gives_rho_one():
    """y = x³: связь монотонная, ρ = 1, хотя r < 1; t и ДИ не пишутся (|ρ| = 1)."""
    x = np.arange(1, 11, dtype=float)
    res = spearman_correlation(x, x**3)
    assert res.statistics["rho"] == pytest.approx(1.0)
    assert "t" not in res.statistics and "ci_lower" not in res.statistics
    assert pearson_correlation(x, x**3).statistics["r"] < 1
    json.dumps(res.to_dict(), allow_nan=False)
