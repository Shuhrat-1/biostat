"""Критерий Фридмана (friedman) — [предл.: Friedman 1937; Conover 1999].

Эталон — пример RoundingTimes из справки R friedman.test (Hollander & Wolfe 1973):
22 игрока × 3 способа обежать первую базу; Friedman χ² = 11.143, df = 2,
P = 0.003805. Остальное — сверка со scipy и инварианты.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy import stats

from core.stats.friedman import friedman, friedman_statistic
from tests import witte_data as W

ROUNDING = np.array([
    [5.40, 5.50, 5.55], [5.85, 5.70, 5.75], [5.20, 5.60, 5.50], [5.55, 5.50, 5.40],
    [5.90, 5.85, 5.70], [5.45, 5.55, 5.60], [5.40, 5.40, 5.35], [5.45, 5.50, 5.35],
    [5.25, 5.15, 5.00], [5.85, 5.80, 5.70], [5.25, 5.20, 5.10], [5.65, 5.55, 5.45],
    [5.60, 5.35, 5.45], [5.05, 5.00, 4.95], [5.50, 5.50, 5.40], [5.45, 5.55, 5.50],
    [5.55, 5.55, 5.35], [5.45, 5.50, 5.55], [5.50, 5.45, 5.25], [5.65, 5.60, 5.40],
    [5.70, 5.65, 5.55], [6.30, 6.30, 6.25],
])
METHODS = ["round_out", "narrow_angle", "wide_angle"]


def _long(y: np.ndarray, levels: list[str]) -> tuple[list[float], list[str], list[str]]:
    n, k = y.shape
    return (y.ravel().tolist(), [levels[j] for _ in range(n) for j in range(k)],
            [f"s{i}" for i in range(n) for _ in range(k)])


def test_rounding_times_matches_r():
    """R friedman.test(RoundingTimes): χ² = 11.143, df = 2, P = 0.003805; R_j = 53, 47, 32."""
    res = friedman(*_long(ROUNDING, METHODS))
    st = res.statistics
    assert st["chi2"] == pytest.approx(11.142857, abs=1e-5)
    assert st["df"] == 2
    assert res.p_value == pytest.approx(0.003805, abs=1e-6)
    assert [r["rank_sum"] for r in res.parameters["ranks"]] == [53, 47, 32]
    assert st["kendalls_w"] == pytest.approx(st["chi2"] / (22 * 2))
    assert st["tie_correction"] < 1 and st["Q_uncorrected"] < st["chi2"]
    json.dumps(res.to_dict(), allow_nan=False)


def test_agrees_with_scipy_and_iman_davenport_formula():
    """χ² = scipy.friedmanchisquare (с поправкой на совпадения); F = (n−1)χ²/(n(k−1)−χ²)."""
    y = np.random.default_rng(7).integers(1, 6, (12, 4)).astype(float)
    st, _ = friedman_statistic(y)
    ref = stats.friedmanchisquare(*y.T)
    assert st["chi2"] == pytest.approx(ref.statistic)
    assert st["p"] == pytest.approx(ref.pvalue)
    assert st["F_iman_davenport"] == pytest.approx(11 * st["chi2"] / (12 * 3 - st["chi2"]))
    assert st["df2_iman_davenport"] == 33


def test_pairs_are_wilcoxon_with_holm():
    """Пары: P = scipy.wilcoxon по разностям, Холм; сдвиг ХЛ рядом с медианой разностей."""
    res = friedman(*_long(ROUNDING, METHODS))
    rows = res.parameters["comparisons"]
    assert [(r["group1"], r["group2"]) for r in rows] == [
        ("round_out", "narrow_angle"), ("round_out", "wide_angle"),
        ("narrow_angle", "wide_angle")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = [stats.wilcoxon(ROUNDING[:, j] - ROUNDING[:, i]).pvalue
               for i, j in ((0, 1), (0, 2), (1, 2))]
    assert [r["p"] for r in rows] == pytest.approx(raw)
    assert all(r["p_adj"] >= r["p"] for r in rows)
    assert rows[1]["hl_shift"] < 0 and rows[1]["reject"]


def test_witte_rm_sleep_ranks():
    """Witte, Table 17.1 (стр. 323) данными как ранги: у всех 3 субъектов 0 < 24 < 48 ч →
    R_j = 3, 6, 9, W Кендалла = 1; χ² = 6 (без совпадений)."""
    values = [v for row in W.RM_SLEEP for v in row]
    res = friedman(values, ["0", "24", "48"] * 3, [s for s in "ABC" for _ in range(3)])
    assert [r["rank_sum"] for r in res.parameters["ranks"]] == [3, 6, 9]
    assert res.statistics["chi2"] == pytest.approx(6.0)
    assert res.statistics["kendalls_w"] == pytest.approx(1.0)
    assert {w.code for w in res.warnings} == {"friedman_small_n"}


def test_two_conditions_incomplete_and_tied():
    """k = 2 — пояснение про знаковый тест; неполный субъект исключён; все равны — ошибка."""
    two = friedman([1, 2, 3, 5, 2, 4, 1, 1], ["a", "b"] * 4, list("ppqqrrss"))
    assert "friedman_two_conditions" in {w.code for w in two.warnings}
    gap = friedman([1, 2, 3, 2, 3, 4, 5, float("nan"), 6], ["a", "b", "c"] * 3, list("pppqqqrrr"))
    assert gap.statistics["n_excluded_subjects"] == 1
    tied = friedman([2, 2, 3, 3], ["a", "b"] * 2, list("ppqq"))
    assert tied.statistics == {} and "friedman_all_tied" in {w.code for w in tied.warnings}
