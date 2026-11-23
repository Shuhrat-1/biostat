"""Попарные сравнения Геймса–Хауэлла в welch_anova [предл.: Games & Howell 1976].

Сравнения после ANOVA (B5) опираются на общую MS_within и после ANOVA Уэлча
не подходят. Эталон P, SE и df — pingouin 0.7.0 ``pairwise_gameshowell``
(независимая реализация, сверено вне тестов; у pingouin diff = A − B, у
BioStat — group2 − group1). Для двух групп Геймс–Хауэлл совпадает с
двусторонним t-тестом Уэлча — это проверяется scipy.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.stats.oneway import welch_anova
from tests import witte_data as W
from tests.map_scenarios import SCENARIOS


def _pairs(res) -> dict[tuple[str, str], dict]:
    return {(c["group1"], c["group2"]): c for c in res.parameters["comparisons"]}


def _s08() -> dict[str, list[float]]:
    sc = next(s for s in SCENARIOS if s["id"] == "S08")
    out: dict[str, list[float]] = {}
    for g, y in zip(sc["columns"]["group"], sc["columns"]["y"]):
        out.setdefault(str(g), []).append(float(y))
    return out


def test_sleep_b_matches_pingouin():
    """Witte, Table 16.3 (исход B): эталон pingouin."""
    pairs = _pairs(welch_anova(*W.SLEEP_B, labels=W.SLEEP_LABELS))
    expected = {("0", "24"): (3.0, 1.527525, 3.92, 0.238269),
                ("0", "48"): (6.0, 1.632993, 4.00, 0.045465),
                ("24", "48"): (3.0, 1.527525, 3.92, 0.238269)}
    for key, (diff, se, df, p) in expected.items():
        row = pairs[key]
        assert row["mean_diff"] == pytest.approx(diff)
        assert row["se"] == pytest.approx(se, abs=1e-6)
        assert row["df"] == pytest.approx(df, abs=1e-6)
        assert row["p_adj"] == pytest.approx(p, abs=1e-6)
    assert [r["reject"] for r in pairs.values()] == [False, True, False]


def test_s08_unequal_variances_matches_pingouin():
    """Сценарий карты S08 (Welham, стр. 107; B4.5 → welch_anova)."""
    data = _s08()
    pairs = _pairs(welch_anova(*data.values(), labels=list(data)))
    row = pairs[("a", "b")]
    assert row["df"] == pytest.approx(9.718861, abs=1e-6)
    assert row["se"] == pytest.approx(1.594246, abs=1e-6)
    assert row["p_adj"] == pytest.approx(0.000278, abs=1e-6)
    assert pairs[("a", "c")]["df"] == pytest.approx(18.0)


def test_two_groups_equal_welch_t_test():
    """k = 2: q = √2·|t|, распределение размаха двух — |t| Уэлча."""
    a, b = W.SLEEP_B[0], W.SLEEP_B[2]
    row = welch_anova(a, b).parameters["comparisons"][0]
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert row["t"] == pytest.approx(ref.statistic)
    assert row["p_adj"] == pytest.approx(ref.pvalue, rel=1e-6)


def test_ci_close_to_exact_quantile_and_consistent_with_p():
    """ДИ на приближении Глисона: < 0.2 % от точного квантиля scipy."""
    data = _s08()
    res = welch_anova(*data.values(), labels=list(data))
    for row in res.parameters["comparisons"]:
        exact = stats.studentized_range.ppf(0.95, 3, row["df"]) * row["se"] / np.sqrt(2)
        half = (row["ci_upper"] - row["ci_lower"]) / 2
        assert half == pytest.approx(exact, rel=2e-3)
        assert row["reject"] == (row["ci_lower"] > 0 or row["ci_upper"] < 0)


def test_note_and_counts():
    res = welch_anova(*W.SLEEP_B, labels=W.SLEEP_LABELS)
    assert "games_howell_pairs" in {w.code for w in res.warnings}
    assert res.statistics["n_comparisons"] == 3
    assert res.statistics["n_significant"] == 1


def test_alpha_validated():
    with pytest.raises(ValueError):
        welch_anova([1, 2, 3], [2, 3, 5], alpha=1.5)
