"""Тесты ANOVA: сверка со scipy/statsmodels и структурными инвариантами."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.stats.anova import (
    kruskal_wallis,
    one_way_anova,
    tukey_hsd,
)

# Классический сбалансированный пример.
G1 = [24.0, 26.0, 25.0, 23.0, 27.0]
G2 = [30.0, 32.0, 31.0, 29.0, 33.0]
G3 = [28.0, 27.0, 29.0, 26.0, 30.0]


def test_anova_f_matches_scipy():
    r = one_way_anova(G1, G2, G3)
    ref = stats.f_oneway(G1, G2, G3)
    assert r.statistics["F"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_anova_ss_decomposition():
    """SS_between + SS_within = SS_total — фундаментальное тождество."""
    r = one_way_anova(G1, G2, G3)
    s = r.statistics
    assert s["ss_between"] + s["ss_within"] == pytest.approx(s["ss_total"])


def test_anova_df():
    r = one_way_anova(G1, G2, G3)
    assert r.statistics["df_between"] == 2  # k-1
    assert r.statistics["df_within"] == 12  # N-k = 15-3


def test_anova_ms_and_f_consistency():
    """F = MS_between / MS_within — сверка внутренней согласованности."""
    r = one_way_anova(G1, G2, G3)
    s = r.statistics
    assert s["F"] == pytest.approx(s["ms_between"] / s["ms_within"])


def test_anova_eta_squared_range():
    r = one_way_anova(G1, G2, G3)
    assert 0.0 <= r.statistics["eta_squared"] <= 1.0


def test_anova_group_means_recorded():
    r = one_way_anova(G1, G2, G3, labels=["A", "B", "C"])
    names = {e.name: e.value for e in r.estimates}
    assert names["mean_A"] == pytest.approx(np.mean(G1))
    assert names["mean_B"] == pytest.approx(np.mean(G2))


def test_anova_too_few_groups():
    r = one_way_anova(G1)
    assert any(w.code == "too_few_groups" for w in r.warnings)


def test_anova_label_mismatch():
    with pytest.raises(ValueError):
        one_way_anova(G1, G2, labels=["only_one"])


def test_anova_variance_heterogeneity_warns():
    rng = np.random.default_rng(11)
    a = rng.normal(10, 1, 30)
    b = rng.normal(10, 9, 30)
    c = rng.normal(10, 1, 30)
    r = one_way_anova(a, b, c)
    assert any(w.code == "variance_heterogeneity" for w in r.warnings)


def test_tukey_matches_statsmodels_structure():
    r = tukey_hsd(G1, G2, G3, labels=["A", "B", "C"])
    comps = r.parameters["comparisons"]
    assert len(comps) == 3  # C(3,2)
    # Конвенция statsmodels: mean_diff = mean(group2) - mean(group1)
    ab = next(c for c in comps if c["group1"] == "A" and c["group2"] == "B")
    assert ab["mean_diff"] == pytest.approx(np.mean(G2) - np.mean(G1))
    assert ab["reject"] is True  # группы явно различны


def test_tukey_ci_brackets_diff():
    """Доверительный интервал содержит точечную разность."""
    r = tukey_hsd(G1, G2, G3)
    for c in r.parameters["comparisons"]:
        assert c["ci_lower"] <= c["mean_diff"] <= c["ci_upper"]


def test_tukey_counts():
    r = tukey_hsd(G1, G2, G3)
    assert r.statistics["n_comparisons"] == 3
    assert r.statistics["n_significant"] == r.statistics["n_comparisons"]


def test_kruskal_matches_scipy():
    r = kruskal_wallis(G1, G2, G3)
    ref = stats.kruskal(G1, G2, G3)
    assert r.statistics["H"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.statistics["df"] == 2


def test_anova_json_serializable():
    import json

    r = one_way_anova(G1, G2, G3, labels=["A", "B", "C"])
    parsed = json.loads(r.to_json())
    assert parsed["method"] == "one_way_anova"
    assert "F" in parsed["statistics"]
