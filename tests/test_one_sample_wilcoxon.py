"""Тест Уилкоксона одной выборки (one_sample_wilcoxon) — [предл.: Wilcoxon 1945].

Эталон — пример из справки R wilcox.test (Hollander & Wolfe 1973, стр. 29):
разности x − y депрессии до/после, alternative = "greater": V = 40, P = 0.01953
(точно 10/512). Остальное — равносильность парному wilcoxon и инварианты.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.stats.hypothesis import one_sample_wilcoxon, wilcoxon

X = np.array([1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
Y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])


def test_depression_example_matches_r():
    """R: wilcox.test(x − y, alternative = "greater") → V = 40, P = 0.01953."""
    res = one_sample_wilcoxon(X - Y, alternative="greater")
    assert res.statistics["r_plus"] == 40 and res.statistics["r_minus"] == 5
    assert res.p_value == pytest.approx(10 / 512)
    assert res.statistics["hl_location"] == pytest.approx(0.46)
    assert res.parameters["hl_ci_method"] == "exact"
    json.dumps(res.to_dict(), allow_nan=False)


def test_same_as_paired_wilcoxon_against_constant():
    """Тест x против μ₀ = парный wilcoxon(x, μ₀…); псевдомедиана = μ₀ + сдвиг ХЛ."""
    one = one_sample_wilcoxon(X, popmean=1.5)
    pair = wilcoxon(X, np.full(X.size, 1.5))
    assert one.p_value == pytest.approx(pair.p_value)
    for key in ("r_plus", "r_minus", "T_min", "hl_shift", "hl_ci_lower", "hl_ci_upper"):
        assert one.statistics[key] == pytest.approx(pair.statistics[key])
    st = one.statistics
    assert st["hl_location"] == pytest.approx(1.5 + st["hl_shift"])
    assert st["hl_location_ci_lower"] <= st["hl_location"] <= st["hl_location_ci_upper"]


def test_all_equal_to_reference_is_error():
    res = one_sample_wilcoxon([2.0, 2.0, 2.0], popmean=2.0)
    assert res.p_value is None and "wilcoxon_all_zero" in {w.code for w in res.warnings}
    with pytest.raises(ValueError):
        one_sample_wilcoxon(X, alternative="larger")
