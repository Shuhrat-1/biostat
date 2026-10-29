"""Гл. 17: поверхности от 3–4 переменных и тригонометрическая регрессия с оцениваемым периодом.

Численных примеров в книге нет (стр. 462, 470–472) — сверка с независимыми
реализациями (statsmodels OLS, scipy curve_fit) и восстановление параметров.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import optimize
from statsmodels.stats.anova import anova_lm

from core.stats.curves import trigonometric_regression
from core.stats.response_surface import is_subterm, response_surface, surface_terms
from tests.book_ch17_data import TEMPERATURE


def _three_factor(seed=4):
    rng = np.random.default_rng(seed)
    grid = np.array(list(product([0, 1, 2], [10, 20, 30], [1, 2, 3])) * 2, float)
    a, b, c = grid.T
    y = 5 + 2 * a - 0.08 * b + 1.5 * c - 0.6 * a ** 2 + 0.05 * a * b + rng.normal(0, 0.5, len(a))
    return {"A": a.tolist(), "B": b.tolist(), "C": c.tolist(), "y": y.tolist()}


def test_surface_terms_generalize_two_variable_order():
    """Для двух переменных порядок членов прежний (Table 17.10); для трёх — 9 членов 2-го порядка."""
    assert surface_terms("cubic", 2)[:5] == [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    quad = surface_terms("quadratic", 3)
    assert len(quad) == 9 and quad[:3] == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert surface_terms("interaction", 3)[3:] == [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    assert is_subterm((1, 0, 0), (2, 1, 0)) and not is_subterm((0, 0, 1), (2, 1, 0))


def test_three_variable_surface_matches_ols():
    """Последовательные SS и оценки совпадают с statsmodels OLS (тип I); остаток — чистая ошибка."""
    data = _three_factor()
    res = response_surface(data["y"], data, "y", "A, B, C", "quadratic", centre="mean",
                           selection="none")
    df = pd.DataFrame(data)
    for k in "ABC":
        df["c" + k] = df[k] - df[k].mean()
    formula = ("y ~ cA + cB + cC + I(cA**2) + I(cA*cB) + I(cA*cC) + I(cB**2) + I(cB*cC) "
               "+ I(cC**2)")
    ols = smf.ols(formula, df).fit()
    seq = [r["ss"] for r in res.parameters["sequential"][:9]]
    assert seq == pytest.approx(anova_lm(ols, typ=1)["sum_sq"].to_numpy()[:9])
    assert [p["estimate"] for p in res.parameters["parameter_tests"]] == pytest.approx(
        ols.params.to_numpy())
    cells = df.groupby(["A", "B", "C"])["y"]
    pe = float(((df["y"] - cells.transform("mean")) ** 2).sum())
    assert (res.parameters["error"]["ss"], res.parameters["error"]["df"]) == (pytest.approx(pe), 27)


def test_three_variable_backward_keeps_true_terms():
    """Обратный отбор с маргинальностью оставляет истинные члены A², A.B и все главные."""
    data = _three_factor()
    res = response_surface(data["y"], data, "y", "A, B, C", "quadratic", selection="backward")
    kept = set(res.parameters["selected_terms"])
    assert {"cA^2", "cA.cB", "cA", "cB", "cC"} <= kept
    assert "cC^2" not in kept
    assert res.parameters["surface"]["others_at_mean"] == ["x3"]


def test_free_period_recovers_simulated_cycle():
    """ω̂ и его SE совпадают с полной нелинейной МНК (scipy curve_fit) — истинный ω = 10.5."""
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 0.5)
    y = 5 + 3 * np.sin(2 * np.pi * t / 10.5 - 1) + rng.normal(0, 0.5, t.size)
    res = trigonometric_regression(t.tolist(), y.tolist(), 12, estimate_period="yes")
    est = res.parameters["period_estimate"]

    def f(v, a, b1, b2, w):
        return a + b1 * np.sin(2 * np.pi * v / w) + b2 * np.cos(2 * np.pi * v / w)

    popt, pcov = optimize.curve_fit(f, t, y, p0=[5, 1, 1, est["estimate"]])
    assert est["estimate"] == pytest.approx(popt[3], rel=1e-6)
    assert est["se"] == pytest.approx(np.sqrt(pcov[3, 3]), rel=1e-4)
    assert est["ci_lower"] < 10.5 < est["ci_upper"]
    se_beta = [p["se"] for p in res.parameters["parameter_tests"]]
    assert se_beta == pytest.approx(np.sqrt(np.diag(pcov))[:3], rel=1e-4)


def test_free_period_on_temperature_data():
    """Table 17.5 (стр. 461): со свободным ω подгонка не хуже, чем при ω = 12; df на 1 меньше."""
    fixed = trigonometric_regression(list(range(1, 13)), TEMPERATURE, 12)
    free = trigonometric_regression(list(range(1, 13)), TEMPERATURE, 12, estimate_period="yes")
    est = free.parameters["period_estimate"]
    assert est["df"] == fixed.statistics["df_resid"] - 1
    assert free.statistics["ms_resid"] * est["df"] <= fixed.statistics["ms_resid"] * 9 + 1e-9
    assert 10 < est["estimate"] < 14
