"""Кривые в RCBD и по группам (упражнения 17.1, 17.4, 17.6; Table 15.13; Example 17.1C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import optimize

from core.stats.ancova import blocked_regression
from core.stats.curve_designs import curve_in_blocks, nonlinear_in_blocks
from tests.book_ch15_data import FORAGE
from tests.book_ch17_data import OLSEN_P, YIELD

_N = np.repeat([0, 50, 100, 150, 200], 4).astype(float)
_B = np.tile(["1", "2", "3", "4"], 5)
_BEFF = {"1": 0.0, "2": 0.5, "3": -0.3, "4": 0.2}


def _rcbd(seed=3, quad=False):
    rng = np.random.default_rng(seed)
    block = np.array([_BEFF[b] for b in _B])
    mean = 4 + 0.03 * _N - 0.0001 * _N ** 2 if quad else 8 - 5 * np.exp(-0.015 * _N)
    return (mean + block + rng.normal(0, 0.2, _N.size)).tolist()


def _rows(seq):
    return {r["term"]: r for r in seq}


def test_straight_line_in_blocks_table_15_13():
    """Книга: Table 15.13 (стр. 408): degree 1 = blocked_regression — SS Block 2.8385, N 5.9283,
    FacN 0.2150, Residual 1.9359."""
    res = curve_in_blocks([r[3] for r in FORAGE], [r[2] for r in FORAGE],
                          block=[str(r[0]) for r in FORAGE], degree=1,
                          column_names={"x": "N", "block": "Block"})
    rows = _rows(res.parameters["sequential"])
    assert [rows[t]["ss"] for t in ("+ Block", "+ N", "+ FacN", "Residual")] == pytest.approx(
        [2.8385, 5.9283, 0.2150, 1.9359], abs=1e-4)
    ref = blocked_regression([r[3] for r in FORAGE], [r[2] for r in FORAGE],
                             [str(r[0]) for r in FORAGE])
    assert res.parameters["parameter_tests"][-1]["estimate"] == pytest.approx(
        ref.parameters["line"]["slope"])


def test_quadratic_in_blocks_matches_ols():
    """Квадратичная кривая в RCBD (упр. 17.4): SS типа I и ДИ среднего блока = statsmodels OLS."""
    y = _rcbd(quad=True)
    res = curve_in_blocks(y, _N.tolist(), block=_B.tolist(), degree=2, x_predict="25")
    df = pd.DataFrame({"y": y, "N": _N, "B": _B, "F": [f"{v:g}" for v in _N]})
    df["c"] = df["N"] - df["N"].mean()
    formulas = ["y ~ 1", "y ~ C(B)", "y ~ C(B) + c", "y ~ C(B) + c + I(c**2)",
                "y ~ C(B) + C(F)"]
    rss = [smf.ols(f, df).fit().ssr for f in formulas]
    seq = [r["ss"] for r in res.parameters["sequential"][:4]]
    assert seq == pytest.approx(np.diff(rss)[:4] * -1, abs=1e-8)
    curve = smf.ols("y ~ C(B) + c + I(c**2)", df).fit()
    c25 = 25 - df["N"].mean()
    exog = pd.DataFrame({"B": ["1", "2", "3", "4"], "c": [c25] * 4})
    pred = curve.get_prediction(exog).summary_frame()
    p = res.parameters["predictions"][0]
    assert p["fitted"] == pytest.approx(pred["mean"].mean())
    weights = np.array([[1, 0, 0, 0, c25, c25 ** 2], [1, 1, 0, 0, c25, c25 ** 2],
                        [1, 0, 1, 0, c25, c25 ** 2], [1, 0, 0, 1, c25, c25 ** 2]]).mean(axis=0)
    se = float(np.sqrt(weights @ curve.cov_params().to_numpy() @ weights))
    assert p["se_mean"] == pytest.approx(se)


def test_transformed_dose_in_blocks():
    """Преобразование дозы (упр. 17.1a): ln(x + c) с блоками и неадекватностью."""
    rng = np.random.default_rng(2)
    dose = np.repeat([0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1], 3)
    blocks = np.tile(["1", "2", "3"], 7)
    y = 50 - 8 * np.log(dose + 0.01) * -1 + rng.normal(0, 1, dose.size)
    res = curve_in_blocks(y.tolist(), dose.tolist(), block=blocks.tolist(), curve="transform",
                          x_transform_kind="ln", x_offset=0.01)
    assert res.parameters["parameter_tests"][-1]["estimate"] == pytest.approx(8, abs=1.0)
    assert res.statistics["p_lack_of_fit"] > 0.01
    bad = curve_in_blocks(y.tolist(), dose.tolist(), block=blocks.tolist(), curve="transform",
                          x_transform_kind="ln")
    assert any(w.code == "x_transform_domain" for w in bad.warnings)


def test_groups_choose_parallel_curves():
    """Группы с одинаковой формой и сдвигом: значима Group, не значимы степени.Group (гл. 15)."""
    rng = np.random.default_rng(1)
    x = np.tile(np.repeat([0, 1, 2, 3, 4], 3), 2).astype(float)
    g = np.repeat(["a", "b"], 15)
    y = 2 + 1.5 * x - 0.2 * x ** 2 + 2 * (g == "b") + rng.normal(0, 0.3, x.size)
    res = curve_in_blocks(y.tolist(), x.tolist(), group=g.tolist(), degree=2)
    assert res.parameters["model_terms"] == ["x", "x^2", "Group"]
    assert len(res.parameters["curves"]) == 2


def test_nonlinear_without_blocks_example_17_1c():
    """Книга: Example 17.1C (стр. 474–475): без блоков = 4.405 − 4.412·exp(−0.359x), ResMS 0.0991."""
    res = nonlinear_in_blocks(YIELD, [str(v) for v in OLSEN_P])
    assert [p["estimate"] for p in res.parameters["parameter_tests"]] == pytest.approx(
        [4.405, -4.412, 0.359], abs=1e-3)
    assert res.statistics["ms_resid"] == pytest.approx(0.0991, abs=1e-4)


def test_nonlinear_in_blocks_matches_curve_fit():
    """Экспонента в RCBD (упр. 17.6): оценки и SE = scipy curve_fit с явными индикаторами блоков."""
    y = np.array(_rcbd())
    res = nonlinear_in_blocks(y.tolist(), _N.tolist(), block=_B.tolist())
    ind = np.column_stack([(_B == b).astype(float) for b in "234"])

    def f(_, a, b2, b3, b4, beta, gamma):
        return a + ind @ np.array([b2, b3, b4]) + beta * np.exp(-gamma * _N)

    popt, pcov = optimize.curve_fit(f, _N, y, p0=[8, 0, 0, 0, -5, 0.01])
    est = [p["estimate"] for p in res.parameters["parameter_tests"]]
    assert est == pytest.approx(popt, rel=1e-5)
    assert [p["se"] for p in res.parameters["parameter_tests"]] == pytest.approx(
        np.sqrt(np.diag(pcov)), rel=1e-3)
    lof = res.parameters["lack_of_fit"]
    assert (lof["df"], lof["df_resid"]) == (2, 12)


def test_nonlinear_group_models_and_selection():
    """Группы: общая → отдельные α → α, β → все; RSS не растёт, выбрана модель «alpha»."""
    rng = np.random.default_rng(3)
    n2, b2 = np.tile(_N, 2), np.tile(_B, 2)
    g = np.repeat(["A", "B"], 20)
    y = 8 + (g == "B") - 5 * np.exp(-0.015 * n2) + np.array([_BEFF[b] for b in b2]) \
        + rng.normal(0, 0.2, n2.size)
    res = nonlinear_in_blocks(y.tolist(), n2.tolist(), block=b2.tolist(), group=g.tolist())
    rss = [r["rss"] for r in res.parameters["group_models"]]
    assert rss == sorted(rss, reverse=True)
    assert res.parameters["selected_model"] == "alpha"
    forced = nonlinear_in_blocks(y.tolist(), n2.tolist(), block=b2.tolist(), group=g.tolist(),
                                 group_model="all")
    labels = [p["term"] for p in forced.parameters["parameter_tests"]]
    assert {"gamma A", "gamma B", "beta A", "beta B"} <= set(labels)
