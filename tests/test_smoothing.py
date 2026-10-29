"""Гл. 17.1.2 (стр. 460): сплайны и loess как альтернатива полиномам.

В книге нет численного примера — сверка с независимыми реализациями (patsy bs,
scipy GCV) и свойствами сглаживателей.
"""

from __future__ import annotations

import numpy as np
import patsy
import pytest
from scipy.interpolate import make_smoothing_spline

from core.stats.smoothing import bspline_knots, smooth_curve
from tests.book_ch17_data import OLSEN_P, YIELD


def _sine(n=60, seed=1):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    return x, np.sin(x) + rng.normal(0, 0.3, n)


def test_bspline_matches_patsy():
    """Регрессионный B-сплайн = МНК по базису patsy bs с теми же узлами (Olsen P, Table 17.1)."""
    x, y = np.array(OLSEN_P), np.array(YIELD)
    res = smooth_curve(OLSEN_P, YIELD, "bspline", n_knots=3, degree=3)
    inner = bspline_knots(x, 3, 3)[4:-4]
    design = patsy.dmatrix(f"bs(x, knots={list(inner)}, degree=3, include_intercept=True) - 1",
                           {"x": x})
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = res.parameters["plots"]["residual_vs_fitted"]["fitted"]
    assert fitted == pytest.approx(np.asarray(design) @ beta)
    assert res.statistics["n_basis"] == 7 and res.statistics["df_resid"] == 13


def test_bspline_reproduces_polynomials():
    """Кубический B-сплайн воспроизводит кубическую зависимость точно; степень 1 без узлов — прямая."""
    x = np.linspace(0, 5, 30)
    y = 1 + x - 0.5 * x ** 2 + 0.1 * x ** 3
    res = smooth_curve(x.tolist(), y.tolist(), "bspline", n_knots=2, degree=3)
    assert res.parameters["curve"]["fitted"] == pytest.approx(
        np.polyval([0.1, -0.5, 1, 1], res.parameters["curve"]["x"]), abs=1e-9)
    line = smooth_curve(x.tolist(), (2 + 3 * x).tolist(), "bspline", n_knots=0, degree=1)
    assert line.statistics["df_vs_line"] == 0


def test_bspline_tests_curvature_against_line():
    """F «сплайн против прямой» значим для синуса и CI охватывает подгонку."""
    x, y = _sine()
    res = smooth_curve(x.tolist(), y.tolist(), "bspline", n_knots=5)
    assert res.statistics["p_vs_line"] < 0.001
    c = res.parameters["curve"]
    assert all(lo <= f <= hi for lo, f, hi in zip(c["lower"], c["fitted"], c["upper"]))


def test_smoothing_spline_gcv_matches_scipy():
    """Наш минимум GCV даёт ту же кривую, что make_smoothing_spline(lam=None) в scipy;
    при заданном λ критерий совпадает со scipy точно."""
    x, y = _sine()
    res = smooth_curve(x.tolist(), y.tolist(), "smoothing_spline")
    grid = np.array(res.parameters["curve"]["x"])
    assert res.parameters["curve"]["fitted"] == pytest.approx(make_smoothing_spline(x, y)(grid),
                                                              abs=1e-4)
    assert 2 < res.statistics["edf"] < 60
    fixed = smooth_curve(x.tolist(), y.tolist(), "smoothing_spline", lam=0.5)
    assert fixed.parameters["curve"]["fitted"] == pytest.approx(
        make_smoothing_spline(x, y, lam=0.5)(grid), abs=1e-8)


def test_smoothing_spline_limits():
    """λ → ∞ — прямая МНК (edf → 2); λ → 0 — интерполяция (edf → n)."""
    x, y = _sine(n=20)
    big = smooth_curve(x.tolist(), y.tolist(), "smoothing_spline", lam=1e9)
    slope, icept = np.polyfit(x, y, 1)
    grid = np.array(big.parameters["curve"]["x"])
    assert big.parameters["curve"]["fitted"] == pytest.approx(icept + slope * grid, abs=5e-4)
    assert big.statistics["edf"] == pytest.approx(2, abs=1e-3)
    assert any(w.code == "smooth_lambda_capped" for w in big.warnings)
    small = smooth_curve(x.tolist(), y.tolist(), "smoothing_spline", lam=1e-10)
    assert small.statistics["edf"] == pytest.approx(20, abs=0.01)


def test_smoothing_spline_with_replicated_x():
    """Повторы x: сглаживаются средние с весами n_x; edf ≤ числа различных x."""
    x = np.repeat(np.arange(8.0), 3)
    rng = np.random.default_rng(2)
    y = np.sqrt(x) + rng.normal(0, 0.2, x.size)
    res = smooth_curve(x.tolist(), y.tolist(), "smoothing_spline")
    assert res.statistics["edf"] <= 8
    assert res.statistics["df_resid"] == pytest.approx(24 - res.statistics["edf"])


def test_loess_linear_and_robust():
    """Loess воспроизводит прямую точно; робастные итерации ослабляют влияние выброса."""
    x = np.linspace(0, 10, 40)
    line = smooth_curve(x.tolist(), (1 + 2 * x).tolist(), "loess", frac=0.5)
    assert line.parameters["curve"]["fitted"] == pytest.approx(
        1 + 2 * np.array(line.parameters["curve"]["x"]))
    y = 1 + 2 * x + np.random.default_rng(3).normal(0, 0.3, x.size)
    y[20] += 30
    plain = smooth_curve(x.tolist(), y.tolist(), "loess", frac=0.3, iterations=0, x_predict="5.2")
    robust = smooth_curve(x.tolist(), y.tolist(), "loess", frac=0.3, iterations=3, x_predict="5.2")
    truth = 1 + 2 * 5.2
    assert abs(robust.parameters["predictions"][0]["fitted"] - truth) < abs(
        plain.parameters["predictions"][0]["fitted"] - truth)
    assert any(w.code == "loess_descriptive" for w in robust.warnings)
