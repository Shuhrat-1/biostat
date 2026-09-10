"""Тесты линейной регрессии: сверка с scipy.linregress и инвариантами."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from core.stats.regression import predict, simple_linear_regression


def _est(result, name):
    return next(e for e in result.estimates if e.name == name)


def test_perfect_line():
    """Точная прямая y = 2x + 1: наклон 2, свободный член 1, R²=1."""
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 3.0, 5.0, 7.0, 9.0]
    r = simple_linear_regression(x, y)
    assert _est(r, "slope").value == pytest.approx(2.0)
    assert _est(r, "intercept").value == pytest.approx(1.0)
    assert r.statistics["r_squared"] == pytest.approx(1.0)


def test_matches_scipy_linregress():
    """Коэффициенты и R² совпадают с scipy.linregress."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 30)
    y = 2.5 + 1.3 * x + rng.normal(0, 1.5, 30)
    r = simple_linear_regression(x, y)
    ref = stats.linregress(x, y)
    assert _est(r, "slope").value == pytest.approx(ref.slope)
    assert _est(r, "intercept").value == pytest.approx(ref.intercept)
    assert r.statistics["r_squared"] == pytest.approx(ref.rvalue**2)
    assert _est(r, "slope").std_error == pytest.approx(ref.stderr)


def test_slope_pvalue_matches_scipy():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 25)
    y = 1.0 + 0.8 * x + rng.normal(0, 0.5, 25)
    r = simple_linear_regression(x, y)
    ref = stats.linregress(x, y)
    assert r.statistics["slope_p"] == pytest.approx(ref.pvalue)


def test_ci_brackets_slope():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 40)
    y = 3.0 + 2.0 * x + rng.normal(0, 2, 40)
    slope = _est(simple_linear_regression(x, y), "slope")
    assert slope.ci_lower < slope.value < slope.ci_upper


def test_residuals_saved_for_diagnostics():
    """Остатки, предсказания и стандартизованные остатки сохранены."""
    x = list(range(10))
    y = [2 * v + 1 for v in x]
    r = simple_linear_regression(x, y)
    p = r.parameters
    assert len(p["residuals"]) == 10
    assert len(p["fitted"]) == 10
    assert len(p["std_residuals"]) == 10
    # Сумма остатков OLS ≈ 0.
    assert sum(p["residuals"]) == pytest.approx(0.0, abs=1e-9)


def test_dropped_nan_pairs():
    x = [1.0, 2.0, np.nan, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, np.nan, 10.0]
    r = simple_linear_regression(x, y)
    assert r.n == 3  # пары с индексами 2 и 3 удалены
    assert any(w.code == "missing_dropped" for w in r.warnings)


def test_length_mismatch():
    with pytest.raises(ValueError):
        simple_linear_regression([1.0, 2.0], [1.0, 2.0, 3.0])


def test_too_few_points():
    r = simple_linear_regression([1.0, 2.0], [2.0, 4.0])
    assert any(w.code == "n_too_small" for w in r.warnings)


def test_no_x_variation():
    r = simple_linear_regression([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    assert any(w.code == "no_x_variation" for w in r.warnings)


def test_invalid_ci_level():
    with pytest.raises(ValueError):
        simple_linear_regression([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ci_level=0)


def test_predict_point_on_line():
    """Предсказание на точной прямой попадает в точку."""
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 3.0, 5.0, 7.0, 9.0]
    r = predict(x, y, [5.0])
    pr = r.parameters["predictions"][0]
    assert pr["predicted"] == pytest.approx(11.0)


def test_predict_obs_ci_wider_than_mean_ci():
    """Интервал предсказания наблюдения шире интервала для среднего."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 30)
    y = 2.0 + 1.5 * x + rng.normal(0, 2, 30)
    pr = predict(x, y, [5.0]).parameters["predictions"][0]
    mean_w = pr["mean_ci_upper"] - pr["mean_ci_lower"]
    obs_w = pr["obs_ci_upper"] - pr["obs_ci_lower"]
    assert obs_w > mean_w


def test_json_serializable():
    import json

    x = list(range(8))
    y = [1.5 * v + 0.3 for v in x]
    r = simple_linear_regression(x, y)
    parsed = json.loads(r.to_json())
    assert parsed["method"] == "simple_linear_regression"
    assert "r_squared" in parsed["statistics"]
