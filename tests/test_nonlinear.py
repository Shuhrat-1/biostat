"""Глава 17.3: нелинейные кривые и сравнение моделей (Example 17.1C, Table 17.12)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import optimize

from core.stats.nonlinear import (
    CURVES,
    curve_comparison,
    curve_value,
    nonlinear_regression,
)
from tests.book_ch17_data import OLSEN_P, YIELD


def _est(res):
    return [e.value for e in res.estimates]


def test_exponential_example_17_1c():
    """Книга: Example 17.1C (стр. 474–475): μ̂ = 4.405 − 4.412·exp(−0.359x), R²adj = 0.831."""
    res = nonlinear_regression(OLSEN_P, YIELD, "exponential")
    assert _est(res) == pytest.approx([4.405, -4.412, 0.359], abs=1e-3)
    assert res.statistics["adj_r_squared"] == pytest.approx(0.831, abs=5e-4)
    derived = {r["quantity"]: r for r in res.parameters["derived"]}
    assert derived["asymptote"]["estimate"] == pytest.approx(4.405, abs=1e-3)


def test_inverse_linear_example_17_1c():
    """Книга: Example 17.1C (стр. 475): 4.962 − 9.073/(1 + 1.208x), R²adj = 0.835.

    Параметры плохо определены (SE β ≈ 6): наш минимум 4.961, −9.080, 1.209 даёт
    RSS меньше, чем округлённые значения книги, — расхождение в допуске сходимости.
    """
    res = nonlinear_regression(OLSEN_P, YIELD, "inverse_linear")
    assert _est(res) == pytest.approx([4.962, -9.073, 1.208], abs=1e-2)
    assert res.statistics["adj_r_squared"] == pytest.approx(0.835, abs=5e-4)
    book = np.sum((curve_value(CURVES["inverse_linear"], np.array(OLSEN_P),
                               np.array([4.962, -9.073, 1.208])) - YIELD) ** 2)
    assert res.statistics["ms_resid"] * 17 <= book


@pytest.mark.parametrize(("model", "ms", "adj", "aic", "sbc"), [
    ("exponential", 0.0991, 83.1, 16.4, 19.4), ("inverse_linear", 0.0965, 83.5, 15.9, 18.9)])
def test_nonlinear_summary_table_17_12(model, ms, adj, aic, sbc):
    """Книга: Table 17.12 (стр. 476): ResMS, R²adj×100, AIC, SBC нелинейных моделей."""
    st = nonlinear_regression(OLSEN_P, YIELD, model).statistics
    assert st["ms_resid"] == pytest.approx(ms, abs=1e-4)
    assert 100 * st["adj_r_squared"] == pytest.approx(adj, abs=0.05)
    assert (st["aic"], st["sbc"]) == pytest.approx((aic, sbc), abs=0.05)


def test_curve_comparison_table_17_12():
    """Книга: Table 17.12 (стр. 476): шесть моделей; R²adj и AIC — у полинома 8, SBC — у нелинейных."""
    res = curve_comparison(OLSEN_P, YIELD,
                           "log10, poly2, poly3, poly8, exponential, inverse_linear")
    rows = {r["model"]: r for r in res.parameters["models_table"]}
    assert [rows[m]["p"] for m in rows] == [2, 3, 4, 9, 3, 3]
    expected = {"log10": (0.1191, 19.2, 21.2), "poly2": (0.1126, 19.0, 22.0),
                "poly3": (0.1114, 19.6, 23.5), "poly8": (0.0799, 15.4, 24.4),
                "exponential": (0.0991, 16.4, 19.4), "inverse_linear": (0.0965, 15.9, 18.9)}
    for model, (ms, aic, sbc) in expected.items():
        assert (rows[model]["ms_resid"], rows[model]["aic"], rows[model]["sbc"]) == pytest.approx(
            (ms, aic, sbc), abs=0.05)
    assert res.parameters["best"] == {"adj_r_squared": "poly8", "aic": "poly8",
                                      "sbc": "inverse_linear"}


def test_curve_comparison_reports_failures():
    """Недопустимые модели (log при x ≤ 0, порядок ≥ числа x) помечаются, остальные считаются."""
    x = [0, 1, 2, 3, 4, 5]
    y = [1.0, 2.1, 2.9, 4.2, 4.8, 6.1]
    res = curve_comparison(x, y, "linear, log10, poly6")
    fitted = {r["model"]: r["fitted"] for r in res.parameters["models_table"]}
    assert fitted == {"linear": True, "log10": False, "poly6": False}
    assert any(w.code == "curve_models_failed" for w in res.warnings)
    with pytest.raises(ValueError):
        curve_comparison(x, y, "spline")


def _s_curve(model, noise=0.0, seed=1):
    """Fig. 17.13c, d (стр. 473): α = 5, β = 10, γ = 2, δ = 2 на x ∈ [0, 4]."""
    x = np.linspace(0, 4, 41)
    y = curve_value(CURVES[model], x, np.array([5.0, 10.0, 2.0, 2.0]))
    return x, y + noise * np.random.default_rng(seed).standard_normal(x.size)


@pytest.mark.parametrize("model", ["logistic", "gompertz"])
def test_s_curves_recover_parameters(model):
    """Почти без шума S-кривые восстанавливают параметры Fig. 17.13 (стр. 473)."""
    x, y = _s_curve(model, noise=1e-6)
    res = nonlinear_regression(x.tolist(), y.tolist(), model)
    assert _est(res) == pytest.approx([5.0, 10.0, 2.0, 2.0], abs=1e-4)


def test_logistic_se_matches_scipy_curve_fit():
    """Приближённые SE s²(JᵀJ)⁻¹ совпадают с эталоном scipy.optimize.curve_fit."""
    x, y = _s_curve("logistic", noise=0.4)
    res = nonlinear_regression(x.tolist(), y.tolist(), "logistic")

    def f(v, a, b, g, d):
        return a + b / (1 + np.exp(-g * (v - d)))

    popt, pcov = optimize.curve_fit(f, x, y, p0=[5, 10, 2, 2])
    assert _est(res) == pytest.approx(popt, rel=1e-5)
    assert [e.std_error for e in res.estimates] == pytest.approx(np.sqrt(np.diag(pcov)), rel=1e-4)
    derived = {r["quantity"]: r["estimate"] for r in res.parameters["derived"]}
    assert derived["value_at_delta"] == pytest.approx(popt[0] + popt[1] / 2)


def test_degenerate_s_curve_warns():
    """Olsen P без нижней асимптоты: δ уходит за пределы x — предупреждение (стр. 475–476)."""
    res = nonlinear_regression(OLSEN_P, YIELD, "logistic")
    assert any(w.code == "nonlinear_degenerate" for w in res.warnings)


def test_prediction_band_and_start_values():
    """Предсказание с ДИ (дельта-метод), заданные начальные значения, сериализация."""
    res = nonlinear_regression(OLSEN_P, YIELD, "exponential",
                               start="alpha=4; beta=-4; gamma=0.3", x_predict="7; 20")
    pred = res.parameters["predictions"]
    assert pred[0]["mean_lower"] < pred[0]["fitted"] < pred[0]["mean_upper"]
    assert any(w.code == "curve_extrapolation" for w in res.warnings)
    assert len(res.parameters["curve"]["x"]) == 101
    json.dumps(res.to_dict())
    with pytest.raises(ValueError):
        nonlinear_regression(OLSEN_P, YIELD, "exponential", start="alpha=4")


def test_power_curve_requires_positive_x():
    """α + β·x^θ (стр. 472) определена при x > 0."""
    res = nonlinear_regression([0, 1, 2, 3, 4], [1, 2, 3, 4, 6], "power")
    assert any(w.code == "x_transform_domain" for w in res.warnings)
