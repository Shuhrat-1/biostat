"""Глава 12: ПЛР, регрессия через начало, неадекватность, калибровка."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import optimize, stats

from core.stats.linear_regression import (
    _summary,
    _xy_frame,
    inverse_prediction,
    parse_numbers,
    regression_details,
)
from tests.book_ch12_data import AIRTEMP, TRANSECT, TRITICUM


def _seeds():
    a = np.array(TRITICUM)
    return a[:, 2], a[:, 1]  # длина, вес


def test_slr_anova_table_12_3():
    r = regression_details(*_seeds())
    model, resid, total = r.parameters["anova_table"]
    assert model["ss"] == pytest.approx(5683.1753, abs=1e-4)
    assert resid["ss"] == pytest.approx(1611.2338, abs=1e-4)
    assert total["ss"] == pytest.approx(7294.4090, abs=1e-4)
    assert (model["df"], resid["df"], total["df"]) == (1, 188, 189)
    assert resid["ms"] == pytest.approx(8.5704, abs=1e-4)
    assert model["F"] == pytest.approx(663.117, abs=1e-3)


def test_parameters_table_12_4_and_cis():
    r = regression_details(*_seeds())
    icept, slope = r.parameters["parameter_tests"]
    assert icept["estimate"] == pytest.approx(-27.931, abs=1e-3)
    assert icept["se"] == pytest.approx(2.2078, abs=1e-4)
    assert icept["t"] == pytest.approx(-12.651, abs=1e-3)
    assert slope["estimate"] == pytest.approx(17.173, abs=1e-3)
    assert slope["se"] == pytest.approx(0.6669, abs=1e-4)
    assert slope["t"] == pytest.approx(25.751, abs=1e-3)
    assert slope["t"] ** 2 == pytest.approx(r.statistics["F_model"])
    assert (icept["ci_lower"], icept["ci_upper"]) == pytest.approx((-32.286, -23.576), abs=1e-3)
    assert (slope["ci_lower"], slope["ci_upper"]) == pytest.approx((15.858, 18.489), abs=1e-3)


def test_predictions_example_12_1e():
    x, y = _seeds()
    r = regression_details(x, y, x_predict="2.45; 4,13")
    low, high = r.parameters["predictions"]
    assert low["fitted"] == pytest.approx(14.144, abs=1e-3)
    assert low["se_mean"] == pytest.approx(0.6023, abs=1e-4)
    # Книга: (12.955, 15.335); 14.144 + 1.973·0.6023 = 15.332 — описка в книге.
    assert (low["mean_lower"], low["mean_upper"]) == pytest.approx((12.955, 15.332), abs=1e-3)
    assert high["se_mean"] == pytest.approx(0.5959, abs=1e-4)
    assert (high["mean_lower"], high["mean_upper"]) == pytest.approx((41.819, 44.170), abs=1e-3)
    assert low["se_new"] == pytest.approx(np.sqrt(0.6023 ** 2 + 8.5704), abs=1e-3)
    assert not any(w.code == "extrapolation" for w in r.warnings)


def test_prediction_at_mean_has_smallest_se():
    x, y = _seeds()
    r = regression_details(x, y, x_predict=str(x.mean()))
    assert r.parameters["predictions"][0]["se_mean"] == pytest.approx(0.2124, abs=1e-4)


def test_goodness_of_fit_example_12_1f():
    r = regression_details(*_seeds())
    assert r.statistics["r_squared"] == pytest.approx(0.779, abs=5e-4)
    assert r.statistics["adj_r_squared"] == pytest.approx(0.778, abs=5e-4)


def test_lack_of_fit_table_12_7():
    x, counts = np.array(TRANSECT, float).T
    r = regression_details(x, counts, transform_kind="log10", lack_of_fit="yes")
    rows = {row["term"]: row for row in r.parameters["anova_table"]}
    assert rows["Model"]["ss"] == pytest.approx(0.3562, abs=1e-4)
    assert rows["Lack of fit"]["ss"] == pytest.approx(0.2789, abs=1e-4)
    assert rows["Pure error"]["ss"] == pytest.approx(0.2905, abs=1e-4)
    assert rows["Total"]["ss"] == pytest.approx(0.9256, abs=1e-4)
    assert (rows["Lack of fit"]["df"], rows["Pure error"]["df"]) == (4, 18)
    assert rows["Model"]["F"] == pytest.approx(22.073, abs=1e-3)
    assert rows["Lack of fit"]["F"] == pytest.approx(4.321, abs=1e-3)
    assert rows["Lack of fit"]["p"] == pytest.approx(0.013, abs=5e-4)
    assert r.statistics["adj_r_squared_groups"] == pytest.approx(0.599, abs=5e-4)
    assert any(w.code == "lack_of_fit" for w in r.warnings)


def test_transect_slr_example_12_2a():
    x, counts = np.array(TRANSECT, float).T
    r = regression_details(x, counts, transform_kind="log10")
    assert r.statistics["F_model"] == pytest.approx(13.764, abs=1e-3)
    assert r.statistics["adj_r_squared"] == pytest.approx(0.357, abs=5e-4)
    assert r.statistics["ms_resid"] == pytest.approx(0.0259, abs=1e-4)


def test_lack_of_fit_needs_replication():
    r = regression_details([1, 2, 3, 4], [1.0, 2.1, 2.9, 4.2], lack_of_fit="yes")
    assert any(w.code == "lack_of_fit_unavailable" for w in r.warnings)
    assert len(r.parameters["anova_table"]) == 3


def test_through_origin_table_12_10():
    a = np.array(AIRTEMP)
    r = regression_details(a[:, 1], a[:, 2], through_origin="yes", slope_null=1.0)
    model, resid, total = r.parameters["anova_table"]
    assert (model["ss"], resid["ss"], total["ss"]) == pytest.approx(
        (17698.267, 63.953, 17762.220), abs=1e-3)
    assert (resid["df"], total["df"]) == (99, 100)
    assert resid["ms"] == pytest.approx(0.646, abs=5e-4)
    assert model["F"] == pytest.approx(27397.119, abs=1e-2)
    (slope,) = r.parameters["parameter_tests"]
    assert slope["estimate"] == pytest.approx(0.979, abs=5e-4)
    assert slope["se"] == pytest.approx(0.0059, abs=5e-5)
    assert slope["t"] == pytest.approx(-3.597, abs=1e-3)
    assert slope["p"] < 0.001
    assert 0 < r.statistics["r_squared_emp"] < 1


def test_air_temperature_intercept_example_12_3():
    a = np.array(AIRTEMP)
    r = regression_details(a[:, 1], a[:, 2])
    icept, slope = r.parameters["parameter_tests"]
    assert (icept["estimate"], icept["se"]) == pytest.approx((-0.262, 0.1625), abs=5e-4)
    assert icept["t"] == pytest.approx(-1.613, abs=1e-3)
    assert icept["p"] == pytest.approx(0.110, abs=5e-4)
    assert (slope["estimate"], slope["se"]) == pytest.approx((0.996, 0.0120), abs=5e-4)
    assert r.statistics["adj_r_squared"] == pytest.approx(0.986, abs=5e-4)


def test_scaling_relationships():
    x, y = _seeds()
    std = regression_details(x, y, scaling="standardized").parameters["scaled"]
    s_x = np.std(x, ddof=1)
    assert std["intercept"] == pytest.approx(y.mean())
    assert std["slope"] == pytest.approx(17.17336 * s_x, rel=1e-6)
    assert std["slope_se"] == pytest.approx(0.666899 * s_x, rel=1e-5)
    centred = regression_details(x, y, scaling="centred").parameters["scaled"]
    assert centred["slope"] == pytest.approx(17.17336, rel=1e-6)
    assert centred["intercept_se"] == pytest.approx(0.2124, abs=1e-4)


def test_extrapolation_warning():
    x, y = _seeds()
    r = regression_details(x, y, x_predict="0")
    assert any(w.code == "extrapolation" for w in r.warnings)


def _band_root(fit, y_new, r, alpha, side):
    """x, где граница ДИ для среднего r новых наблюдений равна y_new (определение Филлера)."""
    t = stats.t.ppf(1 - alpha / 2, fit["df_res"])

    def band(x0):
        se = np.sqrt(fit["s2"] * (1 / r + 1 / fit["n"] + (x0 - fit["x_mean"]) ** 2 / fit["sxx"]))
        return fit["alpha"] + fit["beta"] * x0 + side * t * se - y_new

    x_hat = (y_new - fit["alpha"]) / fit["beta"]
    return optimize.brentq(band, x_hat - 10, x_hat + 10)


@pytest.mark.parametrize("y_new, r", [(20.0, 1), (30.5, 3), (45.0, 2)])
def test_fieller_limits_match_band_inversion(y_new, r):
    x, y = _seeds()
    res = inverse_prediction(x, y, y_new=str(y_new), replicates=r)
    (row,) = res.parameters["calibration"]
    fit = _summary(_xy_frame(x, y)[0], False)
    lower = _band_root(fit, y_new, r, 0.05, +1)  # верхняя граница пересекает раньше
    upper = _band_root(fit, y_new, r, 0.05, -1)
    assert (row["lower"], row["upper"]) == pytest.approx((lower, upper), abs=1e-8)
    assert row["x_hat"] == pytest.approx((y_new + 27.93109) / 17.17336, abs=1e-5)


def test_fieller_undefined_when_slope_not_significant():
    rng = np.random.default_rng(1)
    x = np.arange(10.0)
    res = inverse_prediction(x, rng.normal(size=10), y_new="0.1")
    assert res.parameters["calibration"][0]["g"] >= 1
    assert any(w.code == "fieller_undefined" for w in res.warnings)


def test_parse_numbers_and_json():
    assert parse_numbers("1; 2,5  3") == [1.0, 2.5, 3.0]
    with pytest.raises(ValueError):
        parse_numbers("a; b")
    r = regression_details(*_seeds(), x_predict="3", lack_of_fit="yes")
    json.dumps(r.to_dict())
    json.dumps(inverse_prediction(*_seeds(), y_new="25").to_dict())
