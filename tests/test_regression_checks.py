"""Глава 13: остатки, рычаги, влияние, кросс-валидация.

Эталоны — Table 13.2, Examples 13.2A–C (сверены с картинками PDF 349, 358, 360);
формулы мер влияния и leave-1-out — с полным перебором и statsmodels.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import statsmodels.api as sm

from core.stats.linear_regression import regression_details
from core.stats.regression_checks import (
    half_normal_quantiles,
    influence_measures,
    prediction_summary,
    regression_cross_validation,
    regression_diagnostics,
)
from tests.book_ch12_data import TRITICUM

# Table 13.1: концентрация субстрата и поглощение ELISA (Example 13.1).
ELISA_CONC = [0, 0.5, 1, 2, 4, 8, 16, 32]
ELISA_ABS = [0.100, 0.678, 1.107, 1.609, 1.958, 2.202, 2.414, 2.485]


def _seeds():
    a = np.array(TRITICUM)
    return a[:, 2], a[:, 1]


def test_elisa_table_13_2():
    x = np.log10(np.array(ELISA_CONC) + 1)
    r = regression_details(x, ELISA_ABS)
    icept, slope = r.parameters["parameter_tests"]
    assert (icept["estimate"], icept["se"], icept["t"]) == pytest.approx(
        (0.5458, 0.19392, 2.815), abs=5e-4)
    assert icept["p"] == pytest.approx(0.031, abs=5e-4)
    assert (slope["estimate"], slope["se"], slope["t"]) == pytest.approx(
        (1.5284, 0.23244, 6.575), abs=5e-4)
    assert r.statistics["adj_r_squared"] == pytest.approx(0.858, abs=5e-4)


def test_x_transform_in_diagnostics_matches_pretransformed():
    r = regression_diagnostics(ELISA_CONC, ELISA_ABS, x_transform="log10", x_offset=1)
    assert r.parameters["line"] == pytest.approx({"intercept": 0.54580, "slope": 1.52837},
                                                 abs=1e-5)


def test_leverage_example_13_2b():
    r = regression_diagnostics(*_seeds())
    assert r.parameters["thresholds"]["leverage"] == pytest.approx(0.021, abs=5e-4)
    assert r.statistics["max_leverage"] == pytest.approx(0.042, abs=5e-4)
    lev = [o["leverage"] for o in r.parameters["observations"]]
    assert sum(lev) == pytest.approx(2.0)  # Σ h_ii = p


def test_influence_example_13_2c():
    r = regression_diagnostics(*_seeds())
    assert r.parameters["thresholds"]["modified_cook"] == pytest.approx(1.99, abs=5e-3)
    assert r.statistics["n_influential"] == 8
    refit = r.parameters["refit_without_influential"]
    icept, slope = refit["parameters"]
    assert refit["n_omitted"] == 8
    # Книга: «intercept decreases by −0.15», по расчёту −0.139; наклон +0.006 совпадает.
    assert icept["change"] == pytest.approx(-0.139, abs=5e-4)
    assert slope["change"] == pytest.approx(0.006, abs=5e-4)


def test_largest_residual_example_13_2a():
    r = regression_diagnostics(*_seeds())
    obs = max(r.parameters["observations"], key=lambda o: abs(o["deletion_residual"]))
    assert 36 < obs["fitted"] < 37  # «fitted value of just under 37»


def test_measures_match_statsmodels():
    x, y = _seeds()
    design = sm.add_constant(x)
    m = influence_measures(design, y)
    ref = sm.OLS(y, design).fit().get_influence()
    assert m["leverage"] == pytest.approx(ref.hat_matrix_diag)
    assert m["std_residual"] == pytest.approx(ref.resid_studentized_internal)
    assert m["deletion_residual"] == pytest.approx(ref.resid_studentized_external)
    assert m["cook_d"] == pytest.approx(ref.cooks_distance[0])


def test_prediction_and_deletion_residuals_by_refitting():
    x, y = _seeds()
    m = influence_measures(sm.add_constant(x), y)
    for i in (0, 47, 189):
        keep = np.arange(y.size) != i
        fit = sm.OLS(y[keep], sm.add_constant(x[keep])).fit()
        pred = fit.params[0] + fit.params[1] * x[i]
        assert m["prediction_residual"][i] == pytest.approx(y[i] - pred)
        assert m["s2_deleted"][i] == pytest.approx(fit.mse_resid)


def test_modified_cook_formula():
    x, y = _seeds()
    m = influence_measures(sm.add_constant(x), y)
    n, p, h = y.size, 2, m["leverage"]
    expected = np.abs(m["deletion_residual"]) * np.sqrt((n - p) / p * h / (1 - h))
    assert m["modified_cook"] == pytest.approx(expected)


def test_half_normal_quantiles():
    q = half_normal_quantiles(4)
    assert q[0] == pytest.approx(0.1854, abs=1e-4)  # Φ⁻¹(0.5 + 0.5·0.625/4.25) = Φ⁻¹(0.5735)
    assert np.all(np.diff(q) > 0)


def test_through_origin_has_one_parameter():
    x, y = _seeds()
    r = regression_diagnostics(x, y, through_origin="yes")
    assert r.parameters["p"] == 1
    assert sum(o["leverage"] for o in r.parameters["observations"]) == pytest.approx(1.0)


def test_warnings_list_original_rows():
    x, y = list(_seeds()[0]), list(_seeds()[1])
    x.insert(0, float("nan"))
    y.insert(0, 1.0)
    r = regression_diagnostics(x, y)
    rows = {o["row"] for o in r.parameters["observations"] if o["influential"]}
    assert rows == {21, 32, 48, 123, 139, 188, 189, 191}  # сдвиг на пустую первую строку
    assert any(w.code == "influential_points" for w in r.warnings)


def test_prediction_summary_formulas():
    s = prediction_summary([10.0, 12.0, 14.0], [11.0, 11.0, 16.0])
    assert s["pb"] == pytest.approx(-2 / 3)
    assert s["mad"] == pytest.approx(4 / 3)
    assert s["msep"] == pytest.approx(2.0)
    assert s["rmse"] == pytest.approx(np.sqrt(2.0))
    assert s["pb_pct"] == pytest.approx(100 * (-2 / 3) / 12)


def test_loo_equals_prediction_residuals():
    x, y = _seeds()
    cv = regression_cross_validation(x, y, scheme="loo")
    m = influence_measures(sm.add_constant(x), y)
    d = [row["discrepancy"] for row in cv.parameters["discrepancies"]]
    assert d == pytest.approx(m["prediction_residual"].tolist())
    assert cv.statistics["press"] == pytest.approx(float((m["prediction_residual"] ** 2).sum()))
    assert cv.statistics["msep"] == pytest.approx(cv.statistics["press"] / 190)


def test_split_example_13_2d_structure():
    x, y = _seeds()
    cv = regression_cross_validation(x, y, scheme="split", train_fraction=0.5, seed=7)
    train = cv.parameters["training"]
    assert train["n"] == 95 and cv.statistics["n_validation"] == 95
    # Случайное разбиение книги не воспроизводимо; проверяем согласованность с ручным расчётом.
    val_rows = {r["row"] for r in cv.parameters["discrepancies"]}
    keep = np.array([i + 1 not in val_rows for i in range(190)])
    fit = sm.OLS(y[keep], sm.add_constant(x[keep])).fit()
    assert train["parameters"][1]["estimate"] == pytest.approx(fit.params[1])
    assert train["s"] == pytest.approx(np.sqrt(fit.mse_resid))
    pred = fit.params[0] + fit.params[1] * x[~keep]
    assert cv.statistics["rmse"] == pytest.approx(np.sqrt(((y[~keep] - pred) ** 2).mean()))
    again = regression_cross_validation(x, y, scheme="split", seed=7)
    assert again.statistics["pb"] == cv.statistics["pb"]


def test_kfold_covers_every_row_once():
    x, y = _seeds()
    cv = regression_cross_validation(x, y, scheme="kfold", k_folds=4, seed=3)
    rows = [r["row"] for r in cv.parameters["discrepancies"]]
    assert sorted(rows) == list(range(1, 191))
    assert {r["fold"] for r in cv.parameters["discrepancies"]} == {1, 2, 3, 4}
    assert "press" not in cv.statistics


def test_invalid_scheme_and_json():
    x, y = _seeds()
    with pytest.raises(ValueError):
        regression_cross_validation(x, y, scheme="bootstrap")
    json.dumps(regression_diagnostics(x, y).to_dict())
    json.dumps(regression_cross_validation(x, y).to_dict())
    small = regression_diagnostics([1, 2, 3], [1, 2, 4])
    assert any(w.code == "n_too_small" for w in small.warnings)
