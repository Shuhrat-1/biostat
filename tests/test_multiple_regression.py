"""Глава 14: множественная регрессия, коллинеарность, отбор моделей.

Эталоны — Tables 14.1, 14.3, 14.5, 14.7–14.10, 14.12, 14.14, 14.16 и
Example 14.2 (сверены с картинками PDF 373, 385, 389, 395, 397).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.stats.multiple_regression import (
    model_cross_validation,
    multiple_regression,
    parse_models,
    regression_subsets,
    stepwise_selection,
)
from tests.book_ch12_data import TRITICUM
from tests.book_ch14_data import EXAMINE, EXAMINE_COLUMNS

_WHEAT_NAMES = ["Weight", "Length", "Diameter", "Moisture", "Hardness"]
_BASE = "Longitude, OctRain, NovRain, FebRain, MayRain, DecForest, Urban"


def _wheat():
    a = np.array(TRITICUM)
    cols = dict(zip(_WHEAT_NAMES, (a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])))
    return {k: v.tolist() for k, v in cols.items()}


def _examine():
    a = np.array(EXAMINE)
    return {n: a[:, i].tolist() for i, n in enumerate(EXAMINE_COLUMNS)}


def _fit(variables, **kw):
    cols = _wheat()
    return multiple_regression(cols["Weight"], cols, "Weight", variables=variables, **kw)


def test_correlations_table_14_1():
    r = _fit("Length, Diameter, Moisture, Hardness")
    corr = {(c["var1"], c["var2"]): c for c in r.parameters["correlations"]}
    assert corr[("Length", "Diameter")]["r"] == pytest.approx(0.999, abs=5e-4)
    assert corr[("Hardness", "Weight")]["r"] == pytest.approx(-0.207, abs=5e-4)
    assert corr[("Hardness", "Weight")]["p"] == pytest.approx(0.004, abs=5e-4)
    assert corr[("Moisture", "Weight")]["p"] == pytest.approx(0.390, abs=5e-4)


def test_summary_anova_table_14_3_and_parameters_14_8():
    r = _fit("Length, Hardness")
    model, resid, total = r.parameters["anova_table"]
    assert (model["ss"], resid["ss"], total["ss"]) == pytest.approx(
        (5753.4738, 1540.9352, 7294.4090), abs=1e-4)
    assert (model["df"], resid["df"]) == (2, 187)
    assert model["F"] == pytest.approx(349.106, abs=1e-3)
    icept, length, hard = r.parameters["parameter_tests"]
    assert (icept["estimate"], icept["se"], icept["t"]) == pytest.approx(
        (-27.795, 2.1653, -12.836), abs=5e-4)
    assert (length["estimate"], length["se"], length["t"]) == pytest.approx(
        (16.934, 0.6590, 25.695), abs=5e-4)
    assert (hard["estimate"], hard["se"], hard["t"]) == pytest.approx(
        (-0.049, 0.0168, -2.921), abs=5e-4)
    assert hard["p"] == pytest.approx(0.004, abs=5e-4)


def test_sequential_both_orders_table_14_5():
    a = _fit("Length, Hardness").parameters["sequential"]
    b = _fit("Hardness, Length").parameters["sequential"]
    assert [row["ss"] for row in a] == pytest.approx([5683.18, 70.30], abs=5e-3)
    assert [row["F"] for row in a] == pytest.approx([689.68, 8.53], abs=5e-3)
    assert [row["ss"] for row in b] == pytest.approx([312.83, 5440.64], abs=5e-3)
    assert [row["F"] for row in b] == pytest.approx([37.96, 660.25], abs=5e-3)


def test_marginal_table_14_7_equals_t_squared():
    r = _fit("Length, Hardness")
    marg = r.parameters["marginal"]
    assert [row["ss"] for row in marg] == pytest.approx([5440.6436, 70.2985], abs=1e-4)
    assert [row["F"] for row in marg] == pytest.approx([660.249, 8.531], abs=1e-3)
    t = [row["t"] for row in r.parameters["parameter_tests"][1:]]
    assert [row["F"] for row in marg] == pytest.approx(np.square(t).tolist())


def test_four_variates_table_14_10_and_vif():
    r = _fit("Length, Hardness, Moisture, Diameter")
    est = {row["term"]: row for row in r.parameters["parameter_tests"]}
    assert (est["[1]"]["estimate"], est["[1]"]["se"]) == pytest.approx((-12.970, 10.1545), abs=5e-4)
    assert (est["Length"]["estimate"], est["Length"]["se"]) == pytest.approx(
        (-40.155, 14.5518), abs=5e-4)
    assert (est["Diameter"]["estimate"], est["Diameter"]["se"]) == pytest.approx(
        (90.943, 23.1755), abs=5e-4)
    assert est["Moisture"]["t"] == pytest.approx(-1.922, abs=5e-4)
    # Книга печатает P 0.051, 0.002, 0.014; при t на 185 df точные P: 0.203, 0.006, 0.056.
    assert est["[1]"]["p"] == pytest.approx(0.203, abs=5e-4)
    assert est["Moisture"]["p"] == pytest.approx(0.056, abs=5e-4)
    vif = {v["variable"]: v for v in r.parameters["vif"]}
    assert [vif[n]["r_squared"] for n in ("Length", "Hardness", "Moisture", "Diameter")] == \
        pytest.approx([0.998, 0.029, 0.018, 0.998], abs=1e-3)
    assert vif["Hardness"]["vif"] == pytest.approx(1.03, abs=5e-3)
    # Книга: VIF 602.8; по данным 538.0 — согласовано с SE: VIF = SE²·SSxx_l·(1 − 0)/s².
    ss_len = float(np.var(_wheat()["Length"]) * 190)
    assert vif["Length"]["vif"] == pytest.approx(
        est["Length"]["se"] ** 2 * ss_len / r.statistics["ms_resid"], rel=1e-6)
    assert any(w.code == "collinearity" for w in r.warnings)


def test_incremental_moisture_example_14_1j():
    seq = _fit("Length, Hardness, Moisture").parameters["sequential"]
    assert seq[2]["F"] == pytest.approx(2.57, abs=5e-3)
    assert seq[2]["p"] == pytest.approx(0.111, abs=5e-4)


def test_all_subsets_table_14_12():
    cols = _wheat()
    r = regression_subsets(cols["Weight"], cols, "Weight", variables="Length, Hardness, Moisture")
    rows = {tuple(row["terms"]): row for row in r.parameters["subsets"]}
    expected = {(): (0.00, 1692.0, 1695.3, 704.6), ("Length",): (77.79, 1407.1, 1413.6, 11.2),
                ("Hardness",): (3.78, 1685.7, 1692.2, 668.4),
                ("Length", "Hardness"): (78.65, 1400.6, 1410.4, 4.6),
                ("Hardness", "Moisture"): (4.02, 1686.2, 1696.0, 663.7),
                ("Length", "Hardness", "Moisture"): (78.83, 1400.0, 1413.0, 4.0)}
    for terms, (adj, aic, sbc, cp) in expected.items():
        row = rows[terms]
        assert (100 * row["adj_r_squared"], row["aic"], row["sbc"], row["cp"]) == pytest.approx(
            (adj, aic, sbc, cp), abs=0.051)
    # Книга печатает R²adj = 0.00 для [1] + Moisture; по формуле −0.14 (отрицательное).
    assert 100 * rows[("Moisture",)]["adj_r_squared"] == pytest.approx(-0.14, abs=5e-3)
    assert r.parameters["best"]["sbc"] == ["Length", "Hardness"]
    assert r.parameters["best"]["aic"] == ["Length", "Hardness", "Moisture"]


def test_subsets_with_forced_terms_table_14_16():
    cols = _examine()
    r = regression_subsets(cols["JDay"], cols, "JDay",
                           variables=_BASE + ", C30Day, F60Day, Altitude, Latitude",
                           always_include=_BASE)
    rows = {tuple(row["terms"][7:]): row for row in r.parameters["subsets"]}
    assert len(rows) == 16
    base, f60 = rows[()], rows[("F60Day",)]
    assert (100 * base["adj_r_squared"], base["aic"], base["sbc"]) == pytest.approx(
        (46.23, 549.68, 564.97), abs=5e-3)
    assert (100 * f60["adj_r_squared"], f60["aic"], f60["sbc"]) == pytest.approx(
        (89.28, 469.86, 487.06), abs=5e-3)
    assert rows[("F60Day", "Latitude")]["p_Latitude"] == pytest.approx(0.539, abs=5e-4)
    back = rows[("C30Day", "Altitude", "Latitude")]
    assert (100 * back["adj_r_squared"], back["aic"]) == pytest.approx((89.15, 471.92), abs=5e-3)
    assert back["p_Altitude"] == pytest.approx(0.047, abs=5e-4)
    assert r.parameters["best"]["aic"][7:] == ["F60Day"]


def test_forward_selection_example_14_2():
    cols = _examine()
    r = stepwise_selection(cols["JDay"], cols, "JDay", strategy="forward_stepwise")
    order = ["F60Day", "MayRain", "OctRain", "FebRain", "Urban", "DecForest", "Longitude",
             "NovRain"]
    assert r.parameters["selected"] == order
    assert r.parameters["cycling"] == ["MarRain"]
    cyc = [s for s in r.parameters["steps"] if s["variable"] == "MarRain"]
    assert cyc[0]["F"] == pytest.approx(2.45, abs=5e-3)
    assert 100 * r.statistics["adj_r_squared"] == pytest.approx(89.3, abs=0.05)
    est = {row["term"]: row for row in r.parameters["parameter_tests"]}
    assert (est["F60Day"]["estimate"], est["F60Day"]["se"]) == pytest.approx(
        (-14.74, 1.132), abs=5e-3)
    assert (est["Urban"]["estimate"], est["Urban"]["se"]) == pytest.approx(
        (-194.76, 67.595), abs=5e-3)
    plain = stepwise_selection(cols["JDay"], cols, "JDay", strategy="forward")
    assert plain.parameters["selected"] == [*order, "MarRain"]


def test_backward_selection_example_14_2():
    cols = _examine()
    r = stepwise_selection(cols["JDay"], cols, "JDay", strategy="backward_stepwise")
    assert set(r.parameters["selected"]) == {
        "FebRain", "MayRain", "OctRain", "NovRain", "C30Day", "DecForest", "Urban",
        "Altitude", "Latitude", "Longitude"}
    assert r.parameters["cycling"] == ["MarRain"]  # книга: F = 2.15, по данным 2.31
    assert 100 * r.statistics["adj_r_squared"] == pytest.approx(89.2, abs=0.05)
    est = {row["term"]: row for row in r.parameters["parameter_tests"]}
    assert (est["[1]"]["estimate"], est["[1]"]["se"]) == pytest.approx((-88.4, 52.01), abs=0.05)
    assert est["C30Day"]["estimate"] == pytest.approx(-5.45, abs=5e-3)


def test_forced_terms_are_never_dropped():
    cols = _examine()
    r = stepwise_selection(cols["JDay"], cols, "JDay", strategy="backward",
                           always_include="Sea")
    assert "Sea" in r.parameters["selected"]


def test_partial_residuals_and_prediction():
    r = _fit("Length, Hardness", x_predict="Length=3.2, Hardness=-10")
    part = r.parameters["plots"]["partial_residuals"]["Length"]
    resid = np.array(r.parameters["plots"]["observed_vs_fitted"]["observed"]) - np.array(
        r.parameters["plots"]["observed_vs_fitted"]["fitted"])
    assert part["partial"][0] == pytest.approx(resid[0] + part["slope"] * part["x"][0])
    (pred,) = r.parameters["predictions"]
    assert pred["fitted"] == pytest.approx(-27.79466 + 16.93417 * 3.2 + 0.049014 * 10, abs=1e-4)
    assert pred["mean_lower"] < pred["fitted"] < pred["mean_upper"]
    assert pred["new_upper"] - pred["new_lower"] > pred["mean_upper"] - pred["mean_lower"]
    far = _fit("Length, Hardness", x_predict="Length=5, Hardness=0")
    assert any(w.code == "extrapolation_multi" for w in far.warnings)


def test_perfect_collinearity_is_reported():
    cols = _wheat()
    cols["Twice"] = [2 * v for v in cols["Length"]]
    r = multiple_regression(cols["Weight"], cols, "Weight", variables="Length, Twice")
    assert any(w.code == "perfect_collinearity" for w in r.warnings)


def test_model_cross_validation():
    cols = _wheat()
    assert parse_models("Length + Hardness | Length") == [["Length", "Hardness"], ["Length"]]
    r = model_cross_validation(cols["Weight"], cols, "Weight",
                               models="Length + Hardness | Length | Hardness", scheme="loo")
    rows = {tuple(m["terms"]): m for m in r.parameters["models"]}
    assert rows[("Hardness",)]["msep"] > rows[("Length",)]["msep"]
    assert r.parameters["best"] in (["Length", "Hardness"], ["Length"])


def test_validation_and_json():
    cols = _wheat()
    with pytest.raises(ValueError):
        multiple_regression(cols["Weight"], cols, "Weight", variables="Length, Nope")
    with pytest.raises(ValueError):
        stepwise_selection(cols["Weight"], cols, "Weight", strategy="lasso")
    json.dumps(_fit("Length, Hardness", x_predict="Length=3, Hardness=0").to_dict())
    json.dumps(regression_subsets(cols["Weight"], cols, "Weight").to_dict())
