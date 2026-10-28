"""Глава 15.1–15.2: регрессия с группами (данные Tables 15.1, 15.7)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.stats.regression_groups import regression_with_groups
from tests.book_ch15_data import FOREST, WEEDSEED


def _forest(**kw):
    cols = {"logSD": np.log([r[1] for r in FOREST]).tolist(),
            "logQD": np.log([r[2] for r in FOREST]).tolist()}
    return regression_with_groups(cols["logSD"], cols, [r[0] for r in FOREST], "logSD",
                                  column_names={"group": "Type"}, **kw)


def _weed(**kw):
    cols = {"TotalSeed": [r[3] for r in WEEDSEED], "HLength": [r[1] for r in WEEDSEED],
            "Spikelets": [r[2] for r in WEEDSEED]}
    kw = {"variables": "HLength, Spikelets", "group_first": "yes", "transform_kind": "log10",
          **kw}
    return regression_with_groups(cols["TotalSeed"], cols, [r[0] for r in WEEDSEED],
                                  "TotalSeed", column_names={"group": "Site"}, **kw)


def _rows(table):
    return {r["term"]: r for r in table}


def test_sequential_anova_table_15_5():
    """Книга: Table 15.5 (стр. 393), Example 15.1H."""
    rows = _rows(_forest().parameters["sequential"])
    assert [rows[t]["ss"] for t in ("+ logQD", "+ Type", "+ logQD.Type")] == pytest.approx(
        [4.5833, 1.9403, 0.2011], abs=1e-4)
    assert [rows[t]["F"] for t in ("+ logQD", "+ Type", "+ logQD.Type")] == pytest.approx(
        [78.562, 16.629, 1.724], abs=1e-3)
    assert rows["+ logQD.Type"]["p"] == pytest.approx(0.193, abs=5e-4)
    assert (rows["Residual"]["df"], rows["Residual"]["ss"]) == (35, pytest.approx(2.0419, abs=1e-4))
    assert rows["Total"]["ss"] == pytest.approx(8.7667, abs=1e-4)


def test_groups_first_table_15_6():
    """Книга: Table 15.6 (стр. 397), Example 15.1J."""
    rows = _rows(_forest().parameters["sequential_reverse"])
    assert rows["+ Type"]["ss"] == pytest.approx(0.3025, abs=1e-4)
    assert rows["+ Type"]["F"] == pytest.approx(2.592, abs=1e-3)
    assert rows["+ Type"]["p"] == pytest.approx(0.089, abs=5e-4)
    assert rows["+ logQD"]["F"] == pytest.approx(106.636, abs=1e-3)


def test_standard_models_examples_15_1a_to_e():
    """Книга: Examples 15.1A–G (стр. 385–391): R²adj 0.734, 0.723, 0.511, 0; ModSS 4.583, 6.524, 6.725."""
    models = {m["model"]: m for m in _forest().parameters["standard_models"]}
    assert models["separate_lines"]["adj_r_squared"] == pytest.approx(0.734, abs=5e-4)
    assert models["parallel_lines"]["adj_r_squared"] == pytest.approx(0.723, abs=5e-4)
    assert models["single_line"]["adj_r_squared"] == pytest.approx(0.511, abs=5e-4)
    assert models["null"]["adj_r_squared"] == 0
    assert [models[m]["model_ss"] for m in ("single_line", "parallel_lines", "separate_lines")] \
        == pytest.approx([4.583, 6.524, 6.725], abs=5e-4)
    assert [models[m]["model_df"] for m in ("single_line", "parallel_lines", "separate_lines")] \
        == [1, 3, 5]


def test_selects_parallel_lines_table_15_2():
    """Книга: Table 15.2 (стр. 386), Examples 15.1B и 15.1H (стр. 393)."""
    r = _forest()
    assert r.parameters["model_type"] == "parallel_lines"
    est = _rows(r.parameters["parameter_tests"])
    assert [est[f"α[{t}]"]["estimate"] for t in ("Coigue", "Rauli", "Roble")] == pytest.approx(
        [12.270, 11.926, 11.735], abs=5e-4)
    assert [est[f"α[{t}]"]["se"] for t in ("Coigue", "Rauli", "Roble")] == pytest.approx(
        [0.4393, 0.4236, 0.4041], abs=5e-5)
    assert (est["β_logQD"]["estimate"], est["β_logQD"]["se"]) == pytest.approx(
        (-1.536, 0.1516), abs=5e-4)


def test_intercept_differences_example_15_1f_and_15_1i():
    """Книга: Example 15.1F (стр. 389): ν̂2 −0.344 (SE 0.1083), ν̂3 −0.536 (SE 0.0948); 15.1I (стр. 395)."""
    diffs = {d["contrast"]: d for d in _forest().parameters["intercept_differences"]}
    assert (diffs["Rauli − Coigue"]["difference"], diffs["Rauli − Coigue"]["se"]) == \
        pytest.approx((-0.344, 0.1083), abs=5e-4)
    assert (diffs["Roble − Coigue"]["difference"], diffs["Roble − Coigue"]["se"]) == \
        pytest.approx((-0.536, 0.0948), abs=5e-4)


def test_weed_sequential_tables_15_8():
    """Книга: Table 15.8a–b (стр. 403)."""
    r = _weed()
    a, b = _rows(r.parameters["sequential"]), _rows(r.parameters["sequential_reverse"])
    assert [a[t]["ms"] for t in ("+ Site", "+ HLength", "+ Spikelets", "+ HLength.Site",
                                 "+ Spikelets.Site")] == pytest.approx(
        [4.406, 0.291, 0.155, 0.070, 0.027], abs=5e-4)
    assert [a[t]["p"] for t in ("+ HLength", "+ Spikelets", "+ HLength.Site",
                                "+ Spikelets.Site")] == pytest.approx(
        [0.009, 0.054, 0.172, 0.584], abs=5e-4)
    assert [b[t]["ms"] for t in ("+ Spikelets", "+ HLength", "+ Spikelets.Site",
                                 "+ HLength.Site")] == pytest.approx(
        [0.351, 0.095, 0.024, 0.072], abs=5e-4)
    assert [b[t]["p"] for t in ("+ Spikelets.Site", "+ HLength.Site")] == pytest.approx(
        [0.623, 0.159], abs=5e-4)
    assert a["Residual"]["df"] == 75


def test_weed_selection_example_15_2():
    """Книга: Example 15.2 (стр. 403–404): исключены Spikelets.Site, HLength.Site, HLength."""
    r = _weed()
    steps = r.parameters["selection_steps"]
    assert [s["term"] for s in steps if s["dropped"]] == ["Spikelets.Site", "HLength.Site",
                                                          "HLength"]
    assert steps[2]["F"] == pytest.approx(2.32, abs=5e-3)
    assert steps[2]["p"] == pytest.approx(0.132, abs=5e-4)
    assert r.parameters["selected_terms"] == ["Site", "Spikelets"]
    assert r.statistics["adj_r_squared"] == pytest.approx(0.788, abs=5e-4)


def test_weed_parameters_table_15_10():
    """Книга: Table 15.10 (стр. 405)."""
    est = _rows(_weed().parameters["parameter_tests"])
    assert [est[f"α[{s}]"]["estimate"] for s in "CLPW"] == pytest.approx(
        [3.450, 3.671, 4.343, 4.376], abs=5e-4)
    assert [est[f"α[{s}]"]["se"] for s in "CLPW"] == pytest.approx(
        [0.2646, 0.2249, 0.2547, 0.2142], abs=5e-5)
    assert (est["β_Spikelets"]["estimate"], est["β_Spikelets"]["se"]) == pytest.approx(
        (0.0277, 0.00955), abs=5e-5)
    assert est["β_Spikelets"]["t"] == pytest.approx(2.895, abs=5e-4)


def test_weed_ratios_example_15_2():
    """Книга: Example 15.2 (стр. 405): W/C = 8.43 (5.74, 12.39); 10^β = 1.066 (1.02, 1.11)."""
    r = _weed()
    wc = next(d for d in r.parameters["intercept_differences"] if d["contrast"] == "W − C")
    assert (wc["difference"], wc["se"]) == pytest.approx((0.926, 0.084), abs=5e-4)
    # Книга возводит в степень округлённые 0.926 и 1.093: 8.43 и 12.39; точно 8.44 и 12.41.
    assert (wc["ci_lower"], wc["ci_upper"]) == pytest.approx((0.759, 1.093), abs=1e-3)
    assert (wc["ratio"], wc["ratio_lower"], wc["ratio_upper"]) == pytest.approx(
        (8.442, 5.741, 12.41), abs=5e-3)
    (slope,) = r.parameters["slope_factors"]
    assert (slope["factor"], slope["lower"], slope["upper"]) == pytest.approx(
        (1.066, 1.02, 1.11), abs=5e-3)


def test_poor_overlap_is_flagged():
    """Книга: стр. 396 — сравнивать группы при сходных диапазонах x (порог 0.5 — эвристика BioStat)."""
    assert any(w.code == "groups_poor_overlap" for w in _weed().warnings)
    assert not any(w.code == "groups_poor_overlap" for w in _forest().warnings)


def test_separate_lines_kept_when_slopes_differ():
    rng = np.random.default_rng(4)
    x = np.tile(np.linspace(0, 10, 12), 2)
    g = ["a"] * 12 + ["b"] * 12
    y = np.where(np.array(g) == "a", 1 + 2 * x, 1 - 1 * x) + rng.normal(0, 0.3, 24)
    cols = {"y": y.tolist(), "x": x.tolist()}
    r = regression_with_groups(cols["y"], cols, g, "y")
    assert r.parameters["model_type"] == "separate_lines"
    est = _rows(r.parameters["parameter_tests"])
    assert est["β_x[a]"]["estimate"] == pytest.approx(2, abs=0.1)
    assert est["β_x[b]"]["estimate"] == pytest.approx(-1, abs=0.1)
    assert "intercept_differences" not in r.parameters


def test_guards_and_json():
    cols = {"y": [1.0, 2, 3, 4, 5, 6], "x": [1.0, 2, 3, 4, 5, 6]}
    small = regression_with_groups(cols["y"], cols, ["a", "a", "b", "b", "b", "b"], "y")
    assert small.warnings[-1].code == "group_too_small"
    one = regression_with_groups(cols["y"], cols, ["a"] * 6, "y")
    assert one.warnings[-1].code == "factor_one_level"
    json.dumps(_forest().to_dict())
    json.dumps(_weed().to_dict())
