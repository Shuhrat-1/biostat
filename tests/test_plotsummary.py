"""Графики-итоги (пакет 1): средние с ДИ, взаимодействие, ДИ разностей."""

from __future__ import annotations

import json

import pytest

from core.stats.anova import tukey_hsd
from core.stats.plotsummary import (
    MAX_PAIRS, differences_from, interaction_from_rows, means_from_rows, summary_plots,
)
from core.stats.transforms import transformed_means
from tests.test_transforms import BEETLES
from tests.test_treatment_means import CALCIUM


def _cell(a: str, b: str, mean: float, **extra) -> dict:
    return {"label": f"A={a} × B={b}", "mean": mean, "ci_lower": mean - 1,
            "ci_upper": mean + 1, **extra}


# --- Разности ------------------------------------------------------------------------

def test_tukey_calcium_label_follows_core_sign():
    """Разность ядра — group2 − group1: подпись «B − A», знак как в mean_diff."""
    res = tukey_hsd(*CALCIUM.values(), labels=list(CALCIUM)).to_dict()
    diff = summary_plots("tukey_hsd", res)["differences"]
    first = res["parameters"]["comparisons"][0]
    item = diff["items"][0]
    assert item["label"] == f"{first['group2']} − {first['group1']}"
    assert item["diff"] == first["mean_diff"]
    assert item["lower"] < item["diff"] < item["upper"]
    assert diff["reference"] == 0.0
    json.dumps(diff, allow_nan=False)


def test_half_width_used_when_ci_missing():
    out = differences_from([{"group1": "a", "group2": "b", "mean_diff": 2.0, "half_width": 0.5}])
    assert (out["items"][0]["lower"], out["items"][0]["upper"]) == (1.5, 2.5)


def test_too_many_pairs_left_to_table():
    pairs = [{"group1": "a", "group2": str(i), "mean_diff": 1.0, "ci_lower": 0.0,
              "ci_upper": 2.0} for i in range(MAX_PAIRS + 1)]
    assert differences_from(pairs) is None


def test_equivalence_percent_margin_and_control_label():
    params = {"margin": 10, "margin_kind": "percent", "control_mean": 200.0, "control": "C",
              "equivalence": [{"level": "T", "difference": 5.0, "ci_lower": -3.0,
                               "ci_upper": 13.0}]}
    out = summary_plots("equivalence_test", {"parameters": params})["differences"]
    assert out["margin"] == pytest.approx(20.0)
    assert out["items"][0]["label"] == "T − C"


# --- Средние --------------------------------------------------------------------------

def test_transformed_means_on_original_scale_table_6_1():
    """Гл. 6: средние после log10 на исходной шкале — ДИ асимметричен; отношения — около 1."""
    res = transformed_means(*BEETLES.values(), labels=list(BEETLES)).to_dict()
    out = summary_plots("transformed_means", res)
    item = out["means_ci"]["items"][0]
    assert item["label"] == "1"
    assert item["upper"] - item["mean"] > item["mean"] - item["lower"] > 0
    assert out["differences"]["kind"] == "ratio" and out["differences"]["reference"] == 1.0


def test_back_scale_and_non_estimable_rows():
    rows = [{"label": "a", "mean": 1.0, "ci_lower": 0.5, "ci_upper": 1.5,
             "back_mean": 10.0, "back_ci_lower": 3.0, "back_ci_upper": 30.0},
            {"label": "b", "mean": 2.0, "ci_lower": 1.0, "ci_upper": 3.0, "estimable": False}]
    out = means_from_rows(rows, back=True)
    assert out["items"] == [{"label": "a", "mean": 10.0, "lower": 3.0, "upper": 30.0}]


def test_polynomial_levels_become_numeric_axis():
    means = [{"level": lv, "mean": m, "ci_lower": m - 1, "ci_upper": m + 1}
             for lv, m in (("0", 1.0), ("2,5", 2.0), ("5", 3.0))]
    out = summary_plots("polynomial_contrasts", {"parameters": {"means": means}})["means_ci"]
    assert out["numeric"] and [it["x"] for it in out["items"]] == [0.0, 2.5, 5.0]


def test_ancova_adjusted_means_are_se_bars():
    params = {"adjusted_means": [{"group": "g", "mean": 3.0, "se": 0.2}]}
    out = summary_plots("ancova", {"parameters": params})["means_ci"]
    assert out["interval"] == "se" and out["items"][0]["upper"] == pytest.approx(3.2)


# --- Взаимодействие ------------------------------------------------------------------

def test_interaction_grid_with_missing_cell():
    rows = [_cell("a1", "b1", 1.0), _cell("a2", "b1", 2.0), _cell("a1", "b2", 3.0)]
    out = interaction_from_rows(rows, back=False)
    assert (out["factor_a"], out["factor_b"]) == ("A", "B")
    assert out["levels_a"] == ["a1", "a2"]
    assert out["series"][1] == {"name": "b2", "values": [3.0, None], "lower": [2.0, None],
                                "upper": [4.0, None]}


def test_single_factor_or_three_factor_rows_are_not_interaction():
    assert interaction_from_rows([{"label": "A=a1", "mean": 1.0, "ci_lower": 0,
                                   "ci_upper": 2}], back=False) is None
    three = [{"label": "A=a × B=b × C=c", "mean": 1.0, "ci_lower": 0, "ci_upper": 2}]
    assert interaction_from_rows(three, back=False) is None


def test_model_table_not_drawn_twice():
    params = {"tables": [{"term": "A.B", "rows": [_cell("a1", "b1", 1.0), _cell("a2", "b1", 2.0)]},
                         {"term": "A", "rows": [{"label": "A=a1", "mean": 1.0, "ci_lower": 0,
                                                 "ci_upper": 2}]}]}
    out = summary_plots("factorial_anova", {"parameters": params})
    assert set(out) == {"interaction", "means_ci"}
    assert out["means_ci"]["term"] == "A"


def test_methods_without_summary_return_nothing():
    assert summary_plots("simple_linear_regression", {"parameters": {"x": [1, 2]}}) == {}


# --- Пакет 2: кривые по группам, графики против x -----------------------------------------

def test_group_lines_forest_table_15_2():
    """Регрессия с группами: точки с метками групп и по линии на группу; scatter моста убирается."""
    from tests.test_regression_groups import _forest

    out = summary_plots("regression_with_groups", _forest().to_dict())
    g = out["group_curves"]
    assert [c["name"] for c in g["curves"]] == ["Coigue", "Rauli", "Roble"]
    assert {p[2] for p in g["points"]} == {"Coigue", "Rauli", "Roble"}
    assert out["scatter"] is None
    json.dumps(g, allow_nan=False)


def test_block_curves_need_two_groups_and_keep_finite_band():
    cv = {"x": [0.0, 1.0], "fitted": [1.0, 2.0], "lower": [0.5, 1.5], "upper": [1.5, 2.5]}
    params = {"points": {"x": [0.0, 1.0], "y": [1.1, 1.9]},
              "curves": [{"group": "A", **cv},
                         {"group": "B", **cv, "upper": [1.5, float("nan")]}]}
    g = summary_plots("nonlinear_in_blocks", {"parameters": params})["group_curves"]
    assert "band" in g["curves"][0] and "band" not in g["curves"][1]
    assert g["points"][0] == [0.0, 1.1, None]
    one = {"parameters": {**params, "curves": params["curves"][:1]}}
    assert "group_curves" not in summary_plots("curve_in_blocks", one)


def test_extra_scatters_with_reference_lines():
    plots = {"leverage_vs_x": {"x": [1.0, 2.0], "leverage": [0.2, 0.6]},
             "cook_vs_fitted": {"fitted": [1.0, 2.0], "cook": [0.1, 2.5]},
             "residual_vs_x1": [1.0, 2.0], "residual_vs_x2": [5.0, 6.0],
             "std_residuals": [0.3, -0.3]}
    params = {"plots": plots, "variables": ["H", "I"],
              "thresholds": {"leverage": 0.5, "modified_cook": 1.73}}
    extra = {(e["kind"], e["x_name"]): e for e in
             summary_plots("regression_diagnostics", {"parameters": params})["extra_scatter"]}
    assert extra[("leverage_vs_x", "")]["ref"] == [0.5]
    assert extra[("cook_vs_fitted", "")]["ref"] == [1.73]
    assert extra[("residual_vs_x", "I")]["points"] == [[5.0, 0.3], [6.0, -0.3]]


# --- Пакет 3: GLM, порядковые, сопряжённость, парные -------------------------------------

def _pred(level: str, natural: float, lo, hi, factor: str = "A") -> dict:
    return {"factor": factor, "level": level, "natural": natural, "natural_lower": lo,
            "natural_upper": hi}


def test_glm_predictions_first_factor_on_natural_scale():
    preds = [_pred("a", 0.2, 0.1, 0.35), _pred("b", 0.4, 0.3, 0.5),
             _pred("x", 5.0, 4.0, 6.0, factor="B")]
    out = summary_plots("glm_analysis", {"parameters": {"factor_predictions": preds}})["means_ci"]
    assert (out["term"], out["title"], out["interval"]) == ("A", "predictions", "ci")
    assert [it["label"] for it in out["items"]] == ["a", "b"]
    assert out["items"][0]["upper"] - 0.2 > 0.2 - out["items"][0]["lower"]  # асимметрия


def test_predictions_without_ci_are_points_only():
    preds = [_pred("a", 4.6, None, None), _pred("b", 15.4, None, None)]
    out = summary_plots("count_regression", {"parameters": {"factor_predictions": preds}})
    assert out["means_ci"]["interval"] == "none"
    assert out["means_ci"]["items"][0]["lower"] == out["means_ci"]["items"][0]["upper"] == 4.6


def test_ordinal_probabilities_keep_category_order():
    params = {"categories": ["healthy", "slight", "severe"],
              "category_probabilities": [{"factor": "A", "level": "c", "probabilities":
                                          {"severe": 0.1, "healthy": 0.6, "slight": 0.3}}]}
    st = summary_plots("ordinal_regression", {"parameters": params})["stacked"]
    assert st["categories"] == ["healthy", "slight", "severe"]
    assert st["rows"] == [{"label": "c", "values": [0.6, 0.3, 0.1]}]


def test_contingency_cells_indexed_by_table_order():
    params = {"table": {"rows": ["A", "B"], "columns": ["x", "y"]},
              "cells": [{"row": "B", "column": "y", "observed": 9.0, "expected": 5.0,
                         "adjusted_residual": 2.5}]}
    heat = summary_plots("contingency_test", {"parameters": params})["heatmap"]
    assert heat["cells"] == [[1, 1, 2.5, 9.0, 5.0]]


def test_paired_differences_follow_core_sign_and_line_limit():
    from core.stats.hypothesis import paired_t
    from core.stats.plotsummary import MAX_PAIR_LINES, paired_plot

    before, after = [12.1, 14.3, 11.8, 15.2], [13.0, 15.1, 11.2, 16.8]
    pp = paired_plot(before, after, ("до", "после"))
    mean_diff = sum(b - a for b, a in zip(before, after)) / 4
    core = paired_t(before, after).to_dict()
    assert core["statistics"]["mean_diff"] == pytest.approx(mean_diff)
    assert sum(pp["diff_hist"]["counts"]) == 4 and pp["labels"] == ["до", "после"]
    many = paired_plot([1.0] * (MAX_PAIR_LINES + 1), [2.0] * (MAX_PAIR_LINES + 1), ("a", "b"))
    assert many["pairs"] is None


# --- Пакет 4: мощность, отбор подмножеств, MSEP, поверхность ----------------------------

def test_power_curve_with_target_line():
    rows = [{"n": n, "power": pw} for n, pw in ((2, 0.15), (3, 0.24), (4, None))]
    params = {"replication_table": rows, "target_power": 0.8}
    (power,) = summary_plots("replication_power", {"parameters": params})["extra_scatter"]
    assert power["kind"] == "power_vs_n" and power["line"] is True
    assert power["points"] == [[2, 0.15], [3, 0.24]] and power["ref"] == [0.8]


def test_subsets_cp_has_diagonal_and_terms_as_labels():
    subsets = [{"terms": ["a"], "p": 2, "cp": 5.0, "adj_r_squared": 0.4},
               {"terms": ["a", "b"], "p": 3, "cp": 3.1, "adj_r_squared": 0.6}]
    extra = {e["kind"]: e for e in summary_plots(
        "regression_subsets", {"parameters": {"subsets": subsets}})["extra_scatter"]}
    assert extra["cp_vs_p"]["diagonal"] and not extra["adj_r2_vs_p"]["diagonal"]
    assert extra["cp_vs_p"]["labels"] == ["a", "a, b"]
    assert extra["adj_r2_vs_p"]["points"] == [[2, 0.4], [3, 0.6]]


def test_cv_models_mark_best():
    params = {"models": [{"terms": ["L", "H"], "msep": 8.4}, {"terms": ["L"], "msep": 8.7}],
              "best": ["L", "H"]}
    bars = summary_plots("model_cross_validation", {"parameters": params})["bars"]
    assert bars["best"] == "L, H" and [it["value"] for it in bars["items"]] == [8.4, 8.7]


def test_surface_grid_indices_and_shape_guard():
    sf = {"x1": [0.0, 1.0], "x2": [0.0, 10.0, 20.0], "fitted": [[1, 2, 3], [4, 5, 6]]}
    out = summary_plots("response_surface", {"parameters": {"surface": sf, "variables": ["N", "P"]}})
    assert out["surface"]["names"] == ["N", "P"]
    assert out["surface"]["cells"][-1] == [1, 2, 6]
    bad = {**sf, "fitted": [[1, 2], [4, 5]]}
    assert "surface" not in summary_plots("response_surface", {"parameters": {"surface": bad}})


# --- Пакет 5: кривые моделей в сравнении --------------------------------------------------

def test_curve_comparison_curves_marked_by_criteria_table_17_12():
    """Table 17.12: R²adj и AIC — у полинома 8, SBC — у inverse_linear."""
    from core.stats.nonlinear import curve_comparison
    from tests.book_ch17_data import OLSEN_P, YIELD

    res = curve_comparison(OLSEN_P, YIELD, "log10, poly2, poly8, inverse_linear").to_dict()
    g = summary_plots("curve_comparison", res)["group_curves"]
    names = [c["name"] for c in g["curves"]]
    assert names == ["log10", "poly2", "poly8 (R²adj, AIC)", "inverse_linear (SBC)"]
    assert g["title"] == "models" and len(g["points"]) == len(OLSEN_P)
    json.dumps(g, allow_nan=False)
