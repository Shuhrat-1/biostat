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
