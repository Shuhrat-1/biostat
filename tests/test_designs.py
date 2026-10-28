"""Тесты дизайнов гл. 9: латинский квадрат, расщеплённые делянки, BIBD.

Эталоны — Tables 9.1, 9.2, 9.4–9.6, 9.10, 9.12, 9.14, 9.20 и Example 9.5C, 9.9B
(сверены с картинками страниц PDF 233, 235–239, 244, 247–250, 256, 259); для BIBD
без данных в книге — сверка с OLS statsmodels (внутриблочный анализ).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from core.stats.designs import (
    bibd_anova,
    latin_square_anova,
    satterthwaite_df,
    split_plot_anova,
)

# Table 9.1 (стр. 211): ряд × колонка → вариант (почва, вода) и высота.
LUPIN_LS = [["CH 19.6", "SL 23.5", "CL 21.7", "SH 19.0"],
            ["CL 15.5", "SH 22.4", "CH 23.2", "SL 19.3"],
            ["SH 18.5", "CH 23.5", "SL 26.4", "CL 19.0"],
            ["SL 19.8", "CL 19.8", "SH 23.9", "CH 20.8"]]

# Table 9.10 (стр. 222): блок → целые делянки 1, 2 (орошение — серые) → (вид, урожай).
COMPETITION = {
    1: [("yes", [("-", 7.92), ("Am", 3.62), ("Sm", 5.70), ("Ga", 4.49)]),
        ("no", [("-", 9.11), ("Sm", 6.77), ("Ga", 7.59), ("Am", 4.12)])],
    2: [("no", [("-", 8.02), ("Ga", 5.72), ("Sm", 6.32), ("Am", 3.19)]),
        ("yes", [("Am", 2.52), ("Ga", 4.70), ("-", 7.05), ("Sm", 5.91)])],
    3: [("yes", [("Sm", 4.91), ("Ga", 2.20), ("-", 5.54), ("Am", 1.97)]),
        ("no", [("Am", 2.92), ("Sm", 6.64), ("Ga", 6.90), ("-", 8.18)])],
    4: [("no", [("Am", 3.71), ("-", 7.16), ("Ga", 6.51), ("Sm", 6.65)]),
        ("yes", [("Ga", 4.91), ("Sm", 5.78), ("Am", 2.73), ("-", 8.22)])],
}

# Table 9.20 (стр. 234): BIBD для 7 вариантов, блоки по 3 (a) и по 4 (b).
BIBD_A = [[1, 5, 7], [6, 4, 7], [2, 5, 4], [2, 7, 3], [4, 3, 1], [5, 3, 6], [2, 6, 1]]
BIBD_B = [[4, 1, 2, 7], [7, 6, 5, 2], [3, 6, 2, 4], [3, 4, 5, 7], [6, 5, 1, 4],
          [6, 3, 7, 1], [3, 5, 1, 2]]


def _rows(result) -> dict[str, dict]:
    return {r["term"]: r for r in result.parameters["anova_table"]}


def _lupin() -> dict[str, list[str] | list[float]]:
    data: dict[str, list] = {"y": [], "soil": [], "water": [], "row": [], "col": []}
    for i, cells in enumerate(LUPIN_LS, 1):
        for j, cell in enumerate(cells, 1):
            trt, value = cell.split()
            for key, val in zip(data, (float(value), trt[0], trt[1], str(i), str(j))):
                data[key].append(val)
    return data


def _competition() -> dict[str, list]:
    data: dict[str, list] = {"y": [], "irr": [], "species": [], "block": [], "wp": []}
    for block, plots in COMPETITION.items():
        for wp, (irr, cells) in enumerate(plots, 1):
            for species, value in cells:
                for key, val in zip(data, (value, irr, species, str(block), str(wp))):
                    data[key].append(val)
    return data


def _bibd(design: list[list[int]], seed: int = 5) -> tuple[list[float], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    block = [str(i) for i, b in enumerate(design, 1) for _ in b]
    trt = [f"T{t}" for b in design for t in b]
    y = [10 + 0.5 * int(t[1:]) + int(b) + rng.normal() for t, b in zip(trt, block)]
    return y, trt, block


# --- Латинский квадрат: Tables 9.2, 9.4–9.6 -----------------------------------


@pytest.fixture(scope="module")
def lupin():
    d = _lupin()
    return latin_square_anova(d["y"], d["soil"], d["row"], d["col"], factor_b=d["water"],
                              column_names={"factor_a": "Soil", "factor_b": "Water"})


def test_latin_square_single_factor_table_9_4() -> None:
    """Книга: Table 9.4 (стр. 215)."""
    d = _lupin()
    trt = [s + w for s, w in zip(d["soil"], d["water"])]
    rows = _rows(latin_square_anova(d["y"], trt, d["row"], d["col"]))
    assert (rows["Row"]["ss"], rows["Row"]["F"]) == pytest.approx((6.162, 3.165), abs=5e-4)
    assert rows["Row"]["p"] == pytest.approx(0.107, abs=5e-4)
    assert (rows["Column"]["ss"], rows["Column"]["F"]) == pytest.approx((74.912, 38.478), abs=5e-4)
    assert (rows["A"]["df"], rows["A"]["ss"]) == (3, pytest.approx(24.662, abs=5e-4))
    assert (rows["A"]["F"], rows["A"]["p"]) == pytest.approx((12.667, 0.005), abs=5e-4)
    assert (rows["Residual"]["df"], rows["Residual"]["ss"]) == (6, pytest.approx(3.894, abs=5e-4))


def test_latin_square_factorial_table_9_5(lupin) -> None:
    """Книга: Table 9.5 (стр. 215)."""
    rows = _rows(lupin)
    expected = {"Soil": (5.881, 9.062, 0.024), "Water": (2.176, 3.352, 0.117),
                "Soil.Water": (16.606, 25.588, 0.002)}
    for term, (ss, f, p) in expected.items():
        assert (rows[term]["ss"], rows[term]["F"], rows[term]["p"]) == pytest.approx(
            (ss, f, p), abs=5e-4)
    assert rows["Row"]["stratum"] == "row" and rows["Column"]["stratum"] == "column"
    assert lupin.statistics["ss_total"] == pytest.approx(109.629, abs=5e-4)


def test_latin_square_effects_table_9_2(lupin) -> None:
    """Книга: Table 9.2 (стр. 213)."""
    eff = lupin.parameters["effects"]
    assert list(eff["Row"].values()) == pytest.approx([-0.04, -0.89, 0.86, 0.08], abs=6e-3)
    assert list(eff["Column"].values()) == pytest.approx([-2.64, 1.31, 2.81, -1.47], abs=6e-3)


def test_latin_square_predictions_table_9_6(lupin) -> None:
    """Книга: Table 9.6 (стр. 217)."""
    assert lupin.parameters["prediction_terms"] == ["Soil.Water"]
    table = lupin.parameters["tables"][0]
    means = {r["label"]: r["mean"] for r in table["rows"]}
    assert means["Soil=C × Water=H"] == pytest.approx(21.77, abs=6e-3)
    assert means["Soil=S × Water=L"] == pytest.approx(22.25, abs=6e-3)
    assert table["rows"][0]["se"] == pytest.approx(0.403, abs=5e-4)
    assert (table["sed"], table["lsd"]) == pytest.approx((0.570, 1.394), abs=5e-4)


def test_latin_square_rejects_repeated_treatment_in_row() -> None:
    d = _lupin()
    trt = [s + w for s, w in zip(d["soil"], d["water"])]
    trt[0], trt[1] = trt[1], trt[0]  # ломает «раз в колонке»
    result = latin_square_anova(d["y"], trt, d["row"], d["col"])
    assert result.warnings[0].code == "latin_square_invalid"


# --- Расщеплённые делянки: Tables 9.12, 9.14 ----------------------------------


@pytest.fixture(scope="module")
def competition():
    d = _competition()
    return split_plot_anova(d["y"], d["irr"], d["species"], block=d["block"],
                            column_names={"factor_a": "Irrigation", "factor_b": "Species"})


def test_split_plot_anova_table_9_12(competition) -> None:
    """Книга: Table 9.12 (стр. 225)."""
    rows = _rows(competition)
    expected = {"Block": (3, 6.6473, 1.476, 0.378), "Irrigation": (1, 14.2311, 9.480, 0.054),
                "WholePlot residual": (3, 4.5035, 5.751, 0.006),
                "Species": (3, 85.9257, 109.726, 0.0),
                "Irrigation.Species": (3, 4.3714, 5.582, 0.007)}
    for term, (df, ss, f, p) in expected.items():
        assert rows[term]["df"] == df
        assert rows[term]["ss"] == pytest.approx(ss, abs=5e-5)
        assert (rows[term]["F"], rows[term]["p"]) == pytest.approx((f, p), abs=5e-4)
    assert (rows["Residual"]["df"], rows["Residual"]["ms"]) == (18, pytest.approx(0.2610, abs=5e-5))
    assert rows["Irrigation"]["stratum"] == "whole_plot"
    assert competition.statistics["ss_total"] == pytest.approx(120.3776, abs=5e-5)


def test_split_plot_predictions_table_9_14(competition) -> None:
    """Книга: Table 9.14 (стр. 227)."""
    table = competition.parameters["tables"][0]
    means = {r["label"]: r["mean"] for r in table["rows"]}
    assert means["Irrigation=no × Species=-"] == pytest.approx(8.117, abs=6e-4)
    assert means["Irrigation=yes × Species=Ga"] == pytest.approx(4.075, abs=6e-4)
    assert table["rows"][0]["se"] == pytest.approx(0.3778, abs=5e-5)
    assert table["se_df"] == pytest.approx(6.64, abs=5e-3)
    comps = {(c["group1"], c["group2"]): c for c in table["comparisons"]}
    within = comps[("Irrigation=no × Species=-", "Irrigation=no × Species=Am")]
    across = comps[("Irrigation=no × Species=Ga", "Irrigation=yes × Species=Ga")]
    assert (within["sed"], within["df"]) == (pytest.approx(0.3613, abs=5e-5), 18)
    assert (across["sed"], across["df"]) == pytest.approx((0.5344, 6.64), abs=5e-3)
    assert across["mean_diff"] == pytest.approx(-2.605, abs=6e-4)


def test_satterthwaite_example_9_5c() -> None:
    """Книга: Example 9.5C (стр. 227)."""
    assert satterthwaite_df(1.5012, 0.2610, 4, 3, 18) == pytest.approx(6.64, abs=5e-3)


def test_split_plot_whole_plot_column_matches_block_a(competition) -> None:
    d = _competition()
    alt = split_plot_anova(d["y"], d["irr"], d["species"], block=d["block"], whole_plot=d["wp"])
    assert alt.statistics["ms_wholeplot"] == pytest.approx(
        competition.statistics["ms_wholeplot"])


def test_split_plot_without_blocks_example_9_6() -> None:
    """Книга: Example 9.6, Table 9.16 (стр. 229–230).

    Без блоков: целые делянки — колонка ``whole_plot``; df как в Table 9.16.
    """
    rng = np.random.default_rng(2)
    temp = ["20", "20", "15", "15", "15", "20"]
    y, a, b, bed = [], [], [], []
    for i, t in enumerate(temp, 1):
        for sub in ("Chips", "Vermiculite"):
            y.append(10 + rng.normal())
            a.append(t)
            b.append(sub)
            bed.append(str(i))
    rows = _rows(split_plot_anova(y, a, b, whole_plot=bed))
    assert [rows[k]["df"] for k in ("A", "WholePlot residual", "B", "A.B", "Residual")] == [
        1, 4, 1, 1, 4]
    assert "Block" not in rows


def test_split_plot_main_effect_tables_use_stratum_errors() -> None:
    """Книга: Table 9.13 (стр. 225): SE и df по стратам.

    Без взаимодействия: A — по s_w² на df целых делянок, B — по s² на df субделянок.
    """
    d = _competition()
    rng = np.random.default_rng(0)
    wp_noise = {(b, w): rng.normal(scale=0.3) for b, w in zip(d["block"], d["wp"])}
    shift = {"-": 3.0, "Am": 0.0, "Ga": 1.5, "Sm": 2.0}
    y = [5 + 2 * (i == "yes") + shift[s] + wp_noise[(b, w)] + rng.normal(scale=0.2)
         for i, s, b, w in zip(d["irr"], d["species"], d["block"], d["wp"])]
    res = split_plot_anova(y, d["irr"], d["species"], block=d["block"])
    assert res.parameters["prediction_terms"] == ["A", "B"]
    tables = {t["term"]: t for t in res.parameters["tables"]}
    sw2, s2 = res.statistics["ms_wholeplot"], res.statistics["ms_resid"]
    assert tables["A"]["rows"][0]["se"] == pytest.approx(np.sqrt(sw2 / 16))
    assert tables["A"]["se_df"] == 3
    assert tables["A"]["comparisons"][0]["sed"] == pytest.approx(np.sqrt(2 * sw2 / 16))
    assert tables["B"]["rows"][0]["se"] == pytest.approx(np.sqrt(s2 / 8))
    assert tables["B"]["se_df"] == 18


def test_split_plot_requires_units() -> None:
    d = _competition()
    result = split_plot_anova(d["y"], d["irr"], d["species"])
    assert result.warnings[0].code == "split_plot_no_units"


def test_split_plot_rejects_mixed_whole_plot() -> None:
    d = _competition()
    irr = list(d["irr"])
    irr[0] = "no" if irr[0] == "yes" else "yes"
    result = split_plot_anova(d["y"], irr, d["species"], block=d["block"], whole_plot=d["wp"])
    assert result.warnings[0].code == "split_plot_invalid"


# --- BIBD: Table 9.20, Example 9.9B -------------------------------------------


@pytest.mark.parametrize(("design", "ef", "factor", "res_df"), [
    (BIBD_A, 7 / 9, 0.93, 8), (BIBD_B, 7 / 8, 0.76, 15)])
def test_bibd_design_example_9_9b(design, ef, factor, res_df) -> None:
    """Книга: Example 9.9B (стр. 237)."""
    y, trt, block = _bibd(design)
    result = bibd_anova(y, trt, block)
    info = result.parameters["design"]
    assert info["efficiency_factor"] == pytest.approx(ef)
    assert info["lambda"] == (1 if design is BIBD_A else 2)
    assert not info["resolvable_possible"]
    assert result.statistics["df_resid"] == res_df
    s = np.sqrt(result.statistics["ms_resid"])
    assert result.statistics["sed"] / s == pytest.approx(factor, abs=5e-3)
    assert "Block residual" not in _rows(result)  # m = t: нет df
    assert any(w.code == "bibd_no_block_residual" for w in result.warnings)


def test_bibd_matches_intra_block_ols() -> None:
    y, trt, block = _bibd(BIBD_B)
    result = bibd_anova(y, trt, block)
    frame = pd.DataFrame({"y": y, "g": trt, "b": block})
    fit = smf.ols("y ~ C(b) + C(g)", data=frame).fit()
    ref = anova_lm(fit)
    rows = _rows(result)
    assert rows["Treatment"]["ss"] == pytest.approx(ref.loc["C(g)", "sum_sq"])
    assert rows["Residual"]["ss"] == pytest.approx(ref.loc["Residual", "sum_sq"])
    adj = result.parameters["adjusted_means"]
    assert adj["T5"] - adj["T1"] == pytest.approx(fit.params["C(g)[T.T5]"])
    se = np.sqrt(fit.cov_params().loc["C(g)[T.T5]", "C(g)[T.T5]"])
    assert result.statistics["sed"] == pytest.approx(se)


def test_bibd_strata_add_up_and_between_block_effects() -> None:
    y, trt, block = _bibd(BIBD_A)
    result = bibd_anova(y, trt, block)
    rows = _rows(result)
    total = sum(r["ss"] for r in rows.values())
    assert total == pytest.approx(result.statistics["ss_total"])
    info = result.parameters["design"]
    between = result.parameters["between_block_effects"]
    ss = info["n"] * (1 - info["efficiency_factor"]) * sum(v**2 for v in between.values())
    assert rows["Treatment (between blocks)"]["ss"] == pytest.approx(ss)


def test_bibd_with_block_residual() -> None:
    """t = 4, блоки по 2, все 6 пар: m − t = 2 df остатка в страте блоков."""
    design = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] * 2
    y, trt, block = _bibd(design, seed=9)
    rows = _rows(bibd_anova(y, trt, block))
    assert rows["Block residual"]["df"] == 12 - 4
    assert "F" in rows["Treatment (between blocks)"]


def test_bibd_rejects_unbalanced_and_complete() -> None:
    y, trt, block = _bibd(BIBD_A)
    assert bibd_anova(y[:-1], trt[:-1], block[:-1]).warnings[0].code == "bibd_invalid"
    rcbd = [[1, 2, 3]] * 3
    y, trt, block = _bibd(rcbd)
    assert bibd_anova(y, trt, block).warnings[0].code == "bibd_complete_blocks"


def test_design_results_serialisable(lupin, competition) -> None:
    y, trt, block = _bibd(BIBD_A)
    for result in (lupin, competition, bibd_anova(y, trt, block)):
        json.dumps(result.to_dict(), allow_nan=False)
