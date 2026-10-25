"""Тесты факторного и вложенного ANOVA (гл. 8.1–8.5).

Эталоны — Tables 8.2a–c, 8.4, 8.5, 8.8, 8.9, 8.11, 8.12, 8.14, 8.16
(Welham et al., 2015), сверены с картинками страниц; плюс сверка со statsmodels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from core.report.schema import Severity
from core.stats.factorial import _select_prediction_terms, factorial_anova, nested_anova
from tests.test_blocking import LAYOUT, PLOT_YIELD

# Table 6.1 (стр. 117): яйца жуков, Species × MateType, по 10 самок.
BEETLES = {
    ("vit", "inter"): [57, 15, 40, 34, 42, 19, 43, 39, 36, 24],
    ("vit", "intra"): [90, 80, 101, 59, 73, 51, 43, 57, 42, 66],
    ("vulg", "inter"): [82, 91, 66, 98, 82, 134, 51, 96, 52, 91],
    ("vulg", "intra"): [136, 117, 181, 41, 89, 106, 133, 98, 106, 79],
}

# Table 8.7 (стр. 168): (ladybird, cadaver) × [trefoil R1, R2, bean R1, R2] = (infected, live).
_LB_ROWS = [
    ("+", 5, [(1, 15), (2, 18), (5, 18), (5, 15)]),
    ("+", 5, [(1, 13), (1, 13), (3, 20), (3, 17)]),
    ("+", 5, [(2, 16), (1, 15), (7, 17), (2, 18)]),
    ("+", 10, [(2, 12), (2, 17), (10, 17), (8, 19)]),
    ("+", 10, [(2, 16), (1, 18), (3, 15), (7, 15)]),
    ("+", 10, [(3, 15), (3, 16), (2, 14), (6, 16)]),
    ("+", 20, [(7, 14), (9, 18), (9, 19), (11, 16)]),
    ("+", 20, [(8, 17), (6, 19), (6, 18), (9, 17)]),
    ("+", 20, [(7, 16), (7, 15), (12, 19), (11, 14)]),
    ("-", 5, [(1, 20), (0, 20), (1, 20), (1, 20)]),
    ("-", 5, [(1, 20), (0, 20), (6, 20), (2, 20)]),
    ("-", 5, [(2, 20), (1, 20), (7, 20), (2, 20)]),
    ("-", 10, [(2, 20), (0, 20), (3, 20), (7, 20)]),
    ("-", 10, [(1, 20), (2, 20), (4, 20), (5, 20)]),
    ("-", 10, [(2, 20), (2, 20), (5, 20), (5, 20)]),
    ("-", 20, [(2, 20), (3, 20), (4, 20), (9, 20)]),
    ("-", 20, [(3, 20), (2, 20), (8, 20), (8, 20)]),
    ("-", 20, [(3, 20), (2, 20), (5, 20), (5, 20)]),
]
_LB_COLS = [("trefoil", "1"), ("trefoil", "2"), ("bean", "1"), ("bean", "2")]

# Table 8.10 (стр. 175): по репликам «изолят устойчивых»; проростков в лотке 23, 22, 23.
_BRASSICA = {
    "1": "B3 3,A6 14,A4 5,B1 2,A7 6,B2 2,A1 3,A9 1,A5 2,A2 3,A8 15,A3 4,B4 2",
    "2": "A3 2,A5 2,A8 8,A9 1,A4 1,A1 3,B2 2,B4 0,A7 4,B1 1,A2 1,B3 2,A6 9",
    "3": "A3 3,A7 5,A2 3,A5 2,B1 1,A8 16,A6 15,A1 4,B3 4,A9 0,B4 4,A4 4,B2 1",
}
_SEEDLINGS = {"1": 23, "2": 22, "3": 23}


def _beetles() -> tuple[list[float], list[str], list[str]]:
    y, sp, mt = [], [], []
    for (s, m), v in BEETLES.items():
        y += v
        sp += [s] * len(v)
        mt += [m] * len(v)
    return y, sp, mt


def _ladybird() -> pd.DataFrame:
    """Процент заражённых (inf + 1)/(live + 2)·100 (стр. 168)."""
    rows = [(run, host, str(cad), lb, 100 * (inf + 1) / (live + 2))
            for lb, cad, cells in _LB_ROWS
            for (host, run), (inf, live) in zip(_LB_COLS, cells)]
    return pd.DataFrame(rows, columns=["run", "host", "cadaver", "ladybird", "pct"])


def _brassica() -> pd.DataFrame:
    """Процент устойчивых (res + 1)/(seedlings + 2)·100 (стр. 175)."""
    rows = []
    for rep, cells in _BRASSICA.items():
        for cell in cells.split(","):
            iso, res = cell.split()
            pct = 100 * (int(res) + 1) / (_SEEDLINGS[rep] + 2)
            rows.append((rep, iso[0], iso[1:], pct))
    return pd.DataFrame(rows, columns=["rep", "species", "isolate", "pct"])


def _potato() -> tuple[list[float], list[str], list[str], list[str]]:
    y, typ, trt, blk = [], [], [], []
    for b, layout in LAYOUT.items():
        for t, v in zip(layout, PLOT_YIELD[b]):
            y.append(v)
            typ.append("Control" if t == "Control" else "Fungicide")
            trt.append(t)
            blk.append(str(b))
    return y, typ, trt, blk


def _rows(result) -> dict[str, dict]:
    return {r["term"]: r for r in result.parameters["anova_table"]}


def _table(result, term: str) -> dict:
    return next(t for t in result.parameters["tables"] if t["term"] == term)


def _means(table: dict, key: str = "mean") -> dict[str, float]:
    return {r["label"]: r[key] for r in table["rows"]}


# --- Beetle mating: Tables 8.2, 8.4, 8.5 ---------------------------------------


@pytest.fixture(scope="module")
def beetle():
    y, sp, mt = _beetles()
    return factorial_anova(y, sp, mt, transform_kind="log10",
                           column_names={"factor_a": "Species", "factor_b": "MateType"})


def test_beetle_anova_table_8_4(beetle) -> None:
    rows = _rows(beetle)
    assert rows["Species"]["ss"] == pytest.approx(0.90307, abs=5e-5)
    assert rows["MateType"]["ss"] == pytest.approx(0.38073, abs=5e-5)
    assert rows["Species.MateType"]["ss"] == pytest.approx(0.09135, abs=5e-5)
    assert rows["Species"]["F"] == pytest.approx(37.932, abs=5e-4)
    assert rows["MateType"]["F"] == pytest.approx(15.992, abs=5e-4)
    assert rows["Species.MateType"]["F"] == pytest.approx(3.837, abs=5e-4)
    assert rows["Species.MateType"]["p"] == pytest.approx(0.058, abs=5e-4)
    assert rows["Residual"]["df"] == 36
    assert rows["Residual"]["ms"] == pytest.approx(0.0238, abs=5e-5)
    assert beetle.statistics["ss_total"] == pytest.approx(2.2322, abs=5e-5)
    shares = sum(r["ss_share"] for r in rows.values() if "ss_share" in r)
    assert shares == pytest.approx(1.0)


def test_beetle_effects_table_8_2(beetle) -> None:
    eff = beetle.parameters["effects"]
    assert eff["Species"]["vit"] == pytest.approx(-0.1503, abs=5e-5)
    assert eff["MateType"]["intra"] == pytest.approx(0.0976, abs=5e-5)
    assert eff["Species.MateType"]["vit × inter"] == pytest.approx(-0.0478, abs=5e-5)
    assert eff["Species.MateType"]["vulg × inter"] == pytest.approx(0.0478, abs=5e-5)


def test_beetle_prediction_table_8_5(beetle) -> None:
    """α = 0.05: взаимодействие исключено, предсказание по главным эффектам."""
    assert beetle.parameters["prediction_terms"] == ["Species", "MateType"]
    species = _table(beetle, "Species")
    assert _means(species) == pytest.approx(
        {"Species=vit": 1.6582, "Species=vulg": 1.9587}, abs=5e-5)
    assert species["lsd"] == pytest.approx(0.0990, abs=5e-5)
    assert species["t_crit"] == pytest.approx(2.028, abs=5e-4)
    comp = species["comparisons"][0]
    assert comp["mean_diff"] == pytest.approx(0.301, abs=5e-4)
    assert (comp["ci_lower"], comp["ci_upper"]) == pytest.approx((0.202, 0.399), abs=5e-4)
    # В книге верх 2.52, но 10^0.399 = 2.506: опечатка (стр. 162, сверено по картинке).
    assert (comp["ratio_ci_lower"], comp["ratio_ci_upper"]) == pytest.approx((1.59, 2.51), abs=5e-3)
    mate = _table(beetle, "MateType")["comparisons"][0]
    assert (mate["ratio_ci_lower"], mate["ratio_ci_upper"]) == pytest.approx((1.25, 1.97), abs=5e-3)


def test_beetle_borderline_interaction_flagged(beetle) -> None:
    flag = next(w for w in beetle.warnings if w.code == "interaction_borderline")
    assert flag.severity == Severity.INFO
    assert flag.params == {"term": "Species.MateType", "p": 0.058}


def test_beetle_interaction_kept_at_alpha_010() -> None:
    y, sp, mt = _beetles()
    result = factorial_anova(y, sp, mt, transform_kind="log10", alpha=0.10)
    assert result.parameters["prediction_terms"] == ["A.B"]
    cells = _means(_table(result, "A.B"))
    assert cells["A=vit × B=inter"] == pytest.approx(1.5129, abs=5e-5)
    assert cells["A=vulg × B=intra"] == pytest.approx(2.0085, abs=5e-5)


# --- Ladybird: Tables 8.8, 8.9 (три фактора + блоки-прогоны) -------------------


@pytest.fixture(scope="module")
def ladybird():
    lb = _ladybird()
    return factorial_anova(
        lb["pct"].tolist(), lb["host"].tolist(), lb["cadaver"].tolist(),
        lb["ladybird"].tolist(), block=lb["run"].tolist(), transform_kind="logit_pct",
        column_names={"factor_a": "Host", "factor_b": "Cadaver", "factor_c": "Ladybird"})


def test_ladybird_multistratum_table_8_8(ladybird) -> None:
    rows = _rows(ladybird)
    expected = {"Block": (1, 0.0677, 0.294), "Host": (1, 13.5992, 59.172),
                "Cadaver": (2, 17.0274, 37.044), "Ladybird": (1, 11.0907, 48.257),
                "Host.Cadaver": (2, 0.3078, 0.670), "Host.Ladybird": (1, 0.2279, 0.992),
                "Cadaver.Ladybird": (2, 1.7349, 3.774),
                "Host.Cadaver.Ladybird": (2, 0.1999, 0.435)}
    for term, (df, ss, f) in expected.items():
        assert rows[term]["df"] == df
        assert rows[term]["ss"] == pytest.approx(ss, abs=5e-5)
        assert rows[term]["F"] == pytest.approx(f, abs=5e-4)
    assert rows["Block"]["stratum"] == "block"
    assert rows["Residual"]["df"] == 59
    assert rows["Residual"]["ss"] == pytest.approx(13.5596, abs=5e-5)
    assert ladybird.statistics["ss_total"] == pytest.approx(57.8151, abs=5e-5)


def test_ladybird_predictions_table_8_9(ladybird) -> None:
    assert ladybird.parameters["prediction_terms"] == ["Host", "Cadaver.Ladybird"]
    host = _table(ladybird, "Host")
    assert _means(host) == pytest.approx({"Host=bean": -0.772, "Host=trefoil": -1.641}, abs=5e-4)
    assert _means(host, "back_mean") == pytest.approx(
        {"Host=bean": 31.6, "Host=trefoil": 16.2}, abs=0.05)
    assert host["sed"] == pytest.approx(0.1130, abs=5e-5)
    cl = _table(ladybird, "Cadaver.Ladybird")
    assert cl["sed"] == pytest.approx(0.1957, abs=5e-5)
    expected = {"5 × +": (-1.454, 18.9), "10 × +": (-1.033, 26.3), "20 × +": (0.044, 51.1),
                "5 × -": (-2.038, 11.5), "10 × -": (-1.580, 17.1), "20 × -": (-1.179, 23.5)}
    for key, (logit, pct) in expected.items():
        cad, lb = key.split(" × ")
        row = next(r for r in cl["rows"] if r["label"] == f"Cadaver={cad} × Ladybird={lb}")
        assert row["mean"] == pytest.approx(logit, abs=5e-4)
        assert row["back_mean"] == pytest.approx(pct, abs=0.05)
    assert not any(w.code == "interaction_borderline" for w in ladybird.warnings)


def test_factorial_matches_statsmodels(ladybird) -> None:
    lb = _ladybird()
    lb["y"] = np.log(lb["pct"] / (100 - lb["pct"]))
    ref = anova_lm(smf.ols("y ~ C(run) + C(host)*C(cadaver)*C(ladybird)", data=lb).fit())
    assert _rows(ladybird)["Cadaver.Ladybird"]["ss"] == pytest.approx(
        ref.loc["C(cadaver):C(ladybird)", "sum_sq"])


# --- Отбор членов по маргинальности ------------------------------------------


def test_marginality_keeps_main_effects_of_significant_interaction() -> None:
    terms = [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]
    p = dict.fromkeys(terms, 0.5) | {("a", "b"): 0.01}
    assert _select_prediction_terms(terms, p, 0.05) == [("a", "b")]


def test_marginality_tests_main_effect_after_interactions_drop() -> None:
    terms = [("a",), ("b",), ("a", "b")]
    p = {("a",): 0.001, ("b",): 0.3, ("a", "b"): 0.4}
    assert _select_prediction_terms(terms, p, 0.05) == [("a",)]


# --- Guards -------------------------------------------------------------------


def test_unbalanced_factorial_is_rejected() -> None:
    y, sp, mt = _beetles()
    result = factorial_anova(y[1:], sp[1:], mt[1:])
    assert [(w.code, w.severity) for w in result.warnings] == [
        ("factorial_unbalanced", Severity.ERROR)]


def test_missing_cell_is_rejected() -> None:
    y, sp, mt = _beetles()
    keep = [i for i in range(len(y)) if (sp[i], mt[i]) != ("vulg", "intra")]
    result = factorial_anova([y[i] for i in keep], [sp[i] for i in keep], [mt[i] for i in keep])
    assert result.warnings[0].code == "factorial_unbalanced"


def test_no_replication_is_rejected() -> None:
    result = factorial_anova([1.0, 2.0, 3.0, 5.0], ["a", "a", "b", "b"], ["x", "y", "x", "y"])
    assert result.warnings[0].code == "n_too_small"


def test_single_level_factor_is_rejected() -> None:
    result = factorial_anova([1.0, 2.0, 3.0, 4.0], ["a"] * 4, ["x", "y", "x", "y"])
    assert result.warnings[0].code == "factor_one_level"


def test_result_is_json_serialisable(ladybird) -> None:
    json.dumps(ladybird.to_dict(), allow_nan=False)


# --- Brassica: Tables 8.11, 8.12, 8.14 (вложенная структура) ------------------


@pytest.fixture(scope="module")
def brassica():
    br = _brassica()
    return nested_anova(br["pct"].tolist(), br["species"].tolist(), br["isolate"].tolist(),
                        block=br["rep"].tolist(), transform_kind="logit_pct",
                        column_names={"group": "Species", "member": "Isolate"})


def test_brassica_nested_anova_table_8_12(brassica) -> None:
    rows = _rows(brassica)
    assert (rows["Block"]["df"], rows["Block"]["F"]) == (2, pytest.approx(6.492, abs=5e-4))
    assert rows["Species"]["ss"] == pytest.approx(4.2896, abs=5e-5)
    assert rows["Species"]["F"] == pytest.approx(29.841, abs=5e-4)
    assert rows["Species.Isolate"]["df"] == 11
    assert rows["Species.Isolate"]["F"] == pytest.approx(15.219, abs=5e-4)
    assert rows["Residual"]["df"] == 24
    assert rows["Residual"]["ss"] == pytest.approx(3.4500, abs=5e-5)
    assert brassica.statistics["ss_total"] == pytest.approx(33.6715, abs=5e-5)


def test_brassica_within_group_partition_table_8_14(brassica) -> None:
    rows = _rows(brassica)
    a, b = rows["Species.Isolate[A]"], rows["Species.Isolate[B]"]
    assert (a["df"], b["df"]) == (8, 3)
    assert a["ss"] == pytest.approx(23.4012, abs=5e-5)
    assert b["ss"] == pytest.approx(0.6643, abs=5e-5)
    assert a["F"] == pytest.approx(20.349, abs=5e-4)
    assert (b["F"], b["p"]) == pytest.approx((1.541, 0.230), abs=5e-4)
    assert a["ss"] + b["ss"] == pytest.approx(rows["Species.Isolate"]["ss"])


def test_brassica_effects_table_8_11(brassica) -> None:
    eff = brassica.parameters["effects"]
    assert eff["Species"]["A"] == pytest.approx(0.221, abs=5e-4)
    # −0.498 в книге — среднее округлённых эффектов (−1.990 / 4 = −0.4975).
    assert eff["Species"]["B"] == pytest.approx(-0.498, abs=1e-3)
    nested = eff["Species.Isolate"]
    # Вложенные эффекты книги расходятся с точными в 3-м знаке (±0.001, округление).
    assert nested["A/6"] == pytest.approx(1.555, abs=1.5e-3)
    assert nested["B/3"] == pytest.approx(0.397, abs=1.5e-3)
    assert nested["A/1"] == pytest.approx(-0.210, abs=5e-4)


def test_brassica_group_means(brassica) -> None:
    """Средние видов −1.341 (20.7%) и −2.060 (11.3%), SED = 0.1315 (стр. 176)."""
    species = _table(brassica, "Species")
    assert _means(species) == pytest.approx({"Species=A": -1.341, "Species=B": -2.060}, abs=5e-4)
    assert [e.value for e in brassica.estimates] == pytest.approx([20.7, 11.3], abs=0.05)
    assert species["comparisons"][0]["sed"] == pytest.approx(0.1315, abs=5e-5)
    assert "sed" not in species  # повторность групп разная: SED только попарно


# --- Potato: Table 8.16 (контроль против остальных) ---------------------------


def test_potato_control_vs_treated_table_8_16() -> None:
    y, typ, trt, blk = _potato()
    result = nested_anova(y, typ, trt, block=blk,
                          column_names={"group": "Type", "member": "Fungicide"})
    rows = _rows(result)
    assert rows["Block"]["ss"] == pytest.approx(14987.20, abs=5e-3)
    assert rows["Block"]["F"] == pytest.approx(1.434, abs=5e-4)
    assert rows["Block"]["p"] == pytest.approx(0.2814, abs=5e-5)  # в книге опечатка 0.283
    assert rows["Type"]["F"] == pytest.approx(35.972, abs=5e-4)
    assert rows["Type.Fungicide"]["df"] == 3
    assert rows["Type.Fungicide"]["F"] == pytest.approx(0.778, abs=5e-4)
    assert rows["Type.Fungicide"]["p"] == pytest.approx(0.529, abs=5e-4)
    assert rows["Residual"]["ms"] == pytest.approx(3483.07, abs=5e-3)
    assert not any(t.endswith("]") for t in rows)  # у контроля один член — разбиения нет
    eff = result.parameters["effects"]
    assert eff["Type"]["Control"] == pytest.approx(-158.3, abs=0.05)
    assert eff["Type.Fungicide"]["Fungicide/F1"] == pytest.approx(-34.9, abs=0.05)
    assert eff["Type.Fungicide"]["Control/Control"] == 0.0


def test_nested_without_members_is_rejected() -> None:
    result = nested_anova([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"], ["1", "1", "1", "1"])
    assert result.warnings[0].code == "nested_no_members"


def test_nested_unequal_replication_is_rejected() -> None:
    y, typ, trt, blk = _potato()
    result = nested_anova(y[:-1], typ[:-1], trt[:-1], block=blk[:-1])
    assert result.warnings[0].code == "factorial_unbalanced"


def test_numeric_levels_in_natural_order(ladybird) -> None:
    labels = [r["label"] for r in _table(ladybird, "Cadaver.Ladybird")["rows"]]
    assert [lab.split(" × ")[0] for lab in labels[::2]] == ["Cadaver=5", "Cadaver=10", "Cadaver=20"]


def test_nested_without_residual_variation_does_not_crash() -> None:
    result = nested_anova([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], ["a", "a", "a", "a", "b", "b"],
                          ["1", "1", "2", "2", "1", "1"])
    assert [w.code for w in result.warnings] == ["no_residual_variation"]
    assert "tables" not in result.parameters
