"""Тесты неортогональной ANOVA (гл. 11).

Эталоны — Tables 11.1, 11.3, 11.13, 11.15, 11.17, 11.18, 11.20, 11.23, 11.24 и
Example 11.3 (SE, SED) — сверены с картинками страниц PDF 280–306.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from core.stats.unbalanced import backward_selection, orthogonality, unbalanced_anova
from tests.test_factorial import BEETLES

# Table 11.2 (стр. 260): мать, отец, рост корня (мм).
CROSS = ("F5 M1 76|F3 M2 68|F1 M1 83|F1 M1 83|F4 M1 81|F4 M1 84|F1 M1 85|F3 M2 69|F2 M1 82|"
         "F3 M1 75|F5 M2 77|F2 M2 75|F1 M1 88|F2 M2 78|F5 M1 77|F1 M2 80|F2 M1 79|F2 M2 77|"
         "F1 M2 79|F4 M2 80|F1 M1 84|F4 M1 83|F1 M1 89|F3 M2 70|F3 M1 80|F4 M1 85|F4 M1 86|"
         "F2 M1 81|F5 M1 76|F3 M2 70")

# Table 11.16 (стр. 276): (единица, приготовление, концентрация, поглощение); None — пропуск.
ELISA = [(1, 1, 3, 0.482), (2, 1, 1, 0.783), (3, 2, 1, 1.014), (4, 2, 1, 1.038), (5, 1, 4, 0.092),
         (6, 3, 1, 0.784), (7, 3, 3, 0.327), (8, 2, 2, 0.745), (9, 2, 3, None), (10, 1, 3, 0.431),
         (11, 3, 1, 1.001), (12, 3, 4, 0.048), (13, 1, 4, 0.130), (14, 2, 2, 0.707),
         (15, 1, 1, 0.766), (16, 2, 3, 0.412), (17, 3, 4, 0.056), (18, 2, 4, 0.073),
         (19, 3, 2, 0.808), (20, 3, 2, 0.888), (21, 1, 2, 0.780), (22, 1, 2, 0.759),
         (23, 3, 3, 0.364), (24, 2, 4, 0.070)]

# Table 11.19 (стр. 278): ряды раскладки, в ячейке «доза высота» по колонкам 1–6.
HEIGHTS = ["1 58.4|3 56.8|4 61.8|4 68.2|5 61.6|3 70.3", "3 55.5|4 57.2|6 50.8|6 57.4|1 70.0|5 61.1",
           "6 49.9|2 60.9|6 50.7|1 71.3|5 61.3|4 70.0", "2 56.7|3 54.5|2 61.3|2 66.6|5 65.8|1 69.8"]

# Table 11.21 (стр. 282): блок, тип, размер черенка, урожай.
CUTTINGS = ("1 D M 15.16|1 E L 18.31|1 A M 23.94|1 B S 24.37|1 C L 12.04|2 D L 15.30|"
            "2 A L 19.81|2 E M 16.45|2 B S 18.45|2 C M 17.28|3 B M 24.56|3 A M 24.60|"
            "3 C S 25.11|3 D M 22.90|3 E L 25.71|4 E M 23.84|4 D L 21.30|4 B M 24.51|"
            "4 A S 25.77|4 C M 18.34|5 A S 23.01|5 C L 14.74|5 E M 21.67|5 D S 17.30|5 B M 16.63")


def _rows(table: list[dict]) -> dict[str, dict]:
    return {r["term"]: r for r in table}


def _cross():
    cells = [c.split() for c in CROSS.split("|")]
    return unbalanced_anova([float(c[2]) for c in cells], [c[0] for c in cells],
                            [c[1] for c in cells],
                            column_names={"factor_a": "Female", "factor_b": "Male"})


def _elisa(exclude: tuple[int, ...] = ()):
    data = [e for e in ELISA if e[3] is not None and e[0] not in exclude]
    return unbalanced_anova([e[3] for e in data], [str(e[1]) for e in data],
                            [str(e[2]) for e in data], transform_kind="ln",
                            compare_healy_westmacott="yes",
                            column_names={"factor_a": "Prep", "factor_b": "Conc"})


@pytest.fixture(scope="module")
def cross():
    return _cross()


# --- Ортогональность: Tables 11.1, 11.3 ---------------------------------------


def test_orthogonality_condition_example_11_1b_and_11_2a(cross) -> None:
    """Книга: Examples 11.1B и 11.2A (стр. 259)."""
    beetle = pd.DataFrame([(s, m) for (s, m), v in BEETLES.items() for _ in v],
                          columns=["a", "b"])
    assert orthogonality(beetle, ["a", "b"], {"a": "S", "b": "M"})[0]["orthogonal"]
    info = cross.parameters["orthogonality"][0]
    assert not info["orthogonal"]
    assert info["counts"]["F1"] == {"M1": 6, "M2": 2}
    assert any(w.code == "nonorthogonal_design" for w in cross.warnings)


def test_orthogonal_design_order_invariant() -> None:
    y, a, b = [], [], []
    for (s, m), v in BEETLES.items():
        y += [np.log10(x) for x in v]
        a += [s] * len(v)
        b += [m] * len(v)
    result = unbalanced_anova(y, a, b)
    seq, rev = _rows(result.parameters["anova_table"]), _rows(result.parameters["sequential_reverse"])
    assert seq["A"]["ss"] == pytest.approx(rev["A"]["ss"]) == pytest.approx(0.9031, abs=5e-5)
    assert seq["B"]["ss"] == pytest.approx(0.3807, abs=5e-5)


# --- Root growth: Tables 11.13, 11.15 -----------------------------------------


def test_sequential_tables_11_13(cross) -> None:
    """Книга: Table 11.13 (стр. 268), Example 11.2E."""
    seq = _rows(cross.parameters["anova_table"])
    assert (seq["Female"]["df"], seq["Female"]["ss"]) == (4, pytest.approx(610.62, abs=6e-3))
    assert seq["Male"]["ss"] == pytest.approx(136.48, abs=5e-3)
    assert seq["Female"]["F"] == pytest.approx(41.324, abs=5e-4)
    assert seq["Male"]["F"] == pytest.approx(36.945, abs=5e-4)
    assert seq["Female.Male"]["F"] == pytest.approx(2.821, abs=5e-4)
    assert seq["Female.Male"]["p"] == pytest.approx(0.053, abs=5e-4)
    assert (seq["Residual"]["df"], seq["Residual"]["ss"]) == (20, pytest.approx(73.88, abs=5e-3))
    rev = _rows(cross.parameters["sequential_reverse"])
    assert rev["Male"]["ss"] == pytest.approx(354.08, abs=5e-3)
    assert (rev["Female"]["ss"], rev["Female"]["F"]) == pytest.approx((393.02, 26.597), abs=5e-3)
    # В книге Total df напечатан 28; при N = 30 он равен 29.
    assert sum(r["df"] for r in seq.values()) == 29


def test_type2_sums_of_squares(cross) -> None:
    """Книга: Table 11.13 (стр. 268): SS Female | Male, Male | Female, Female.Male."""
    t2 = _rows(cross.parameters["type2"])
    assert t2["Female"]["ss"] == pytest.approx(393.02, abs=5e-3)  # Female | Male
    assert t2["Male"]["ss"] == pytest.approx(136.48, abs=5e-3)  # Male | Female
    assert t2["Female.Male"]["ss"] == pytest.approx(41.68, abs=5e-3)


def test_selection_drops_interaction_example_11_2f(cross) -> None:
    """Книга: Example 11.2F (стр. 271)."""
    assert cross.parameters["final_terms"] == ["Female", "Male"]
    assert cross.parameters["selection_steps"][0]["dropped"] == "Female.Male"


def test_predictions_table_11_15(cross) -> None:
    """Книга: Table 11.15 (стр. 271)."""
    tables = {t["term"]: t for t in cross.parameters["tables"]}
    cells = {r["label"]: r for r in tables["Female.Male"]["rows"]}
    expected = {"Female=F1 × Male=M1": (85.1, 0.81), "Female=F1 × Male=M2": (80.3, 1.03),
                "Female=F3 × Male=M2": (70.4, 0.94), "Female=F5 × Male=M2": (72.9, 1.29)}
    for label, (mean, se) in expected.items():
        assert cells[label]["mean"] == pytest.approx(mean, abs=0.05)
        assert cells[label]["se"] == pytest.approx(se, abs=6e-3)
    female = {r["label"]: r for r in tables["Female"]["rows"]}
    assert female["Female=F3"]["mean"] == pytest.approx(72.8, abs=0.05)
    assert female["Female=F3"]["se"] == pytest.approx(0.91, abs=6e-3)
    male = {r["label"]: r for r in tables["Male"]["rows"]}
    assert male["Male=M2"]["mean"] == pytest.approx(75.8, abs=0.05)
    assert male["Male=M2"]["se"] == pytest.approx(0.70, abs=6e-3)
    diff = next(c for c in tables["Female"]["comparisons"]
                if (c["group1"], c["group2"]) == ("Female=F1", "Female=F3"))
    assert diff["mean_diff"] == pytest.approx(-9.9, abs=0.05)


def test_predictions_match_statsmodels(cross) -> None:
    cells = [c.split() for c in CROSS.split("|")]
    frame = pd.DataFrame(cells, columns=["f", "m", "y"]).astype({"y": float})
    fit = smf.ols("y ~ C(f) + C(m)", data=frame).fit()
    pred = fit.get_prediction(pd.DataFrame({"f": ["F4"], "m": ["M2"]})).summary_frame()
    row = next(r for r in cross.parameters["tables"][0]["rows"]
               if r["label"] == "Female=F4 × Male=M2")
    assert row["mean"] == pytest.approx(pred["mean"].iloc[0])
    assert row["se"] == pytest.approx(pred["mean_se"].iloc[0])


# --- Willow cuttings: Tables 11.23, 11.24 -------------------------------------


@pytest.fixture(scope="module")
def cuttings():
    cells = [c.split() for c in CUTTINGS.split("|")]
    return unbalanced_anova([float(c[3]) for c in cells], [c[1] for c in cells],
                            block=[c[0] for c in cells], extraneous=[c[2] for c in cells],
                            column_names={"factor_a": "Type", "extraneous": "Size"})


def test_cuttings_sequential_table_11_23(cuttings) -> None:
    """Книга: Table 11.23 (стр. 283)."""
    rows = _rows(cuttings.parameters["anova_table"])
    expected = {"Block": (4, 186.4298, 7.385, 0.009), "Size": (2, 49.0181, 3.883, 0.066),
                "Type": (4, 96.9158, 3.839, 0.050), "Size.Type": (6, 21.0791, 0.557, 0.754)}
    for term, (df, ss, f, p) in expected.items():
        assert rows[term]["df"] == df  # Size.Type: 8 − 2 пустые ячейки = 6
        assert rows[term]["ss"] == pytest.approx(ss, abs=5e-5)
        assert (rows[term]["F"], rows[term]["p"]) == pytest.approx((f, p), abs=5e-4)
    assert rows["Residual"]["ss"] == pytest.approx(50.4896, abs=5e-5)


def test_cuttings_selection_keeps_extraneous(cuttings) -> None:
    """Книга: Example 11.5 (стр. 283).

    Size.Type удалено, Size (посторонний фактор) сохранён при P = 0.066 (стр. 283).
    """
    assert cuttings.parameters["final_terms"] == ["Size", "Type"]
    assert cuttings.statistics["df_resid_final"] == 14


def test_cuttings_size_predictions_table_11_24(cuttings) -> None:
    """Книга: Table 11.24 (стр. 284)."""
    size = next(t for t in cuttings.parameters["tables"] if t["term"] == "Size")
    rows = {r["label"]: r for r in size["rows"]}
    expected = {"Size=S": (22.394, 0.9889), "Size=M": (20.109, 0.6667),
                "Size=L": (19.347, 0.9190)}
    for label, (mean, se) in expected.items():
        assert (rows[label]["mean"], rows[label]["se"]) == pytest.approx((mean, se), abs=6e-4)
    assert max(c["lsd"] for c in size["comparisons"]) == pytest.approx(3.0335, abs=5e-4)


def test_cuttings_type_sed_range_fig_11_6(cuttings) -> None:
    """Книга: Fig. 11.6 (стр. 283).

    Книга: SED 1.449–1.538; минимальный SED (C–D) по расчёту 1.430.
    """
    types = next(t for t in cuttings.parameters["tables"] if t["term"] == "Type")
    assert types["sed_max"] == pytest.approx(1.538, abs=5e-4)
    assert types["sed_min"] == pytest.approx(1.430, abs=5e-4)
    pair = next(c for c in types["comparisons"] if (c["group1"], c["group2"]) == ("Type=A", "Type=B"))
    assert pair["sed"] == pytest.approx(1.449, abs=5e-4)


def test_cuttings_empty_cells_not_estimable() -> None:
    cells = [c.split() for c in CUTTINGS.split("|")]
    result = unbalanced_anova([float(c[3]) for c in cells], [c[2] for c in cells],
                              [c[1] for c in cells], alpha=0.9999)
    table = result.parameters["tables"][0]
    missing = {r["label"] for r in table["rows"] if not r["estimable"]}
    assert missing == {"A=L × B=B", "A=S × B=E"}
    assert any(w.code == "prediction_not_estimable" for w in result.warnings)


# --- Plant heights: Table 11.20 -----------------------------------------------


def test_heights_eliminating_columns_table_11_20() -> None:
    """Книга: Table 11.20 (стр. 279)."""
    plants = [(c + 1, cell.split()) for row in HEIGHTS for c, cell in enumerate(row.split("|"))]
    y, dose, col = [float(p[1][1]) for p in plants], [p[1][0] for p in plants], [
        str(p[0]) for p in plants]
    ignoring = _rows(unbalanced_anova(y, dose).parameters["anova_table"])
    assert (ignoring["A"]["ss"], ignoring["A"]["F"]) == pytest.approx((536.1, 4.03), abs=0.05)
    result = unbalanced_anova(y, dose, block=col)
    rows = _rows(result.parameters["anova_table"])
    assert (rows["Block"]["ss"], rows["Block"]["F"]) == pytest.approx((618.9, 33.49), abs=0.05)
    assert (rows["A"]["ss"], rows["A"]["F"]) == pytest.approx((348.1, 18.83), abs=0.05)
    assert rows["Residual"]["df"] == 13
    table = result.parameters["tables"][0]
    # Книга: «min SED = 1.14» — по расчёту 1.41 (перестановка цифр); max 2.10 совпадает.
    assert (table["sed_min"], table["sed_max"]) == pytest.approx((1.411, 2.104), abs=5e-4)


# --- ELISA: Tables 11.17, 11.18 -----------------------------------------------


def test_elisa_one_missing_table_11_17() -> None:
    """Книга: Table 11.17 (стр. 276)."""
    result = _elisa()
    exact, hw = _rows(result.parameters["anova_table"]), _rows(result.parameters["healy_westmacott"])
    assert (exact["Prep"]["ss"], exact["Prep"]["F"]) == pytest.approx((0.159, 7.21), abs=6e-3)
    assert (exact["Prep.Conc"]["F"], exact["Residual"]["df"]) == (pytest.approx(8.82, abs=5e-3), 11)
    assert (hw["Prep"]["ss"], hw["Prep"]["F"]) == pytest.approx((0.161, 7.30), abs=6e-3)
    assert (hw["Conc"]["F"], hw["Residual"]["df"]) == (pytest.approx(711.90, abs=5e-2), 11)


def test_elisa_six_missing_table_11_18() -> None:
    """Книга: Table 11.18 (стр. 276)."""
    result = _elisa(exclude=(4, 11, 15, 17, 20))
    exact, hw = _rows(result.parameters["anova_table"]), _rows(result.parameters["healy_westmacott"])
    assert (exact["Prep"]["F"], exact["Prep"]["p"]) == pytest.approx((4.52, 0.064), abs=6e-3)
    assert exact["Prep.Conc"]["p"] == pytest.approx(0.037, abs=5e-4)
    assert hw["Prep"]["F"] == pytest.approx(11.69, abs=6e-3)
    assert (hw["Prep.Conc"]["F"], hw["Prep.Conc"]["p"]) == pytest.approx((7.64, 0.013), abs=6e-3)
    assert hw["Residual"]["df"] == 6
    table = result.parameters["tables"][0]
    assert sorted({round(r["se"], 3) for r in table["rows"]}) == [0.079, 0.111]
    assert (table["sed_min"], table["sed_max"]) == pytest.approx((0.111, 0.157), abs=5e-4)
    assert any(w.code == "healy_westmacott_approximate" for w in result.warnings)


def test_healy_westmacott_unavailable_with_block() -> None:
    cells = [c.split() for c in CUTTINGS.split("|")]
    result = unbalanced_anova([float(c[3]) for c in cells], [c[1] for c in cells],
                              block=[c[0] for c in cells], compare_healy_westmacott="yes")
    assert any(w.code == "healy_westmacott_unavailable" for w in result.warnings)


# --- Отбор и границы ----------------------------------------------------------


def test_backward_selection_respects_marginality(cross) -> None:
    """Книга: Правило маргинальности, стр. 270.

    Первый шаг проверяет только взаимодействие (не главные эффекты).
    """
    first = cross.parameters["selection_steps"][0]["tests"]
    assert [t["term"] for t in first] == ["Female.Male"]
    assert callable(backward_selection)


def test_saturated_model_rejected() -> None:
    result = unbalanced_anova([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"], ["x", "y", "x", "y"])
    assert result.warnings[0].code == "n_too_small"


def test_unbalanced_results_serialisable(cross, cuttings) -> None:
    for result in (cross, cuttings, _elisa(exclude=(4, 11, 15, 17, 20))):
        json.dumps(result.to_dict(), allow_nan=False)
