"""Глава 15.3–15.5: регрессия в RCBD и ковариационный анализ (данные Tables 15.11, 15.15)."""

from __future__ import annotations

import json

import pytest

from core.stats.ancova import ancova, blocked_regression
from tests.book_ch15_data import FORAGE, TGW


def _forage():
    return blocked_regression([r[3] for r in FORAGE], [r[2] for r in FORAGE],
                              [str(r[0]) for r in FORAGE],
                              column_names={"x": "N", "block": "Block"})


def _trt():
    return ancova([r[5] for r in TGW], [r[3] for r in TGW], [str(r[4]) for r in TGW],
                  column_names={"group": "Trt", "covariate": "Damage"})


def _factorial():
    return ancova([r[5] for r in TGW], [r[2] for r in TGW], [r[4] for r in TGW],
                  factor_b=[r[1] for r in TGW],
                  column_names={"group": "Variety", "factor_b": "GR", "covariate": "Damage"})


def _rows(table):
    return {r["term"]: r for r in table}


def test_forage_lack_of_fit_table_15_13():
    """Книга: Table 15.13 (стр. 408); SS Block и Residual совпадают с Table 15.12 (стр. 407)."""
    rows = _rows(_forage().parameters["sequential"])
    assert [rows[t]["ss"] for t in ("+ Block", "+ N", "+ FacN", "Residual")] == pytest.approx(
        [2.8385, 5.9283, 0.2150, 1.9359], abs=1e-4)
    assert rows["+ N"]["F"] == pytest.approx(18.374, abs=1e-3)
    assert rows["+ FacN"]["F"] == pytest.approx(0.333, abs=1e-3)
    assert rows["+ FacN"]["p"] == pytest.approx(0.729, abs=5e-4)
    assert rows["Total"]["ss"] == pytest.approx(10.9178, abs=1e-4)


def test_forage_parameters_table_15_14():
    """Книга: Table 15.14 (стр. 408) и усреднённый сдвиг α* = 10.825 (SE 0.2505), стр. 409."""
    r = _forage()
    est = _rows(r.parameters["parameter_tests"])
    assert (est["[1]"]["estimate"], est["[1]"]["se"]) == pytest.approx((10.982, 0.3279), abs=5e-4)
    assert (est["Block 2"]["estimate"], est["Block 2"]["se"]) == pytest.approx(
        (0.345, 0.3667), abs=5e-4)
    assert est["Block 3"]["t"] == pytest.approx(-2.223, abs=5e-4)
    assert (est["N"]["estimate"], est["N"]["se"]) == pytest.approx((0.0090, 0.00191), abs=5e-5)
    assert r.parameters["line"]["intercept"] == pytest.approx(10.825, abs=5e-4)
    assert r.parameters["line"]["intercept_se"] == pytest.approx(0.2505, abs=5e-5)
    assert not any(w.code == "lack_of_fit" for w in r.warnings)


def test_covariate_check_example_15_4a():
    """Книга: Example 15.4A (стр. 412): ANOVA ковариаты F(3, 20) = 2.642, P = 0.077."""
    (row, *_) = _trt().parameters["covariate_check"]
    assert (row["F"], row["p"]) == pytest.approx((2.642, 0.077), abs=5e-4)


def test_separate_lines_and_treatment_test_example_15_4a():
    """Книга: Example 15.4A (стр. 412): Damage.Trt F = 2.710 (P 0.080); Trt F = 28.874.

    Книга пишет F(3, 16) для Trt, но 28.874 получается против ResMS модели параллельных
    линий с 19 df (против 16 df полной модели было бы 36.67).
    """
    r = _trt()
    seq = _rows(r.parameters["sequential"])
    assert (seq["+ Damage.Trt"]["F"], seq["+ Damage.Trt"]["p"]) == pytest.approx(
        (2.710, 0.080), abs=5e-4)
    second = r.parameters["selection"][1]["tests"][0]
    assert second["term"] == "Trt"
    assert (second["F"], second["df_resid"]) == (pytest.approx(28.874, abs=1e-3), 19)
    assert r.parameters["model_type"] == "parallel_lines"


def test_parallel_lines_parameters_table_15_16():
    """Книга: Table 15.16 (стр. 413)."""
    est = _rows(_trt().parameters["parameter_tests"])
    assert (est["[1]"]["estimate"], est["[1]"]["se"]) == pytest.approx((3.926, 0.0954), abs=5e-4)
    assert (est["Damage"]["estimate"], est["Damage"]["se"]) == pytest.approx(
        (-0.0190, 0.00237), abs=5e-5)
    assert [est[f"Trt {g}"]["estimate"] for g in ("+N", "-B", "-N")] == pytest.approx(
        [0.648, -0.026, 0.787], abs=5e-4)
    assert [est[f"Trt {g}"]["t"] for g in ("+N", "-B", "-N")] == pytest.approx(
        [5.188, -0.235, 7.158], abs=5e-4)


def test_adjusted_and_unadjusted_means_table_15_17():
    """Книга: Table 15.17 (стр. 413) при среднем повреждении 30.83 %; LSD 0.2301–0.2615 (стр. 412)."""
    r = _trt()
    adj = {m["group"]: m for m in r.parameters["adjusted_means"]}
    assert [adj[g]["mean"] for g in ("+B", "+N", "-B", "-N")] == pytest.approx(
        [3.339, 3.987, 3.313, 4.126], abs=5e-4)
    assert [adj[g]["se"] for g in ("+B", "+N", "-B", "-N")] == pytest.approx(
        [0.0797, 0.0881, 0.0780, 0.0797], abs=5e-5)
    raw = {m["group"]: m for m in r.parameters["unadjusted_means"]}
    assert [raw[g]["mean"] for g in ("+B", "+N", "-B", "-N")] == pytest.approx(
        [3.482, 3.654, 3.361, 4.268], abs=6e-4)  # 3.3605 и 4.2685 — округление книги
    assert raw["+B"]["se"] == pytest.approx(0.1586, abs=5e-5)
    assert r.statistics["covariate_mean"] == pytest.approx(30.83, abs=5e-3)
    assert (r.statistics["lsd_min"], r.statistics["lsd_max"]) == pytest.approx(
        (0.2301, 0.2615), abs=5e-5)


def test_factorial_covariate_check_example_15_4b():
    """Книга: Example 15.4B (стр. 415): F(1, 20) = 1.865 для GR и Variety, 4.197 для GR.V."""
    rows = _rows(_factorial().parameters["covariate_check"])
    assert rows["+ Variety"]["F"] == pytest.approx(1.865, abs=5e-4)
    assert rows["+ GR"]["F"] == pytest.approx(1.865, abs=5e-4)
    assert (rows["+ Variety.GR"]["F"], rows["+ Variety.GR"]["p"]) == pytest.approx(
        (4.197, 0.054), abs=5e-4)


def test_factorial_selection_table_15_18():
    """Книга: Table 15.18 (стр. 416).

    Модели 1, 2, 4, 5 совпадают с книгой. На шаге 2 книга по смыслу сначала исключает
    Damage.GR (P = 0.160), BioStat — по правилу стр. 414 наименее значимый Variety.GR
    (P = 0.806); итоговая модель одна и та же.
    """
    sel = _factorial().parameters["selection"]
    m1 = _rows(sel[0]["tests"])
    assert m1["Damage.Variety.GR"]["p"] == pytest.approx(0.567, abs=5e-4)
    assert m1["Damage.Variety.GR"]["F"] == pytest.approx(0.341, abs=5e-4)
    m2 = _rows(sel[1]["tests"])
    assert [m2[t]["p"] for t in ("Variety.GR", "Damage.Variety", "Damage.GR")] == pytest.approx(
        [0.806, 0.015, 0.160], abs=5e-4)
    assert m2["Damage.GR"]["F"] == pytest.approx(2.163, abs=5e-4)
    m4 = _rows(sel[3]["tests"])
    assert (m4["GR"]["p"], m4["Damage.Variety"]["p"]) == pytest.approx((0.839, 0.020), abs=5e-4)
    assert m4["GR"]["F"] == pytest.approx(0.043, abs=5e-4)
    assert _rows(sel[4]["tests"])["Damage.Variety"]["p"] == pytest.approx(0.015, abs=5e-4)


def test_factorial_final_lines_example_15_4b():
    """Книга: Example 15.4B (стр. 416): B: 3.744 − 0.0125·Damage, N: 4.795 − 0.0233·Damage."""
    r = _factorial()
    assert r.parameters["selected_terms"] == ["Damage", "Variety", "Damage.Variety"]
    est = _rows(r.parameters["parameter_tests"])
    assert (est["[1]"]["estimate"], est["[1]"]["se"]) == pytest.approx((3.744, 0.1007), abs=5e-4)
    assert (est["Variety N"]["estimate"], est["Variety N"]["se"]) == pytest.approx(
        (1.051, 0.1345), abs=5e-4)
    assert (est["Damage"]["estimate"], est["Damage"]["se"]) == pytest.approx(
        (-0.0125, 0.00344), abs=5e-5)
    assert (est["Damage.Variety N"]["estimate"], est["Damage.Variety N"]["se"]) == \
        pytest.approx((-0.0108, 0.00403), abs=5e-5)
    lines = {g["group"]: g for g in r.parameters["group_lines"]}
    assert (lines["N"]["intercept"], lines["N"]["slope"]) == pytest.approx(
        (4.795, -0.0233), abs=5e-4)
    assert "adjusted_means" not in r.parameters


def test_block_is_structural_and_json():
    rows = [r[5] for r in TGW]
    blocks = [str(i % 3) for i in range(24)]
    r = ancova(rows, [r[3] for r in TGW], [r[4] for r in TGW], block=blocks)
    assert r.parameters["selected_terms"][0] == "Block"
    assert len(r.parameters["adjusted_means"]) == 4
    json.dumps(r.to_dict())
    json.dumps(_forage().to_dict())
    json.dumps(_factorial().to_dict())


def test_guards():
    bad = ancova([1.0, 2, 3, 4], ["a"] * 4, [1, 2, 3, 4])
    assert bad.warnings[-1].code == "factor_one_level"
    br = blocked_regression([1.0, 2, 3, 4], [0, 1, 0, 1], ["a", "a", "b", "b"])
    assert br.warnings[-1].code == "blocked_regression_invalid"


def test_dummy_variates_table_15_20():
    """Книга: Tables 15.19–15.20 (стр. 418–420): фактор как набор фиктивных переменных."""
    import pandas as pd

    from core.stats.ancova import Terms, sequential_rows

    calcium = "DABCDADACBACADCBDCBB"
    length = [47, 58, 80, 49, 49, 52, 45, 74, 70, 68, 58, 72, 79, 48, 74, 72, 38, 71, 74, 85]
    frame = pd.DataFrame({"y": length, "A": list(calcium)})
    model = Terms(frame, ["A"], [])
    assert model.columns(("A",)).shape == (20, 3)  # d2, d3, d4 при first-level-zero
    rows = _rows(sequential_rows(model, [("A",)], {"A": "(d2 + d3 + d4)"}))
    assert rows["+ (d2 + d3 + d4)"]["ss"] == pytest.approx(2462.95, abs=5e-3)
    assert rows["+ (d2 + d3 + d4)"]["F"] == pytest.approx(10.753, abs=5e-4)
    assert rows["Residual"]["ms"] == pytest.approx(76.35, abs=5e-3)
