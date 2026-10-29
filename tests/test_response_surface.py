"""Глава 17.2: поверхности отклика, хлопок (Table 17.7; Tables 17.8–17.11)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.stats.response_surface import eligible, response_surface, surface_terms
from tests.book_ch17_data import COTTON

_COLUMNS = {"H": [r[0] for r in COTTON], "I": [r[1] for r in COTTON],
            "Weight": [r[2] for r in COTTON]}


def _cotton(order, centre="mean", selection="none", **kw):
    """Средние Table 17.7 с ResMS 174 на 75 df (стр. 464), как в книге."""
    return response_surface(_COLUMNS["Weight"], _COLUMNS, "Weight", "H, I", order, centre,
                            selection, 174, 75, **kw)


def _rows(res):
    return {r["term"]: r for r in res.parameters["sequential"]}


def test_interaction_model_table_17_8():
    """Книга: Table 17.8 (стр. 465): SS 573, 5, 3102, Remainder 3295 (21 df), Total 20 025 (99 df)."""
    rows = _rows(_cotton("interaction", centre="none"))
    assert [rows[t]["ss"] for t in ("+ H", "+ I", "+ H.I", "Remainder")] == pytest.approx(
        [573, 5, 3102, 3295], abs=0.5)
    assert [rows[t]["F"] for t in ("+ H", "+ I", "+ H.I", "Remainder")] == pytest.approx(
        [3.29, 0.03, 17.83, 0.90], abs=5e-3)
    assert (rows["Remainder"]["df"], rows["Remainder"]["p"]) == (21, pytest.approx(0.589, abs=1e-3))
    assert (rows["Total"]["df"], rows["Total"]["ss"]) == (99, pytest.approx(20025, abs=0.5))


def test_interaction_model_parameters_table_17_9():
    """Книга: Table 17.9 (стр. 466): 99.350, −29.050, −0.573, 0.557; SE 7.915, 6.4622, 0.1616, 0.1319."""
    res = _cotton("interaction", centre="none")
    est = [(p["estimate"], p["se"]) for p in res.parameters["parameter_tests"]]
    assert [e for e, _ in est] == pytest.approx([99.350, -29.050, -0.573, 0.557], abs=5e-4)
    assert [s for _, s in est] == pytest.approx([7.915, 6.4622, 0.1616, 0.1319], abs=5e-4)
    assert res.parameters["parameter_tests"][3]["t"] == pytest.approx(4.223, abs=1e-3)


def test_interaction_model_summary_example_17_3a():
    """Книга: Example 17.3A (стр. 465): R²adj = 0.158, 46.0% вариации вариантов, ModSS 3680 (стр. 470).

    F модели в книге 6.06 (стр. 465), но по её же числам ModMS/ResMS = (3680/3)/174 = 7.05;
    6.06 ≈ ModMS/TotMS (202.3) — расхождение книги; P < 0.001 в обоих случаях.
    """
    st = _cotton("interaction", centre="none").statistics
    assert st["F_model"] == pytest.approx(3680 / 3 / 174, abs=5e-3)
    assert st["p_model"] < 0.001
    assert st["adj_r_squared"] == pytest.approx(0.158, abs=5e-4)
    assert st["treatment_variation_explained"] == pytest.approx(0.460, abs=5e-4)
    assert st["model_ss"] == pytest.approx(3680, abs=0.5)


def test_second_order_example_17_3b():
    """Книга: Example 17.3B (стр. 468): второй порядок — 17.5% всей и 52.3% вариации вариантов."""
    st = _cotton("quadratic").statistics
    assert st["adj_r_squared"] == pytest.approx(0.175, abs=5e-4)
    assert st["treatment_variation_explained"] == pytest.approx(0.523, abs=5e-4)


def test_third_order_sequential_table_17_10():
    """Книга: Table 17.10 (стр. 469): SS девяти членов, Remainder 1049 на 15 df, F и P."""
    rows = list(_rows(_cotton("cubic")).values())
    assert [r["ss"] for r in rows[:10]] == pytest.approx(
        [573, 5, 109, 3102, 553, 5, 237, 1180, 163, 1049], abs=0.6)
    assert [r["F"] for r in rows[:10]] == pytest.approx(
        [3.29, 0.03, 0.63, 17.83, 3.18, 0.03, 1.36, 6.78, 0.94, 0.40], abs=5e-3)
    assert [r["p"] for r in rows[5:10]] == pytest.approx([0.871, 0.247, 0.011, 0.336, 0.975],
                                                         abs=1e-3)
    assert rows[9]["df"] == 15


def test_backward_selection_example_17_3b():
    """Книга: Example 17.3B (стр. 469): исключаются (cH)³, (cI)³, (cH)².cI, (cH)²; cH.(cI)² остаётся."""
    res = _cotton("cubic", selection="backward")
    steps = res.parameters["selection_steps"]
    assert [s["term"] for s in steps if s["dropped"]] == ["cH^3", "cI^3", "cH^2.cI", "cH^2"]
    assert [s["p"] for s in steps[:4]] == pytest.approx([0.871, 0.336, 0.247, 0.432], abs=1e-3)
    assert res.parameters["selected_terms"] == ["cH", "cI", "cH.cI", "cI^2", "cH.cI^2"]


def test_final_model_parameters_table_17_11():
    """Книга: Table 17.11 (стр. 470): 75.29, 4.844, −0.016, 0.557, −0.0070, −0.01452 с SE."""
    res = _cotton("cubic", selection="backward")
    est = [(p["estimate"], p["se"], p["t"]) for p in res.parameters["parameter_tests"]]
    assert [e[0] for e in est] == pytest.approx(
        [75.29, 4.844, -0.016, 0.557, -0.0070, -0.01452], abs=5e-3)
    assert [e[1] for e in est] == pytest.approx(
        [4.111, 5.814, 0.0933, 0.1319, 0.00394, 0.005574], rel=1e-3)
    assert [e[2] for e in est] == pytest.approx([18.313, 0.833, -0.169, 4.223, -1.783, -2.604],
                                                abs=1e-3)


def test_final_model_summary_and_expansion_example_17_3b():
    """Книга: Example 17.3B (стр. 469–470): 23.2% и 71.7%, ModSS 5414; раскрытие
    105.34 − 40.66H − 1.17I + 1.718HI + 0.007I² − 0.0145HI²."""
    res = _cotton("cubic", selection="backward")
    st = res.statistics
    assert (st["adj_r_squared"], st["treatment_variation_explained"]) == pytest.approx(
        (0.232, 0.717), abs=5e-4)
    assert st["model_ss"] == pytest.approx(5414, abs=0.5)
    f = (st["model_ss"] - 3680.0) / 2 / 174
    assert f == pytest.approx(4.98, abs=5e-3)
    raw = [p["estimate"] for p in res.parameters["raw_parameters"]]
    assert raw == pytest.approx([105.34, -40.66, -1.17, 1.718, 0.0075, -0.0145], abs=5e-3)


def test_marginality_rules():
    """Подчлены (стр. 468): у x1².x2 подчлены x1, x1², x2, x1.x2; проверяются только старшие."""
    assert surface_terms("quadratic") == [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    assert eligible([(1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]) == [(1, 2)]
    assert eligible([(1, 0), (0, 1), (2, 0), (1, 1)]) == [(2, 0), (1, 1)]


def test_replicated_data_equal_means_with_replicates():
    """Сырые данные с повторами ≡ средние с replicates = r и их чистой ошибкой.

    Книга (Table 17.8) применяет ResMS 174 к средним без множителя r = 4 (SS вариантов
    6975, а не 4·6975), поэтому её тесты воспроизводятся при replicates = 1.
    """
    rng = np.random.default_rng(3)
    dev = rng.standard_normal((25, 4))
    dev -= dev.mean(axis=1, keepdims=True)
    raw = {"H": np.repeat(_COLUMNS["H"], 4), "I": np.repeat(_COLUMNS["I"], 4),
           "y": (np.repeat(_COLUMNS["Weight"], 4) + 10 * dev.ravel())}
    cols = {k: v.tolist() for k, v in raw.items()}
    from_raw = response_surface(cols["y"], cols, "y", "H, I", "quadratic", selection="none")
    err = from_raw.parameters["error"]
    from_means = _cotton("quadratic", replicates=4)
    means = response_surface(_COLUMNS["Weight"], _COLUMNS, "Weight", "H, I", "quadratic",
                             selection="none", error_ms=err["ms"], error_df=err["df"],
                             replicates=4)
    assert err["df"] == 75
    assert [r["ss"] for r in means.parameters["sequential"]] == pytest.approx(
        [r["ss"] for r in from_raw.parameters["sequential"]])
    assert [p["se"] for p in means.parameters["parameter_tests"]] == pytest.approx(
        [p["se"] for p in from_raw.parameters["parameter_tests"]])
    assert from_means.parameters["sequential"][0]["ss"] == pytest.approx(4 * 572.911, abs=0.01)


def test_surface_needs_pure_error_and_serializes():
    """Без повторов и без заданного ResMS — ошибка; результат с поверхностью сериализуется."""
    res = response_surface(_COLUMNS["Weight"], _COLUMNS, "Weight", "H, I", "quadratic")
    assert any(w.code == "surface_no_pure_error" for w in res.warnings)
    ok = _cotton("cubic", selection="backward")
    assert len(ok.parameters["surface"]["fitted"]) == 21
    json.dumps(ok.to_dict())
    with pytest.raises(ValueError):
        response_surface(_COLUMNS["Weight"], _COLUMNS, "Weight", "H", "quadratic")
