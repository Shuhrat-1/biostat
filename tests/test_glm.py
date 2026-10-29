"""Глава 18: биномиальные и пуассоновские GLM (Tables 18.1–18.11)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from core.stats.glm import glm_analysis
from tests.book_ch18_data import APHIDS, DEMETHYLATION, LADYBIRD

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _demeth(**kw):
    d = DEMETHYLATION
    return glm_analysis([r[1] for r in d], covariate=[str(r[0]) for r in d],
                        trials=[str(r[2]) for r in d], column_names={"covariate": "Dose"}, **kw)


def _log_demeth(**kw):
    return _demeth(covariate_transform="ln", covariate_offset=0.1, **kw)


def _ladybird(with_prey=True, **kw):
    d = LADYBIRD
    extra = {"factor_b": [r[3] for r in d]} if with_prey else {}
    return glm_analysis([r[4] for r in d], factor_a=[r[2] for r in d], block=[str(r[0]) for r in d],
                        n_trials=6, column_names={"factor_a": "Sex", "factor_b": "Prey",
                                                  "block": "Row"}, **extra, **kw)


def _aphids(**kw):
    return glm_analysis([r[2] for r in APHIDS], factor_a=[str(r[0]) for r in APHIDS],
                        distribution="poisson", column_names={"factor_a": "Field"}, **kw)


def _est(res):
    return [r["estimate"] for r in res.parameters["parameter_tests"]]


def _rows(table):
    return {r["term"]: r for r in table}


def test_dose_model_example_18_1b():
    """Книга: Example 18.1B (стр. 485): η̂ = 2.793 − 7.623·Dose."""
    assert _est(_demeth()) == pytest.approx([2.793, -7.623], abs=5e-4)


def test_log_dose_model_example_18_1b():
    """Книга: Example 18.1B (стр. 485–486): η̂ = −3.188 − 3.148·ln(Dose + 0.1)."""
    assert _est(_log_demeth()) == pytest.approx([-3.188, -3.148], abs=5e-4)


def test_log_dose_anodev_table_18_3():
    """Книга: Table 18.3 (стр. 488): Model 1874.772 (1 df), Residual 26.623 (22), Total 1901.395."""
    res = _log_demeth()
    rows = _rows(res.parameters["anodev"])
    assert [rows[t]["deviance"] for t in ("Model", "Residual", "Total")] == pytest.approx(
        [1874.772, 26.623, 1901.395], abs=5e-4)
    assert rows["Residual"]["mean_deviance"] == pytest.approx(1.210, abs=5e-4)
    assert rows["Model"]["p"] < 0.001 and "F" not in rows["Model"]


def test_no_overdispersion_example_18_1d():
    """Книга: Example 18.1D (стр. 489): 26.62 на 22 df, P = 0.226 — φ не оценивается."""
    res = _log_demeth()
    disp = res.parameters["dispersion"]
    assert disp["p_overdispersion"] == pytest.approx(0.226, abs=5e-4)
    assert not disp["estimated"] and disp["phi"] == 1.0


def test_prediction_at_dose_0_3():
    """Книга: Example 18.1B (стр. 486–487): η̂(0.3) = −0.304, p̂ = 0.425."""
    pred = _log_demeth(x_predict="0.3").parameters["predictions"][0]
    assert pred["eta"] == pytest.approx(-0.304, abs=5e-4)
    assert pred["natural"] == pytest.approx(0.425, abs=5e-4)
    assert pred["natural_lower"] < pred["natural"] < pred["natural_upper"]


def test_ladybird_anodev_table_18_6():
    """Книга: Table 18.6 (стр. 492): 71.635 (17), 69.659 (42), 141.294 (59); F 2.54, P 0.007; φ 1.659."""
    res = _ladybird()
    rows = _rows(res.parameters["anodev"])
    assert [rows[t]["deviance"] for t in ("Model", "Residual", "Total")] == pytest.approx(
        [71.635, 69.659, 141.294], abs=5e-4)
    assert (rows["Model"]["mean_deviance"], rows["Model"]["F"]) == pytest.approx(
        (4.214, 2.54), abs=5e-3)
    assert rows["Model"]["p"] == pytest.approx(0.007, abs=5e-4)
    disp = res.parameters["dispersion"]
    assert disp["estimated"] and disp["phi"] == pytest.approx(1.659, abs=5e-4)
    assert disp["p_overdispersion"] == pytest.approx(0.005, abs=5e-4)


def test_ladybird_sequential_tables_18_7():
    """Книга: Table 18.7a, b (стр. 494): F Row 1.41, Sex 20.18/20.50, Prey 3.10/2.79, S.P 0.16."""
    res = _ladybird()
    a, b = _rows(res.parameters["sequential"]), _rows(res.parameters["sequential_reverse"])
    assert [a[t]["F"] for t in ("+ Row", "+ Sex", "+ Prey", "+ Sex.Prey")] == pytest.approx(
        [1.41, 20.18, 3.10, 0.16], abs=5e-3)
    assert [b[t]["F"] for t in ("+ Prey", "+ Sex", "+ Prey.Sex")] == pytest.approx(
        [2.79, 20.50, 0.16], abs=5e-3)
    assert [a[t]["mean_deviance"] for t in ("+ Row", "+ Sex", "+ Prey")] == pytest.approx(
        [2.34, 33.47, 5.15], abs=5e-3)
    assert (a["+ Prey"]["p"], b["+ Prey"]["p"], a["+ Sex.Prey"]["p"]) == pytest.approx(
        (0.085, 0.103, 0.690), abs=5e-4)


def test_ladybird_final_model_example_18_2b():
    """Книга: Example 18.2B (стр. 494): Row + Sex — ResMDev 1.706 на 44 df, F(Sex) = 19.617."""
    res = _ladybird(with_prey=False)
    assert res.parameters["dispersion"]["phi"] == pytest.approx(1.706, abs=5e-4)
    assert res.statistics["df_resid"] == 44
    assert _rows(res.parameters["sequential"])["+ Sex"]["F"] == pytest.approx(19.617, abs=5e-3)


def test_ladybird_parameters_table_18_8():
    """Книга: Table 18.8 (стр. 497): оценки совпадают; SE GenStat на ~0.3–1.5 % меньше.

    В книге SE рядов с одинаковыми оценками различаются (ряды 9 и 13: 1.1303 и 1.1387),
    хотя информация для них одинакова — признак неполной сходимости весов в GenStat.
    """
    res = _ladybird(with_prey=False)
    rows = res.parameters["parameter_tests"]
    assert [r["estimate"] for r in rows] == pytest.approx(
        [0.386, -0.200, -1.774, -0.408, -0.626, -1.121, -0.408, 0.582, -2.246, -1.417, -1.417,
         -0.626, -2.246, -0.408, -1.774, -1.550], abs=5e-4)
    assert [r["se"] for r in rows] == pytest.approx(
        [0.6013, 0.8259, 1.0117, 0.8354, 0.8496, 0.8983, 0.8356, 0.8171, 1.1303, 0.9420, 0.9414,
         0.8492, 1.1387, 0.8355, 1.0119, 0.3712], rel=0.015)
    assert rows[-1]["term"] == "Sex M" and rows[-1]["statistic"] == pytest.approx(-4.176, abs=0.02)


def test_ladybird_cell_predictions_table_18_9():
    """Книга: Table 18.9 (стр. 499): ряд 1 — 0.386/−1.164 (0.595/0.238), ряд 8 — 0.968/−0.582."""
    cells = {(c["blk"], c["A"]): c for c in _ladybird(with_prey=False).parameters["cells"]}
    assert (cells["1", "F"]["eta"], cells["1", "M"]["eta"]) == pytest.approx((0.386, -1.164),
                                                                              abs=5e-4)
    assert (cells["1", "F"]["natural"], cells["1", "M"]["natural"]) == pytest.approx(
        (0.595, 0.238), abs=5e-4)
    assert (cells["8", "F"]["natural"], cells["8", "M"]["natural"]) == pytest.approx(
        (0.725, 0.359), abs=5e-4)


def test_ladybird_averaged_predictions_example_18_2e():
    """Книга: Example 18.2E (стр. 500): средний ряд на шкале logit и среднее после обратного
    преобразования; SE GenStat чуть меньше (0.2196, 0.3128)."""
    preds = {p["level"]: p for p in _ladybird(with_prey=False).parameters["factor_predictions"]}
    f, m = preds["F"], preds["M"]
    assert (f["eta"], m["eta"]) == pytest.approx((-0.553, -2.103), abs=5e-4)
    assert (f["se"], m["se"]) == pytest.approx((0.2196, 0.3128), rel=0.01)
    assert (f["lower"], f["upper"], m["lower"], m["upper"]) == pytest.approx(
        (-0.996, -0.110, -2.733, -1.473), abs=5e-3)
    assert (f["natural"], f["natural_lower"], f["natural_upper"]) == pytest.approx(
        (0.37, 0.27, 0.47), abs=5e-3)
    assert (m["natural"], m["natural_lower"], m["natural_upper"]) == pytest.approx(
        (0.11, 0.06, 0.19), abs=5e-3)
    assert (f["natural_se"], m["natural_se"]) == pytest.approx((0.051, 0.030), abs=1e-3)
    assert (f["mean_after"], f["mean_after_se"]) == pytest.approx((0.38, 0.044), abs=5e-3)
    assert (m["mean_after"], m["mean_after_se"]) == pytest.approx((0.13, 0.032), abs=5e-3)


def test_aphid_parameters_example_18_3b():
    """Книга: Example 18.3B (стр. 506): 1.526, 1.208, 0.371; μ̂ = 4.6, 15.4, 6.7 = средние полей."""
    res = _aphids()
    assert _est(res) == pytest.approx([1.526, 1.208, 0.371], abs=5e-4)
    means = [p["natural"] for p in res.parameters["factor_predictions"]]
    by_field = [np.mean([r[2] for r in APHIDS if r[0] == f]) for f in (1, 2, 3)]
    assert means == pytest.approx(by_field)


def test_aphid_anodev_table_18_11():
    """Книга: Table 18.11 (стр. 507): 105.456 (2), 191.475 (42), 296.931 (44); φ 4.559, F 11.566."""
    res = _aphids()
    rows = _rows(res.parameters["anodev"])
    assert [rows[t]["deviance"] for t in ("Model", "Residual", "Total")] == pytest.approx(
        [105.456, 191.475, 296.931], abs=5e-4)
    assert rows["Total"]["mean_deviance"] == pytest.approx(6.748, abs=5e-4)
    assert res.parameters["dispersion"]["phi"] == pytest.approx(4.559, abs=5e-4)
    assert rows["Model"]["F"] == pytest.approx(11.566, abs=5e-4)


def test_aphid_ratios_example_18_3c():
    """Книга: Example 18.3C (стр. 507–509): поле 2/1 — 1.208 (SE 0.2929), ДИ (0.617, 1.799),
    отношение 3.35 (1.85, 6.05); поле 3/1 — SE 0.334, P 0.273, 1.45 (0.74, 2.84)."""
    comp = {c["pair"]: c for c in _aphids().parameters["comparisons"]}
    two, three = comp["2 − 1"], comp["3 − 1"]
    assert (two["difference"], two["se"]) == pytest.approx((1.208, 0.2929), abs=5e-4)
    assert (two["lower"], two["upper"]) == pytest.approx((0.617, 1.799), abs=5e-4)
    assert (two["ratio"], two["ratio_lower"], two["ratio_upper"]) == pytest.approx(
        (3.35, 1.85, 6.05), abs=5e-3)
    assert (three["se"], three["p"]) == pytest.approx((0.334, 0.273), abs=5e-4)
    assert (three["ratio"], three["ratio_lower"], three["ratio_upper"]) == pytest.approx(
        (1.45, 0.74, 2.84), abs=5e-3)
    assert two["ratio_kind"] == "ratio"


def test_pearson_dispersion_matches_definition():
    """Оценка Пирсона φ̂ = Σ (y − μ̂)²/Var(μ̂) / ResDF (уравнение 18.4)."""
    res = _aphids(dispersion_method="pearson")
    y = np.array([r[2] for r in APHIDS], float)
    mu = np.repeat([np.mean(y[:15]), np.mean(y[15:30]), np.mean(y[30:])], 15)
    assert res.parameters["dispersion"]["phi"] == pytest.approx(((y - mu) ** 2 / mu).sum() / 42)


def test_fixed_dispersion_warns_when_deviance_large():
    """φ = 1 при большом остаточном отклонении — предупреждение (стр. 496)."""
    res = _aphids(dispersion_method="fixed")
    assert any(w.code == "glm_overdispersion_ignored" for w in res.warnings)
    assert "F" not in _rows(res.parameters["anodev"])["Model"]


def test_binary_and_underdispersion():
    """Бинарные данные — φ не оценивается (стр. 501); φ̂ < 1 не используется (стр. 493)."""
    y = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    x = [str(v) for v in range(12)]
    res = glm_analysis(y, covariate=x, n_trials=1, dispersion_method="deviance")
    assert any(w.code == "glm_binary" for w in res.warnings)
    assert res.parameters["dispersion"]["phi"] == 1.0
    under = glm_analysis([5, 5, 6, 5, 9, 10, 9, 10], factor_a=list("aaaabbbb"),
                         distribution="poisson", dispersion_method="deviance")
    assert under.parameters["dispersion"]["underdispersion"]
    assert under.parameters["dispersion"]["phi"] == 1.0


def test_probit_link_and_guards():
    """Пробит-связь (стр. 501) близка к логиту; без m или с y > m — ошибка; сериализация."""
    probit = _log_demeth(link="probit")
    assert probit.parameters["link"] == "probit"
    logit = _log_demeth()
    p_fit = [c for c in probit.parameters["curve"]["fitted"]]
    l_fit = [c for c in logit.parameters["curve"]["fitted"]]
    assert p_fit == pytest.approx(l_fit, abs=0.05)
    json.dumps(probit.to_dict())
    no_m = glm_analysis([1, 2, 3, 4], covariate=["1", "2", "3", "4"])
    assert any(w.code == "glm_no_trials" for w in no_m.warnings)
    bad = glm_analysis([1, 7, 3, 4], covariate=["1", "2", "3", "4"], n_trials=6)
    assert any(w.code == "glm_bad_response" for w in bad.warnings)
    with pytest.raises(ValueError):
        glm_analysis([1, 2], covariate=["1", "2"], distribution="poisson", link="logit")


def test_curve_on_original_dose_scale():
    """Кривая на исходной шкале дозы (Fig. 18.3a): убывает, ДИ охватывает подгонку."""
    curve = _log_demeth().parameters["curve"]
    assert curve["x"][0] == 0 and curve["x"][-1] == 1.5
    assert curve["fitted"][0] > curve["fitted"][-1]
    assert all(lo <= f <= hi for lo, f, hi in zip(curve["lower"], curve["fitted"], curve["upper"]))
