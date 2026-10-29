"""Глава 16: линейные смешанные модели (REML, Вальд, Кенвард–Роджер, предсказания)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from core.stats.mixed import mixed_model
from tests.book_ch16_data import sosr_long
from tests.test_designs import _competition

_NAMES = {"factor_a": "Irrigation", "factor_b": "Species", "stratum_1": "Block",
          "stratum_2": "WholePlot"}


def _weed():
    d = _competition()
    return mixed_model(d["y"], d["irr"], d["block"], factor_b=d["species"], stratum_2=d["wp"],
                       column_names=_NAMES)


def _fse(response: str = "weeds"):
    d = sosr_long()
    return mixed_model(d[response], d["year"], d["farm"], factor_b=d["trt"],
                       stratum_2=d["field"], transform_kind="log10",
                       column_names={"factor_a": "Year", "factor_b": "Treatment",
                                     "stratum_1": "Farm", "stratum_2": "Field"})


def _rows(table, key="term"):
    return {r[key]: r for r in table}


def test_variance_components_table_16_3():
    """Книга: Table 16.3 (стр. 433)."""
    vc = _rows(_weed().parameters["variance_components"])
    assert (vc["Block"]["estimate"], vc["Block"]["se"]) == pytest.approx((0.0893, 0.2732), abs=5e-5)
    assert (vc["Block.WholePlot"]["estimate"], vc["Block.WholePlot"]["se"]) == pytest.approx(
        (0.3100, 0.3072), abs=5e-5)
    assert (vc["Residual"]["estimate"], vc["Residual"]["se"]) == pytest.approx(
        (0.2610, 0.0870), abs=5e-5)


def test_fixed_effects_table_16_1():
    """Книга: Table 16.1 (стр. 430), Example 16.1C: μ̂11 8.117 (SE 0.4063), Irrigation −0.935 (0.5344)."""
    fe = _rows(_weed().parameters["fixed_effects"])
    assert (fe["[1]"]["estimate"], fe["[1]"]["se"]) == pytest.approx((8.1175, 0.4063), abs=5e-5)
    assert (fe["Irrigation yes"]["estimate"], fe["Irrigation yes"]["se"]) == pytest.approx(
        (-0.935, 0.5344), abs=5e-4)
    assert [fe[f"Species {s}"]["estimate"] for s in ("Am", "Ga", "Sm")] == pytest.approx(
        [-4.632, -1.437, -1.522], abs=1e-3)
    assert fe["Species Am"]["se"] == pytest.approx(0.3613, abs=5e-5)
    assert [fe[f"Irrigation yes.Species {s}"]["estimate"] for s in ("Am", "Ga", "Sm")] == \
        pytest.approx([0.160, -1.670, -0.085], abs=5e-4)
    assert fe["Irrigation yes.Species Am"]["se"] == pytest.approx(0.5109, abs=5e-5)


def test_wald_and_kenward_roger_table_16_2():
    """Книга: Table 16.2 (стр. 431); F и ddf совпадают с Table 9.12 (стр. 225)."""
    inc = _rows(_weed().parameters["incremental"])
    irr, spp, ixs = inc["+ Irrigation"], inc["+ Species"], inc["+ Irrigation.Species"]
    assert [irr["wald"], spp["wald"], ixs["wald"]] == pytest.approx([9.480, 329.178, 16.747],
                                                                    abs=1e-3)
    assert irr["p_wald"] == pytest.approx(0.002, abs=5e-4)
    assert spp["p_wald"] == pytest.approx(4.8e-71, rel=0.02)
    assert ixs["p_wald"] == pytest.approx(8.0e-4, abs=5e-5)
    assert [irr["F"], spp["F"], ixs["F"]] == pytest.approx([9.480, 109.726, 5.582], abs=1e-3)
    assert [irr["ddf"], spp["ddf"], ixs["ddf"]] == pytest.approx([3.0, 18.0, 18.0], abs=1e-6)
    assert [irr["p"], ixs["p"]] == pytest.approx([0.054, 0.007], abs=5e-4)
    assert spp["p"] == pytest.approx(9.3e-12, rel=0.02)


def test_covariances_example_16_1f():
    """Книга: Example 16.1F (стр. 432–433): Var 0.6604; Cov в одной целой делянке 0.3994."""
    cov = _rows(_weed().parameters["covariance_structure"], "shared")
    assert cov["то же наблюдение (дисперсия)"]["covariance"] == pytest.approx(0.6604, abs=5e-5)
    assert cov["Block.WholePlot"]["covariance"] == pytest.approx(0.3994, abs=5e-5)
    assert cov["Block"]["covariance"] == pytest.approx(0.0893, abs=5e-5)


def test_stratum_variances_example_16_1h():
    """Книга: Example 16.1H (стр. 435): 2.2158, 1.5012, 0.2610 = ResMS страт Table 9.12."""
    xi = [r["variance"] for r in _weed().parameters["stratum_variances"]]
    assert xi == pytest.approx([2.2158, 1.5012, 0.2610], abs=5e-5)


def test_predictions_and_sed_example_16_1i():
    """Книга: Example 16.1I (стр. 437): SE 0.4063, SED 0.5344 и 0.3613; средние Table 9.14."""
    r = _weed()
    preds = _rows(r.parameters["predictions"], "cell")
    assert all(p["se"] == pytest.approx(0.4063, abs=5e-5) for p in preds.values())
    assert preds["no × -"]["mean"] == pytest.approx(8.1175)
    assert (r.statistics["sed_min"], r.statistics["sed_max"]) == pytest.approx(
        (0.3613, 0.5344), abs=5e-5)


def test_matches_statsmodels_reml_for_single_random_term():
    """Эталон: statsmodels MixedLM (REML) для одного случайного члена Block."""
    d = _competition()
    r = mixed_model(d["y"], d["irr"], d["block"], factor_b=d["species"])
    ref = smf.mixedlm("y ~ C(irr) * C(species)", pd.DataFrame(d), groups=d["block"]).fit(
        reml=True)
    vc = _rows(r.parameters["variance_components"])
    assert vc["s1"]["estimate"] == pytest.approx(float(ref.cov_re.iloc[0, 0]), rel=1e-4)
    assert vc["Residual"]["estimate"] == pytest.approx(ref.scale, rel=1e-4)
    assert r.parameters["fixed_effects"][0]["estimate"] == pytest.approx(ref.fe_params.iloc[0])


def test_fse_variance_components_table_16_5():
    """Книга: Table 16.5 (стр. 440)."""
    for resp, expected in (("weeds", [0.0517, 0.0628, 0.1139]),
                           ("seedbank", [0.0521, 0.0539, 0.0443])):
        vc = [v["estimate"] for v in _fse(resp).parameters["variance_components"]]
        assert vc == pytest.approx(expected, abs=5e-5)


def test_fse_unbalanced_tests_table_16_6():
    """Книга: Table 16.6 (стр. 440): df Year 34.7 / 32.5 — несбалансированная структура."""
    w = _rows(_fse().parameters["incremental"])
    assert (w["+ Year"]["F"], w["+ Year"]["ddf"]) == pytest.approx((1.068, 34.7), abs=0.05)
    assert w["+ Year"]["F"] == pytest.approx(1.068, abs=5e-4)
    assert w["+ Year"]["p"] == pytest.approx(0.355, abs=5e-4)
    assert (w["+ Treatment"]["F"], w["+ Treatment"]["ddf"]) == pytest.approx((5.415, 56.0),
                                                                              abs=5e-4)
    assert w["+ Year.Treatment"]["F"] == pytest.approx(0.048, abs=5e-4)
    s = _rows(_fse("seedbank").parameters["incremental"])
    assert (s["+ Year"]["F"], s["+ Year"]["ddf"]) == pytest.approx((0.728, 32.5), abs=0.05)
    assert [s["+ Treatment"]["p"], s["+ Year.Treatment"]["p"]] == pytest.approx(
        [0.240, 0.216], abs=5e-4)


def test_fse_treatment_difference_example_16_2():
    """Книга: Example 16.2 (стр. 441): C − GM = 0.145 (SE 0.0719) на log10-шкале."""
    means = _rows(_fse().parameters["factor_means"], "factor")["Treatment"]
    (diff,) = means["differences"]
    assert (diff["difference"], diff["sed"]) == pytest.approx((0.145, 0.0719), abs=5e-4)


def test_negative_component_allowed():
    """Книга: стр. 433–435 — отрицательные компоненты допускаются (как в GenStat)."""
    rng = np.random.default_rng(0)
    block = [str(b) for b in range(6) for _ in range(4)]
    trt = ["a", "b", "c", "d"] * 6
    # Внутри блоков значения нарочно разнесены сильнее, чем между блоками.
    y = [10 + (1 if t in "ab" else -1) * 3 * (int(b) % 2 * 2 - 1) + rng.normal(0, 0.1)
         for b, t in zip(block, trt)]
    r = mixed_model(y, trt, block)
    vc = _rows(r.parameters["variance_components"])
    assert vc["s1"]["estimate"] < 0
    assert any(w.code == "negative_variance_component" for w in r.warnings)


def test_guards_and_json():
    d = _competition()
    one = mixed_model(d["y"], ["a"] * len(d["y"]), d["block"])
    assert one.warnings[-1].code == "factor_one_level"
    flat = mixed_model(d["y"], d["irr"], ["b"] * len(d["y"]))
    assert flat.warnings[-1].code == "mixed_no_structure"
    json.dumps(_weed().to_dict())
    json.dumps(_fse().to_dict())


def _fse_covariate(selection: str = "backward"):
    d = sosr_long()
    return mixed_model(d["weeds"], d["year"], d["farm"], factor_b=d["trt"], stratum_2=d["field"],
                       covariate=d["seedbank"], covariate_transform="log10", selection=selection,
                       transform_kind="log10",
                       column_names={"factor_a": "Year", "factor_b": "Treatment",
                                     "stratum_1": "Farm", "stratum_2": "Field",
                                     "covariate": "LogSeedbank"})


def test_backward_selection_table_16_8():
    """Книга: Table 16.8 (стр. 442): маргинальные F с переподгонкой REML."""
    steps = _fse_covariate().parameters["selection_steps"]
    m1 = _rows(steps[0]["tests"])["LogSeedbank.Year.Treatment"]
    assert (m1["F"], m1["ddf"], m1["p"]) == pytest.approx((0.262, 60.3, 0.770), abs=0.05)
    assert m1["F"] == pytest.approx(0.262, abs=5e-4)
    m2 = _rows(steps[1]["tests"])
    assert [m2[t]["F"] for t in ("Year.Treatment", "LogSeedbank.Year", "LogSeedbank.Treatment")] \
        == pytest.approx([0.103, 0.075, 0.081], abs=5e-4)
    assert [m2[t]["ddf"] for t in ("Year.Treatment", "LogSeedbank.Year",
                                   "LogSeedbank.Treatment")] == pytest.approx(
        [61.0, 75.5, 63.7], abs=0.05)
    m3 = _rows(steps[4]["tests"])
    assert steps[4]["terms"] == ["LogSeedbank", "Year", "Treatment"]
    assert (m3["LogSeedbank"]["F"], m3["LogSeedbank"]["ddf"]) == pytest.approx((31.797, 85.3),
                                                                                abs=0.05)
    assert m3["Year"]["F"] == pytest.approx(0.698, abs=1.5e-3)  # по расчёту 0.699
    assert m3["Year"]["ddf"] == pytest.approx(43.5, abs=0.05)
    assert (m3["Treatment"]["F"], m3["Treatment"]["ddf"], m3["Treatment"]["p"]) == \
        pytest.approx((7.969, 57.8, 0.007), abs=0.05)


def test_final_model_table_16_9():
    """Книга: Table 16.9 (стр. 443) и Example 16.2: μ̂1 1.297 (0.2205), β̂ 0.612 (0.1055)."""
    r = _fse_covariate()
    assert r.parameters["selected_terms"] == ["LogSeedbank", "Treatment"]
    vc = [(v["estimate"], v["se"]) for v in r.parameters["variance_components"]]
    assert [v[0] for v in vc] == pytest.approx([0.0040, 0.0494, 0.109], abs=5e-4)
    # SE по матрице средней информации: 0.0225, 0.0305, 0.0205 (книга: 0.023, 0.031, 0.021).
    assert [v[1] for v in vc] == pytest.approx([0.023, 0.031, 0.021], abs=1e-3)
    fe = _rows(r.parameters["fixed_effects"])
    assert (fe["[1]"]["estimate"], fe["[1]"]["se"]) == pytest.approx((1.297, 0.2205), abs=5e-4)
    assert (fe["LogSeedbank"]["estimate"], fe["LogSeedbank"]["se"]) == pytest.approx(
        (0.612, 0.1055), abs=5e-4)
    assert (fe["Treatment GM"]["estimate"], fe["Treatment GM"]["se"]) == pytest.approx(
        (-0.173, 0.0611), abs=5e-4)


def test_variance_accounted_example_16_2():
    """Книга: стр. 444: базовые компоненты 0.0542, 0.0589, 0.1187 (сумма 0.2318) → 30 %.

    В формуле книги знаменатель напечатан 0.2328 вместо 0.2318; точно 29.8 %.
    """
    r = _fse_covariate()
    assert r.parameters["variance_components_null"] == pytest.approx([0.0542, 0.0589, 0.1187],
                                                                     abs=5e-5)
    assert r.statistics["total_variance_null"] == pytest.approx(0.2318, abs=5e-5)
    assert r.statistics["variance_accounted_pct"] == pytest.approx(29.8, abs=0.05)


def test_covariate_prediction_at_mean():
    """Предсказания с ковариатой — при её среднем; разность обработок = эффект (стр. 443)."""
    r = _fse_covariate()
    (diff,) = _rows(r.parameters["factor_means"], "factor")["Treatment"]["differences"]
    assert (diff["difference"], diff["sed"]) == pytest.approx((0.173, 0.0611), abs=5e-4)


def test_selection_none_keeps_full_model_and_bad_option():
    r = _fse_covariate("none")
    assert "selection_steps" not in r.parameters
    assert len(r.parameters["fixed_effects"]) == 12  # [1] + L + 2Y + T + 2LY + LT + 2YT + 2LYT
    with pytest.raises(ValueError):
        _fse_covariate("forward")
