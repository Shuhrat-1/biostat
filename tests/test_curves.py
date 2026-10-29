"""Глава 17.1: преобразование x, полиномы, тригонометрическая регрессия (Tables 17.1–17.6, 17.12)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.stats.curves import (
    amplitude_phase,
    polynomial_regression,
    trigonometric_regression,
    x_transform_regression,
)
from tests.book_ch17_data import OLSEN_P, TEMPERATURE, YIELD


def _log_model(**kw):
    return x_transform_regression(OLSEN_P, YIELD, "log10", **kw)


def _params(res, key="parameter_tests"):
    return [(r["estimate"], r["se"], r["t"]) for r in res.parameters[key]]


def test_log_olsen_p_table_17_2():
    """Книга: Table 17.2 (стр. 454): 1.674 + 2.644·LogOP, SE 0.2518 и 0.3044, t 6.646 и 8.684."""
    (a, se_a, t_a), (b, se_b, t_b) = _params(_log_model())
    assert (a, b) == pytest.approx((1.674, 2.644), abs=1e-3)
    assert (se_a, se_b) == pytest.approx((0.2518, 0.3044), abs=1e-4)
    assert (t_a, t_b) == pytest.approx((6.646, 8.684), abs=1e-3)


def test_log_olsen_p_anova_example_17_1a():
    """Книга: Example 17.1A (стр. 454): F(1,18) = 75.410, R²adj = 0.797, s² = 0.1191 (стр. 455)."""
    st = _log_model().statistics
    assert st["F_model"] == pytest.approx(75.410, abs=1e-3)
    assert st["adj_r_squared"] == pytest.approx(0.797, abs=5e-4)
    assert st["ms_resid"] == pytest.approx(0.1191, abs=1e-4)


def test_log_olsen_p_prediction_at_mean():
    """Книга: Example 17.1A (стр. 454–455): μ̂(7.095) = 3.923, 95% ДИ (3.756, 4.090)."""
    pred = _log_model(x_predict="7.095").parameters["predictions"][0]
    assert pred["fitted"] == pytest.approx(3.923, abs=1e-3)
    assert (pred["mean_lower"], pred["mean_upper"]) == pytest.approx((3.756, 4.090), abs=1e-3)


def test_transform_alternatives_include_log():
    """Сравнение преобразований (стр. 452): log10 и ln дают одинаковый R²adj, выбран log10."""
    rows = {r["x_transform"]: r for r in _log_model().parameters["alternatives"]}
    assert rows["log10"]["chosen"] and not rows["ln"]["chosen"]
    assert rows["ln"]["adj_r_squared"] == pytest.approx(rows["log10"]["adj_r_squared"])
    assert rows["log10"]["adj_r_squared"] > rows["none"]["adj_r_squared"]


def test_x_transform_domain_error():
    """log x требует x + c > 0 (стр. 453): без сдвига — ошибка, со сдвигом — расчёт."""
    bad = x_transform_regression([0, 1, 2, 3], [1, 2, 3, 5], "log10")
    assert any(w.code == "x_transform_domain" for w in bad.warnings)
    assert x_transform_regression([0, 1, 2, 3], [1, 2, 3, 5], "log10", x_offset=1).estimates


def test_cubic_sequential_anova_table_17_3():
    """Книга: Table 17.3 (стр. 457): SS 6.9157, 2.2909, 0.1318, Residual 1.7829, Total 11.1213."""
    rows = polynomial_regression(OLSEN_P, YIELD, 3).parameters["anova_table"]
    assert [r["ss"] for r in rows] == pytest.approx(
        [6.9157, 2.2909, 0.1318, 1.7829, 11.1213], abs=1e-4)
    assert [r["F"] for r in rows[:3]] == pytest.approx([62.063, 20.559, 1.183], abs=1e-3)
    assert rows[2]["p"] == pytest.approx(0.293, abs=1e-3)


def test_cubic_flags_top_term_and_vif():
    """Книга: Example 17.1B (стр. 457–458): кубический член незначим, VIF > 100."""
    res = polynomial_regression(OLSEN_P, YIELD, 3)
    codes = {w.code for w in res.warnings}
    assert {"poly_top_not_significant", "poly_collinearity"} <= codes
    assert min(v["vif"] for v in res.parameters["vif"]) > 100
    assert res.statistics["adj_r_squared"] == pytest.approx(0.810, abs=5e-4)


def test_quadratic_parameters_table_17_4():
    """Книга: Table 17.4 (стр. 458): 1.283, 0.593, −0.0279; SE 0.3290, 0.0964, 0.00618."""
    res = polynomial_regression(OLSEN_P, YIELD, 2)
    (a, se_a, _), (b1, se1, t1), (b2, se2, t2) = _params(res, "raw_parameters")
    assert (a, b1, b2) == pytest.approx((1.283, 0.593, -0.0279), abs=5e-4)
    assert (se_a, se1, se2) == pytest.approx((0.3290, 0.0964, 0.00618), abs=1e-4)
    assert (t1, t2) == pytest.approx((6.156, -4.510), abs=1e-3)
    assert res.statistics["adj_r_squared"] == pytest.approx(0.808, abs=5e-4)


@pytest.mark.parametrize("basis", ["centred", "orthogonal"])
def test_bases_give_same_model(basis):
    """Центрирование и ортогональные полиномы (стр. 458–459) не меняют подгонку и простые степени."""
    raw = polynomial_regression(OLSEN_P, YIELD, 3)
    other = polynomial_regression(OLSEN_P, YIELD, 3, basis)
    assert _params(other, "raw_parameters") == pytest.approx(_params(raw, "raw_parameters"))
    assert [r["ss"] for r in other.parameters["anova_table"]] == pytest.approx(
        [r["ss"] for r in raw.parameters["anova_table"]])
    assert max(v["vif"] for v in other.parameters["vif"]) < max(
        v["vif"] for v in raw.parameters["vif"])


@pytest.mark.parametrize(("degree", "ms", "adj", "aic", "sbc"), [
    (2, 0.1126, 80.8, 19.0, 22.0), (3, 0.1114, 81.0, 19.6, 23.5), (8, 0.0799, 86.3, 15.4, 24.4)])
def test_polynomial_summary_table_17_12(degree, ms, adj, aic, sbc):
    """Книга: Table 17.12 (стр. 476): ResMS, R²adj×100, AIC, SBC полиномов порядка 2, 3, 8."""
    st = polynomial_regression(OLSEN_P, YIELD, degree, "orthogonal").statistics
    assert st["ms_resid"] == pytest.approx(ms, abs=1e-4)
    assert 100 * st["adj_r_squared"] == pytest.approx(adj, abs=0.05)
    assert (st["aic"], st["sbc"]) == pytest.approx((aic, sbc), abs=0.05)


def test_log_summary_table_17_12():
    """Книга: Table 17.12 (стр. 476): SLR по log10(Olsen P) — AIC 19.2, SBC 21.2."""
    st = _log_model().statistics
    assert (st["aic"], st["sbc"]) == pytest.approx((19.2, 21.2), abs=0.05)


def test_polynomial_degree_limits():
    """Порядок ≥ числа различных x — ошибка; k − 1 — предупреждение о переподгонке (стр. 459)."""
    x, y = [1, 2, 3, 1, 2, 3], [1.0, 4.1, 8.9, 1.2, 3.9, 9.1]
    assert any(w.code == "poly_degree_too_high" for w in polynomial_regression(x, y, 3).warnings)
    assert any(w.code == "poly_interpolates" for w in polynomial_regression(x, y, 2).warnings)


def test_polynomial_extrapolation_warning_and_curve():
    """Полиномы нельзя экстраполировать (стр. 459); кривая строится на плотной сетке."""
    res = polynomial_regression(OLSEN_P, YIELD, 2, x_predict="7; 20")
    assert any(w.code == "curve_extrapolation" for w in res.warnings)
    curve = res.parameters["curve"]
    assert len(curve["x"]) == 101 and curve["x"][0] == pytest.approx(1.5)
    assert all(lo <= f <= hi for lo, f, hi in zip(curve["lower"], curve["fitted"], curve["upper"]))


def test_polynomial_lack_of_fit_with_replicates():
    """Неадекватность против чистой ошибки при повторах x (стр. 459, 12.8)."""
    x = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    y = [0.1, -0.1, 2.0, 1.8, 0.2, 0.0, 2.1, 1.9, 0.1, -0.1]
    res = polynomial_regression(x, y, 1)
    lof = res.parameters["lack_of_fit"]
    assert (lof["lack_of_fit_df"], lof["pure_error_df"]) == (3, 5)
    assert lof["pure_error_ss"] == pytest.approx(0.1)
    assert any(w.code == "curve_lack_of_fit" for w in res.warnings)


def _temperature(**kw):
    return trigonometric_regression(list(range(1, 13)), TEMPERATURE, 12, **kw)


def test_temperature_parameters_table_17_6():
    """Книга: Table 17.6 (стр. 462): 9.12, −4.09, −5.13; SE 0.139, 0.197; t 65.432, −20.722, −26.007."""
    res = _temperature()
    est, se, t = zip(*_params(res))
    assert est == pytest.approx((9.125, -4.087, -5.129), abs=1e-3)
    assert se == pytest.approx((0.139, 0.197, 0.197), abs=1e-3)
    assert t == pytest.approx((65.432, -20.722, -26.007), abs=1e-3)
    assert res.statistics["adj_r_squared"] == pytest.approx(0.990, abs=5e-4)


def test_temperature_amplitude_phase_example_17_2():
    """Книга: Example 17.2 (стр. 462): γ = 6.56, θ = 2.244 рад = 4.285 мес., max 7.285, min 1.285."""
    res = _temperature()
    harm = res.parameters["harmonics_table"][0]
    assert harm["amplitude"] == pytest.approx(6.56, abs=5e-3)
    assert harm["phase_rad"] == pytest.approx(2.244, abs=1e-3)
    assert harm["phase_time"] == pytest.approx(4.285, abs=1e-3)
    ext = res.parameters["extremes"]
    assert (ext["time_max"], ext["time_min"]) == pytest.approx((7.285, 1.285), abs=1e-3)
    assert ext["value_max"] - ext["value_min"] == pytest.approx(13.12, abs=0.01)


def test_amplitude_phase_delta_method():
    """SE γ и θ дельта-методом (стр. 461) совпадают с численным градиентом."""
    b = np.array([-4.087, -5.129])
    cov = np.array([[0.04, 0.01], [0.01, 0.03]])
    out = amplitude_phase(*b, cov)

    def grad(f):
        h = 1e-6
        return np.array([(f(b + h * e) - f(b - h * e)) / (2 * h) for e in np.eye(2)])

    g_gamma = grad(lambda v: np.hypot(*v))
    g_theta = grad(lambda v: np.arctan2(-v[1], v[0]))
    assert out["se_amplitude"] == pytest.approx(np.sqrt(g_gamma @ cov @ g_gamma))
    assert out["se_phase_rad"] == pytest.approx(np.sqrt(g_theta @ cov @ g_theta))


def test_two_harmonics_reduce_residual():
    """Вторая гармоника — ещё 2 df; экстремумы ищутся по сетке, результат сериализуется."""
    one, two = _temperature(), _temperature(harmonics=2)
    assert two.statistics["ms_resid"] < one.statistics["ms_resid"]
    assert two.parameters["anova_table"][1]["df"] == 2
    assert two.parameters["extremes"]["se_time"] is None
    json.dumps(two.to_dict())
