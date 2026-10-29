"""Глава 18: отбор членов GLM, неадекватность и доза для заданной вероятности (Table 18.13)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.stats.glm import contains, eligible, fieller_ratio, glm_analysis
from tests.book_ch18_data import CONIDIA, DEMETHYLATION, LADYBIRD

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _conidia(**kw):
    c = CONIDIA
    return glm_analysis([r[3] for r in c], factor_a=[r[0] for r in c], block=[str(r[1]) for r in c],
                        covariate=[str(r[2]) for r in c], distribution="poisson",
                        covariate_transform="ln", selection="backward",
                        column_names={"factor_a": "Source", "block": "Run", "covariate": "Time"},
                        **kw)


def _p(step):
    return {t["term"][2:]: t["p"] for t in step["tests"]}


def test_conidia_full_model_example_18_4():
    """Книга: Example 18.4 (стр. 510): ResDev 2006.1 на 13 df, φ̂ = 154.3."""
    first = _conidia(lack_of_fit="yes").parameters["selection_steps"][0]
    assert (first["deviance"], first["df_resid"]) == (pytest.approx(2006.1, abs=0.05), 13)
    assert first["phi"] == pytest.approx(154.3, abs=0.05)


def test_conidia_marginal_tests_table_18_13():
    """Книга: Table 18.13 (стр. 510): модели 1–4, P маргинальных F; φ модели 2 = 116.4 (стр. 511)."""
    res = _conidia(lack_of_fit="yes")
    steps = res.parameters["selection_steps"]
    assert _p(steps[0]) == pytest.approx({"FacTime.Source": 0.987}, abs=1e-3)
    assert _p(steps[1]) == pytest.approx({"FacTime": 0.038, "Time.Source": 0.432}, abs=1.5e-3)
    assert _p(steps[2]) == pytest.approx({"FacTime": 0.034, "Source": 0.666}, abs=1e-3)
    assert _p(steps[3]) == pytest.approx({"FacTime": 0.028}, abs=1e-3)
    assert [s["dropped"] for s in steps] == ["FacTime.Source", "Time.Source", "Source", None]
    assert (steps[1]["phi"], steps[1]["df_resid"]) == (pytest.approx(116.4, abs=0.05), 18)
    assert res.parameters["final_terms"] == ["Run", "Time", "FacTime"]


def test_conidia_period_predictions():
    """Книга: Example 18.4 (стр. 511): модель с неадекватностью — предсказание для каждого
    периода среднего прогона; совпадает с моделью [1] + Run + Period."""
    preds = _conidia(lack_of_fit="yes").parameters["factor_predictions"]
    assert [p["level"] for p in preds] == ["5", "10", "15", "20", "25", "40", "80"]
    c = CONIDIA
    period = glm_analysis([r[3] for r in c], factor_a=[str(r[2]) for r in c],
                          block=[str(r[1]) for r in c], distribution="poisson")
    ref = {p["level"]: p for p in period.parameters["factor_predictions"]}
    for p in preds:
        assert (p["eta"], p["se"]) == pytest.approx((ref[p["level"]]["eta"], ref[p["level"]]["se"]))


def test_conidia_power_slope_example_18_4():
    """Книга: Example 18.4 (стр. 512): [1] + Run + logTime, β̂ = 1.43, 95% ДИ (1.20, 1.65)."""
    res = _conidia()
    assert res.parameters["final_terms"] == ["Run", "Time"]
    slope = res.parameters["parameter_tests"][-1]
    assert slope["estimate"] == pytest.approx(1.43, abs=5e-3)
    assert (slope["ci_lower"], slope["ci_upper"]) == pytest.approx((1.20, 1.65), abs=5e-3)


def test_ladybird_backward_selection_example_18_2b():
    """Книга: Example 18.2B (стр. 494): исключаются Sex.Prey (P 0.690) и Prey; Row не проверяется."""
    d = LADYBIRD
    res = glm_analysis([r[4] for r in d], factor_a=[r[2] for r in d], factor_b=[r[3] for r in d],
                       block=[str(r[0]) for r in d], n_trials=6, selection="backward",
                       column_names={"factor_a": "Sex", "factor_b": "Prey", "block": "Row"})
    steps = res.parameters["selection_steps"]
    assert _p(steps[0])["Sex.Prey"] == pytest.approx(0.690, abs=1e-3)
    assert [s["dropped"] for s in steps] == ["Sex.Prey", "Prey", None]
    assert res.parameters["final_terms"] == ["Row", "Sex"]
    assert all("Row" not in _p(s) for s in steps)


def test_marginality_with_lack_of_fit_term():
    """Ковариата маргинальна фактору неадекватности; блок не проверяется (Table 18.13)."""
    assert contains(("x",), ("lof",)) and contains(("x", "A"), ("lof", "A"))
    assert not contains(("lof",), ("x", "A"))
    terms = [("blk",), ("x",), ("lof",), ("A",), ("x", "A"), ("lof", "A")]
    assert eligible(terms) == [("lof", "A")]
    assert eligible(terms[:5]) == [("lof",), ("x", "A")]


def _doses(p="0.5; 0.9"):
    d = DEMETHYLATION
    return glm_analysis([r[1] for r in d], covariate=[str(r[0]) for r in d],
                        trials=[str(r[2]) for r in d], covariate_transform="ln",
                        covariate_offset=0.1, dose_for_p=p, x_predict="0.2632")


def test_dose_for_probability_inverts_prediction():
    """Доза для p (калибровка, стр. 502): w₀ = −α̂/β̂ по Example 18.1B (стр. 486) = −1.0127,
    при ED50 предсказанная вероятность равна 0.5."""
    res = _doses()
    ed50 = res.parameters["doses"][0]
    assert ed50["dose_model"] == pytest.approx(3.188 / -3.148, abs=1e-3)
    assert ed50["dose"] == pytest.approx(np.exp(ed50["dose_model"]) - 0.1)
    assert res.parameters["predictions"][0]["natural"] == pytest.approx(0.5, abs=1e-3)
    assert ed50["lower"] < ed50["dose"] < ed50["upper"]


def test_fieller_limits_solve_the_quadratic():
    """Пределы Филлера — корни |g − a − b·w| = crit·√Var(a + b·w) (Collett, гл. 4)."""
    a, b, v11, v22, v12, crit, g = -3.2, -3.1, 0.026, 0.016, 0.018, 1.96, 0.0
    lo, hi = fieller_ratio(g - a, b, v11, v22, -v12, crit)
    for w in (lo, hi):
        assert abs(g - a - b * w) == pytest.approx(crit * np.sqrt(v11 + 2 * w * v12 + w ** 2 * v22))
    assert fieller_ratio(1.0, 0.1, 1.0, 1.0, 0.0, 1.96) is None
