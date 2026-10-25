"""Регрессионные тесты исправлений существующего кода (патч после гл. 6)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from core.report.schema import StatResult
from core.stats.anova import one_way_anova
from core.stats.descriptive import describe_all
from core.stats.diagnostics import diagnose
from core.stats.hypothesis import levene_p_value, two_sample_t
from core.stats.regression import simple_linear_regression


def _strict_json(result: StatResult) -> dict:
    """Сериализация так же строго, как JSON.parse в браузере."""
    return json.loads(json.dumps(result.to_dict(), allow_nan=False))


def test_nan_and_inf_serialize_as_null():
    r = StatResult(method="x", statistics={"a": float("nan"), "b": float("inf"), "c": 1.5})
    payload = _strict_json(r)["statistics"]
    assert payload == {"a": None, "b": None, "c": 1.5}


def test_describe_all_zero_mean_cv_is_null():
    """Было: cv_pct = NaN → JSON.parse падал. CV при нулевом среднем не определён (стр. 16)."""
    payload = _strict_json(describe_all({"z": [-1.0, 0.0, 1.0]}))
    assert payload["parameters"]["summary"]["z"]["cv_pct"] is None


def test_one_way_anova_constant_data_json_safe():
    """Было: eta_squared = NaN при ss_total = 0."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy по-разному предупреждает о константах
        r = one_way_anova([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert _strict_json(r)["statistics"]["eta_squared"] is None


def test_levene_degenerate_for_pairs_returns_none():
    assert levene_p_value(np.array([1.0, 2.5]), np.array([4.0, 3.5])) is None
    assert levene_p_value(np.array([1.0, 2.0, 4.0]), np.array([3.0, 5.0, 9.0])) is not None


def test_anova_pairs_no_false_heterogeneity_warning():
    """Было: Левене p = 0 → ложное variance_heterogeneity."""
    r = one_way_anova([1.0, 2.5], [4.0, 3.5], [6.0, 9.0])
    codes = {w.code for w in r.warnings}
    assert "variance_heterogeneity" not in codes
    assert "levene_undefined" in codes
    assert "levene_p" not in r.statistics


def test_two_sample_t_pairs_uses_pooled_variant():
    """Было: молча Уэлч. Теперь — Стьюдент (вариант книги по умолчанию) с пометкой."""
    r = two_sample_t([1.0, 2.5], [4.0, 3.5])
    assert r.parameters["variant"] == "student"
    assert r.statistics["t"] == pytest.approx(stats.ttest_ind([1.0, 2.5], [4.0, 3.5]).statistic)
    assert "levene_undefined" in {w.code for w in r.warnings}


def test_two_sample_t_normal_case_unchanged():
    """Обычный случай: выбор по Левене как прежде (Example 2.1C → Стьюдент)."""
    r = two_sample_t([7, 9, 6, 12, 4, 6, 9], [12, 8, 12, 9, 8, 16, 7])
    assert r.parameters["variant"] == "student"
    assert r.statistics["levene_p"] == pytest.approx(0.6947, abs=5e-4)


def _regression():
    rng = np.random.default_rng(31)
    x = np.linspace(0, 10, 25)
    y = 1 + 0.8 * x + rng.normal(0, 1, 25)
    y[3] += 6  # точка с большим остатком и плечом ≠ среднему
    return x, y, simple_linear_regression(x, y)


def test_diagnose_uses_internally_studentized_residuals():
    """Было: resid / SD(resid). Теперь e/SE(e) по книге (стр. 95–96, 328–329)."""
    x, y, reg = _regression()
    plots = diagnose(reg).parameters["plots"]
    ref = sm.OLS(y, sm.add_constant(x)).fit().get_influence().resid_studentized_internal
    got = plots["scale_location"]["sqrt_abs_std_residuals"]
    assert got == pytest.approx(list(np.sqrt(np.abs(ref))))


def test_diagnose_qq_uses_book_positions_on_std_residuals():
    """Q-Q: (i − 0.375)/(n + 0.25) против упорядоченных стандартизованных остатков (стр. 101)."""
    x, y, reg = _regression()
    qq = diagnose(reg).parameters["plots"]["qq"]
    n = len(y)
    assert qq["theoretical_quantiles"][0] == pytest.approx(stats.norm.ppf(0.625 / (n + 0.25)))
    assert qq["sample_quantiles"] == pytest.approx(sorted(reg.parameters["std_residuals"]))


def test_requirements_include_charset_normalizer():
    """Без него не импортируется core/validation/dialect.py в чистом окружении."""
    text = open("requirements.txt", encoding="utf-8").read()
    assert "charset-normalizer" in text
