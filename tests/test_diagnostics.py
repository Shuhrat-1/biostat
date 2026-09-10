"""Тесты диагностики допущений: эталоны и срабатывание предупреждений."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

from core.stats.diagnostics import diagnose, normality_test
from core.stats.regression import simple_linear_regression


def _clean_regression():
    """Регрессия на данных с нормальными остатками (допущения выполнены)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    y = 2.0 + 1.5 * x + rng.normal(0, 1, 40)
    return simple_linear_regression(x, y)


def test_diagnose_requires_residuals():
    """Диагностика отклоняет источник без сохранённых остатков."""
    from core.stats.descriptive import describe

    with pytest.raises(ValueError):
        diagnose(describe([1.0, 2.0, 3.0, 4.0]))


def test_diagnose_clean_data_no_warnings():
    """На чистых данных нет предупреждений о нарушениях."""
    r = diagnose(_clean_regression())
    codes = {w.code for w in r.warnings}
    assert "normality_violated" not in codes
    assert "autocorrelation" not in codes


def test_diagnose_shapiro_matches_scipy():
    reg = _clean_regression()
    r = diagnose(reg)
    resid = np.asarray(reg.parameters["residuals"])
    assert r.statistics["shapiro_p"] == pytest.approx(stats.shapiro(resid).pvalue)


def test_diagnose_dw_matches_statsmodels():
    reg = _clean_regression()
    r = diagnose(reg)
    resid = np.asarray(reg.parameters["residuals"])
    assert r.statistics["durbin_watson"] == pytest.approx(durbin_watson(resid))


def test_diagnose_plot_data_complete():
    """Все четыре графика с полными рядами одинаковой длины."""
    reg = _clean_regression()
    plots = diagnose(reg).parameters["plots"]
    n = reg.n
    assert set(plots) == {
        "residual_vs_fitted",
        "qq",
        "histogram",
        "scale_location",
    }
    assert len(plots["residual_vs_fitted"]["fitted"]) == n
    assert len(plots["residual_vs_fitted"]["residuals"]) == n
    assert len(plots["qq"]["theoretical_quantiles"]) == n
    assert len(plots["qq"]["sample_quantiles"]) == n
    assert len(plots["scale_location"]["sqrt_abs_std_residuals"]) == n


def test_qq_theoretical_quantiles_sorted():
    """Теоретические квантили Q-Q монотонно возрастают."""
    plots = diagnose(_clean_regression()).parameters["plots"]
    tq = plots["qq"]["theoretical_quantiles"]
    assert tq == sorted(tq)


def test_diagnose_detects_nonnormal_residuals():
    """Кривая зависимость → остатки от линейной модели ненормальны."""
    x = np.linspace(0, 10, 60)
    y = x**2  # сильно нелинейно
    reg = simple_linear_regression(x, y)
    r = diagnose(reg)
    assert any(w.code == "normality_violated" for w in r.warnings)


def test_diagnose_detects_autocorrelation():
    """Тренд в остатках (нелинейность) даёт низкий Дарбин-Уотсон."""
    x = np.linspace(0, 10, 60)
    y = x**2
    r = diagnose(simple_linear_regression(x, y))
    assert r.statistics["durbin_watson"] < 1.0
    assert any(w.code == "autocorrelation" for w in r.warnings)


def test_normality_test_matches_scipy():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 100)
    r = normality_test(data)
    ref = stats.shapiro(data)
    assert r.statistics["W"] == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_normality_test_flags_skewed():
    rng = np.random.default_rng(2)
    data = rng.exponential(1.0, 100)
    r = normality_test(data)
    assert any(w.code == "normality_violated" for w in r.warnings)


def test_normality_test_too_small():
    r = normality_test([1.0, 2.0])
    assert any(w.code == "n_out_of_range" for w in r.warnings)


def test_diagnose_json_serializable():
    import json

    r = diagnose(_clean_regression())
    parsed = json.loads(r.to_json())
    assert parsed["method"] == "diagnose"
    assert "plots" in parsed["parameters"]
