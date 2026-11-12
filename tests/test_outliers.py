"""Кандидат в выбросы: тест Бонферрони для самого крайнего остатка (core/stats/outliers.py).

Решение аудита 1 (01-01) [предл.: Cook & Weisberg 1982]; книга — стр. 109–110:
кандидата проверяют по записям, метод из-за него не меняют.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from core.stats.outliers import deleted_residuals, mark_residual_plot, outlier_candidate

# Синтетика: 4 группы по 4, SD ≈ 2 во всех; одно значение (54.0) завышено на 14.
GROUPS = [[11.8, 9.7, 10.7, 11.0], [18.6, 21.1, 19.9, 18.8],
          [30.8, 31.7, 26.7, 29.5], [54.0, 41.7, 39.4, 42.0]]


def _fit():
    y = np.array([v for g in GROUPS for v in g])
    X = np.kron(np.eye(4), np.ones((4, 1)))
    return sm.OLS(y, X).fit()


def test_candidate_matches_statsmodels_outlier_test():
    """Удалённые остатки и P Бонферрони — как у statsmodels ``outlier_test``."""
    fit = _fit()
    infl = fit.get_influence()
    r = infl.resid_studentized_internal
    table = fit.outlier_test(method="bonf")
    assert deleted_residuals(r, fit.df_resid) == pytest.approx(table[:, 0])
    worst = int(np.argmax(np.abs(r)))
    assert outlier_candidate(r, fit.df_resid) == worst == 12
    assert table[worst, 2] < 0.05 and (np.delete(table[:, 2], worst) > 0.05).all()


def test_no_candidate_in_clean_data_and_tiny_models():
    r = np.array([0.4, -1.1, 0.9, -0.3, 1.2, -1.1, 0.2, -0.2])
    assert outlier_candidate(r, 6) is None
    assert outlier_candidate([2.0, -2.0], 1) is None  # слишком мало для теста
    assert outlier_candidate([0.0, 0.0, 0.0, 0.0], 3) is None


def test_small_trial_where_threshold_three_is_unreachable():
    """3 группы по 3: |r| ≤ √6 < 3, а удалённый остаток выброс находит."""
    groups = [[20.1, 20.9, 19.6], [24.8, 25.5, 24.2], [29.9, 30.4, 47.0]]
    y = np.array([v for g in groups for v in g])
    fit = sm.OLS(y, np.kron(np.eye(3), np.ones((3, 1)))).fit()
    r = fit.get_influence().resid_studentized_internal
    assert np.abs(r).max() < 3
    assert outlier_candidate(r, fit.df_resid) == 8


def test_mark_residual_plot_adds_position_for_linear_models_only():
    fit = _fit()
    r = fit.get_influence().resid_studentized_internal.tolist()
    bundle = {"residual_vs_fitted": {"fitted": fit.fittedvalues.tolist(), "residuals": r}}
    mark_residual_plot(bundle, "one_way_anova", {"df_resid": int(fit.df_resid)})
    assert bundle["residual_vs_fitted"]["outlier"] == 12
    glm = {"residual_vs_fitted": {"fitted": fit.fittedvalues.tolist(), "residuals": r}}
    mark_residual_plot(glm, "glm_analysis", {"df_resid": int(fit.df_resid)})
    assert "outlier" not in glm["residual_vs_fitted"]
    clean = {"residual_vs_fitted": {"fitted": [1, 2, 3, 4, 5, 6], "residuals": [0.5, -1, 1, -0.4, 0.6, 0.3]}}
    mark_residual_plot(clean, "one_way_anova", {"df_resid": 3})
    assert "outlier" not in clean["residual_vs_fitted"]
    mark_residual_plot({}, "one_way_anova", {})  # нет графика — нет ошибки
