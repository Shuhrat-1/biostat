"""Простая линейная регрессия (гл. 5 книги).

OLS через statsmodels (проверенный эталон для SE, t, p, R²), результат
кладётся в единую модель ``StatResult``. Остатки и предсказанные значения
сохраняются в ``parameters`` — их подхватывает модуль diagnostics для
построения диагностических графиков (residual plot, Q-Q и т.д.).
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from core.report.schema import Estimate, Severity, StatResult

_MIN_N = 3  # нужно > 2 точек, иначе регрессия вырождена


def _clean_xy(
    x: np.ndarray | list[float], y: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Выровнять x, y по длине и удалить пары с NaN.

    Возвращает очищенные x, y и число удалённых пар.
    """
    ax = np.asarray(x, dtype=float).ravel()
    ay = np.asarray(y, dtype=float).ravel()
    if ax.size != ay.size:
        raise ValueError("x и y должны быть одной длины")
    mask = ~(np.isnan(ax) | np.isnan(ay))
    return ax[mask], ay[mask], int((~mask).sum())


def simple_linear_regression(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    ci_level: float = 0.95,
) -> StatResult:
    """Простая линейная регрессия y = b0 + b1*x методом наименьших квадратов.

    Возвращает коэффициенты с доверительными интервалами, R², F-тест
    значимости модели и сохраняет остатки/предсказания для диагностики.
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level должен быть в (0, 1)")

    cx, cy, n_dropped = _clean_xy(x, y)
    n = cx.size
    result = StatResult(
        method="simple_linear_regression",
        n=n,
        parameters={"ci_level": ci_level},
    )
    if n_dropped:
        result.add_warning(
            "missing_dropped", f"Удалено пар с пропусками: {n_dropped}", Severity.INFO
        )
    if n < _MIN_N:
        result.add_warning("n_too_small", "Нужно ≥ 3 точек", Severity.ERROR)
        return result
    if np.ptp(cx) == 0:
        result.add_warning(
            "no_x_variation", "x не варьирует — наклон не определён", Severity.ERROR
        )
        return result

    model = sm.OLS(cy, sm.add_constant(cx)).fit()
    alpha = 1 - ci_level
    conf = model.conf_int(alpha)

    # coef[0] — свободный член, coef[1] — наклон.
    names = ["intercept", "slope"]
    for i, name in enumerate(names):
        result.estimates.append(
            Estimate(
                name=name,
                value=float(model.params[i]),
                ci_lower=float(conf[i, 0]),
                ci_upper=float(conf[i, 1]),
                ci_level=ci_level,
                std_error=float(model.bse[i]),
            )
        )

    result.p_value = float(model.f_pvalue)  # значимость модели в целом
    result.statistics = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "slope_t": float(model.tvalues[1]),
        "slope_p": float(model.pvalues[1]),
        "intercept_t": float(model.tvalues[0]),
        "intercept_p": float(model.pvalues[0]),
        "df_resid": int(model.df_resid),
        "residual_std_error": float(np.sqrt(model.mse_resid)),
    }
    # Сохраняем для diagnostics — остатки, предсказания, x, стандартизованные остатки.
    influence = model.get_influence()
    result.parameters["residuals"] = model.resid.tolist()
    result.parameters["fitted"] = model.fittedvalues.tolist()
    result.parameters["x"] = cx.tolist()
    result.parameters["std_residuals"] = influence.resid_studentized_internal.tolist()
    return result


def predict(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    x_new: np.ndarray | list[float],
    ci_level: float = 0.95,
) -> StatResult:
    """Предсказание отклика в новых точках x_new.

    Даёт два интервала: доверительный для среднего отклика (mean_ci)
    и предсказательный для отдельного наблюдения (obs_ci, шире).
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level должен быть в (0, 1)")

    cx, cy, _ = _clean_xy(x, y)
    result = StatResult(method="predict", n=cx.size, parameters={"ci_level": ci_level})
    if cx.size < _MIN_N:
        result.add_warning("n_too_small", "Нужно ≥ 3 точек", Severity.ERROR)
        return result

    model = sm.OLS(cy, sm.add_constant(cx)).fit()
    xn = np.asarray(x_new, dtype=float).ravel()
    exog = sm.add_constant(xn, has_constant="add")
    frame = model.get_prediction(exog).summary_frame(alpha=1 - ci_level)

    predictions = []
    for i, xv in enumerate(xn):
        row = frame.iloc[i]
        predictions.append(
            {
                "x": float(xv),
                "predicted": float(row["mean"]),
                "mean_ci_lower": float(row["mean_ci_lower"]),
                "mean_ci_upper": float(row["mean_ci_upper"]),
                "obs_ci_lower": float(row["obs_ci_lower"]),
                "obs_ci_upper": float(row["obs_ci_upper"]),
            }
        )
    result.parameters["predictions"] = predictions
    return result
