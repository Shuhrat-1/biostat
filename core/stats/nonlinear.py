"""Нелинейная регрессия от одной переменной и сравнение кривых (гл. 17.3 книги).

- ``nonlinear_regression`` — МНК для стандартных кривых (стр. 472–474):
  экспоненциальная α + β·exp(−γx), обратная линейная α + β/(1 + γx),
  логистическая α + β/(1 + exp(−γ(x − δ))), Гомперца
  α + β·exp(−exp(−γ(x − δ))) и степенная α + β·x^θ. Начальные значения —
  сетка по нелинейным параметрам с МНК для α, β (кривые частично линейны),
  затем Левенберг–Марквардт. SE приближённые: s²(JᵀJ)⁻¹; предсказания и
  производные параметры — дельта-методом (стр. 476).
- ``curve_comparison`` — ResMS, R²adj, AIC, SBC набора моделей для одних
  данных: преобразования x, полиномы и нелинейные кривые (Table 17.12).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.curves import (
    X_TRANSFORMS,
    _finish_plots,
    _prepare,
    fit_basis,
    polynomial_basis,
    x_transform,
)
from core.stats.linear_regression import parse_numbers
from core.stats.multiple_regression import goodness

_GRID = 101
_SCALES = np.geomspace(0.01, 100, 41)


@dataclass(frozen=True)
class Curve:
    """Кривая f(x; α, β, θ…) = α + β·g(x; θ…): имена параметров и форма g."""

    names: tuple[str, ...]
    shape: Callable[[np.ndarray, np.ndarray], np.ndarray]
    formula: str


def _exp_shape(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.exp(-t[0] * x)


def _inv_shape(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 1 / (1 + t[0] * x)


def _logistic_shape(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-t[0] * (x - t[1])))


def _gompertz_shape(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.exp(-np.exp(-t[0] * (x - t[1])))


def _power_shape(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.power(x, t[0])


CURVES: dict[str, Curve] = {
    "exponential": Curve(("alpha", "beta", "gamma"), _exp_shape, "α + β·exp(−γx)"),
    "inverse_linear": Curve(("alpha", "beta", "gamma"), _inv_shape, "α + β/(1 + γx)"),
    "logistic": Curve(("alpha", "beta", "gamma", "delta"), _logistic_shape,
                      "α + β/(1 + exp(−γ(x − δ)))"),
    "gompertz": Curve(("alpha", "beta", "gamma", "delta"), _gompertz_shape,
                      "α + β·exp(−exp(−γ(x − δ)))"),
    "power": Curve(("alpha", "beta", "theta"), _power_shape, "α + β·x^θ"),
}


# --- Подгонка -----------------------------------------------------------------


def curve_value(curve: Curve, x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """f(x) = α + β·g(x; θ)."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return params[0] + params[1] * curve.shape(np.asarray(x, float), params[2:])


def _profile(curve: Curve, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """RSS и (α, β) при фиксированных нелинейных параметрах θ (частичная линейность)."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        g = curve.shape(x, theta)
    if not np.all(np.isfinite(g)) or np.ptp(g) < 1e-12 * max(1.0, np.abs(g).max()):
        return float("inf"), np.zeros(2)
    design = np.column_stack([np.ones_like(g), g])
    ab = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ ab
    return float(resid @ resid), ab


def _theta_grid(curve: Curve, x: np.ndarray) -> list[np.ndarray]:
    """Сетка нелинейных параметров в масштабе данных: γ ~ ±k/размах, δ — по диапазону x."""
    span = float(np.ptp(x)) or 1.0
    rates = np.concatenate([_SCALES, -_SCALES]) / span
    if curve.names[2] == "theta":
        return [np.array([t]) for t in np.linspace(-4, 4, 81) if abs(t) > 1e-9]
    if len(curve.names) == 3:
        return [np.array([g]) for g in rates]
    deltas = np.linspace(x.min(), x.max(), 21)
    return [np.array([g, d]) for g in rates[::2] for d in deltas]


def _start(curve: Curve, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Лучшая точка сетки по профильной RSS."""
    best, start = float("inf"), None
    for theta in _theta_grid(curve, x):
        rss, ab = _profile(curve, x, y, theta)
        if rss < best:
            best, start = rss, np.concatenate([ab, theta])
    return start


def _jacobian(curve: Curve, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """∂f/∂параметры: аналитически для α, β, численно (центральные разности) для θ."""
    cols = [np.ones_like(x), curve.shape(x, p[2:])]
    for j in range(2, p.size):
        h = 1e-6 * max(1.0, abs(p[j]))
        up, down = p.copy(), p.copy()
        up[j] += h
        down[j] -= h
        cols.append((curve_value(curve, x, up) - curve_value(curve, x, down)) / (2 * h))
    return np.column_stack(cols)


def fit_curve(curve: Curve, x: np.ndarray, y: np.ndarray, start: np.ndarray | None = None) -> dict | None:
    """МНК-оценки, приближённая ковариация s²(JᵀJ)⁻¹, RSS; None — не сошлось."""
    p0 = _start(curve, x, y) if start is None else np.asarray(start, float)
    if p0 is None:
        return None
    sol = optimize.least_squares(lambda p: curve_value(curve, x, p) - y, p0, method="lm",
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    resid = sol.fun
    if not sol.success or not np.all(np.isfinite(resid)):
        return None
    jac = _jacobian(curve, x, sol.x)
    df = len(y) - sol.x.size
    rss = float(resid @ resid)
    s2 = rss / df
    return {"beta": sol.x, "cov": s2 * np.linalg.pinv(jac.T @ jac), "rss": rss, "df": df,
            "s2": s2, "p": sol.x.size, "fitted": y + resid, "resid": -resid, "design": jac}


# --- Таблицы и предсказания -----------------------------------------------------


def _anova(fit: dict, y: np.ndarray) -> list[dict]:
    """Приближённая ANOVA: модель (p − 1 df) против остатка (стр. 476)."""
    tot = float(((y - y.mean()) ** 2).sum())
    mod_df = fit["p"] - 1
    f = (tot - fit["rss"]) / mod_df / fit["s2"] if fit["s2"] > 0 else float("inf")
    return [{"term": "Regression", "df": mod_df, "ss": tot - fit["rss"],
             "ms": (tot - fit["rss"]) / mod_df, "F": f,
             "p": float(stats.f.sf(f, mod_df, fit["df"]))},
            {"term": "Residual", "df": fit["df"], "ss": fit["rss"], "ms": fit["s2"]},
            {"term": "Total", "df": len(y) - 1, "ss": tot, "ms": tot / (len(y) - 1)}]


def _param_rows(curve: Curve, fit: dict, alpha: float) -> list[dict]:
    """Оценка, приближённые SE, t, P и ДИ."""
    se = np.sqrt(np.clip(np.diag(fit["cov"]), 0, None))
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = []
    for j, name in enumerate(curve.names):
        b = float(fit["beta"][j])
        t = b / se[j] if se[j] > 0 else float("nan")
        rows.append({"term": name, "estimate": b, "se": float(se[j]), "t": float(t),
                     "p": float(2 * stats.t.sf(abs(t), fit["df"])),
                     "ci_lower": b - t_crit * se[j], "ci_upper": b + t_crit * se[j]})
    return rows


def _derived(model: str, fit: dict) -> list[dict]:
    """Асимптоты, φ = exp(−γ), середина S-кривой — дельта-методом."""
    b, cov = fit["beta"], fit["cov"]

    def row(name: str, value: float, grad: np.ndarray) -> dict:
        return {"quantity": name, "estimate": float(value),
                "se": float(np.sqrt(max(grad @ cov @ grad, 0.0)))}

    e = np.eye(b.size)
    rows = []
    if model in ("exponential", "inverse_linear"):
        rows.append(row("value_at_0", b[0] + b[1], e[0] + e[1]))
        if b[2] > 0:
            rows.append(row("asymptote", b[0], e[0]))
    if model == "exponential":
        phi = np.exp(-b[2])
        rows.append(row("phi", phi, -phi * e[2]))
    if model in ("logistic", "gompertz"):
        rows += [row("lower_asymptote", b[0], e[0]), row("upper_asymptote", b[0] + b[1],
                                                          e[0] + e[1])]
    if model == "logistic":
        rows.append(row("value_at_delta", b[0] + b[1] / 2, e[0] + e[1] / 2))
    return rows


def _predict(curve: Curve, fit: dict, points: np.ndarray, alpha: float) -> list[dict]:
    """μ̂(x) с ДИ среднего (дельта-метод) и нового наблюдения."""
    if points.size == 0:
        return []
    g = _jacobian(curve, points, fit["beta"])
    mean = curve_value(curve, points, fit["beta"])
    var = np.clip(np.einsum("ij,jk,ik->i", g, fit["cov"], g), 0, None)
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    se_m, se_o = np.sqrt(var), np.sqrt(var + fit["s2"])
    return [{"x": float(points[i]), "fitted": float(mean[i]), "se_mean": float(se_m[i]),
             "mean_lower": float(mean[i] - t_crit * se_m[i]),
             "mean_upper": float(mean[i] + t_crit * se_m[i]), "se_new": float(se_o[i]),
             "new_lower": float(mean[i] - t_crit * se_o[i]),
             "new_upper": float(mean[i] + t_crit * se_o[i])} for i in range(points.size)]


def _parse_start(text: str, curve: Curve) -> np.ndarray | None:
    """«alpha=4; beta=-4; gamma=0.3» → вектор; пусто — автоматический подбор."""
    if not (text or "").strip():
        return None
    values = {}
    for part in text.replace(",", ";").split(";"):
        key, sep, value = part.partition("=")
        if sep:
            values[key.strip()] = float(value)
    missing = [n for n in curve.names if n not in values]
    if missing:
        raise ValueError(f"Начальные значения не заданы для: {', '.join(missing)}")
    return np.array([values[n] for n in curve.names])


def _outputs(model: str, curve: Curve, fit: dict, frame: pd.DataFrame, x_predict: str,
             result: StatResult) -> None:
    """Таблицы, статистики, оценки, предсказания и кривая с ДИ."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    alpha = result.parameters["alpha"]
    params = _param_rows(curve, fit, alpha)
    rows = _anova(fit, y)
    result.parameters.update({"anova_table": rows, "parameter_tests": params,
                              "derived": _derived(model, fit), "formula": curve.formula})
    result.p_value = rows[0]["p"]
    tot = rows[2]["ss"]
    result.statistics.update({"F_model": rows[0]["F"], "p_model": rows[0]["p"],
                              "df_resid": fit["df"], "ms_resid": fit["s2"], "n_params": fit["p"],
                              **goodness(fit["rss"], tot, len(y), fit["p"])})
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))
    _finish_plots(frame, fit, result)
    grid = np.linspace(x.min(), x.max(), _GRID)
    band = _predict(curve, fit, grid, alpha)
    result.parameters["curve"] = {k: [r[src] for r in band] for k, src in
                                  (("x", "x"), ("fitted", "fitted"), ("lower", "mean_lower"),
                                   ("upper", "mean_upper"))}
    points = np.asarray(parse_numbers(x_predict), float)
    result.parameters["predictions"] = _predict(curve, fit, points, alpha)
    result.parameters["points"] = {"x": x.tolist(), "y": y.tolist()}


def nonlinear_regression(
    x: list[float],
    y: list[float],
    model: str = "exponential",
    start: str = "",
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Нелинейная кривая по МНК с приближённой инференцией (17.3, Example 17.1C)."""
    if model not in CURVES:
        raise ValueError(f"model ∈ {tuple(CURVES)}")
    curve = CURVES[model]
    frame, result, ok = _prepare("nonlinear_regression", x, y, alpha, transform_kind, offset,
                                 len(curve.names) + 1, model=model)
    if not ok:
        return result
    x_arr, y_arr = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    if model == "power" and np.any(x_arr <= 0):
        result.add_warning("x_transform_domain", "Степенная кривая требует x > 0", Severity.ERROR)
        return result
    fit = fit_curve(curve, x_arr, y_arr, _parse_start(start, curve))
    if fit is not None and model == "logistic" and fit["beta"][2] < 0:
        # Та же кривая при (α + β, −β, −γ, δ): приводим к γ > 0, как в книге (стр. 474).
        a, b, g, d = fit["beta"]
        fit = fit_curve(curve, x_arr, y_arr, np.array([a + b, -b, -g, d]))
    if fit is None:
        result.add_warning("nonlinear_not_converged", "Итерации не сошлись: задай начальные "
                           "значения параметров или выбери другую кривую.", Severity.ERROR)
        return result
    _outputs(model, curve, fit, frame, x_predict, result)
    _degenerate(model, fit, x_arr, result)
    result.add_warning("nonlinear_approximate", "SE, t и F нелинейной модели приближённые "
                       "(линеаризация); проверь кривую на графике.", Severity.INFO)
    _extrapolation(frame, result)
    return result


def _degenerate(model: str, fit: dict, x: np.ndarray, result: StatResult) -> None:
    """S-кривая с точкой перегиба δ вне данных: асимптота не определяется (стр. 475–476)."""
    if model in ("logistic", "gompertz") and not x.min() <= fit["beta"][3] <= x.max():
        result.add_warning("nonlinear_degenerate", f"Точка перегиба δ = {fit['beta'][3]:.3g} вне "
                           "диапазона x: данные не содержат обеих частей S-кривой, асимптоты "
                           "не оцениваются. Выбери кривую с одной асимптотой.",
                           params={"delta": f"{fit['beta'][3]:.3g}"})


def _extrapolation(frame: pd.DataFrame, result: StatResult) -> None:
    """Предупреждение о точках предсказания вне диапазона x."""
    lo, hi = float(frame["x"].min()), float(frame["x"].max())
    outside = [r["x"] for r in result.parameters["predictions"] if not lo <= r["x"] <= hi]
    if outside:
        listed = ", ".join(f"{p:g}" for p in outside)
        result.add_warning("curve_extrapolation", f"Предсказание вне диапазона x [{lo:g}; {hi:g}]: "
                           f"{listed}.", params={"points": listed, "low": f"{lo:g}",
                                                  "high": f"{hi:g}"})


# --- curve_comparison -------------------------------------------------------------


def _linear_candidate(token: str, x: np.ndarray, y: np.ndarray) -> dict | None:
    """linear, log10…reciprocal, polyK → МНК-подгонка; None — недопустимо на данных."""
    if token == "linear":
        design = np.column_stack([np.ones_like(x), x])
    elif token in X_TRANSFORMS:
        try:
            design = np.column_stack([np.ones_like(x), x_transform(x, token)])
        except ValueError:
            return None
    else:
        degree = int(token[4:])
        if not 1 <= degree < np.unique(x).size:
            return None
        design = polynomial_basis(x, degree, "orthogonal")(x)
    return fit_basis(design, y) if np.all(np.isfinite(design)) else None


def _candidate(token: str, x: np.ndarray, y: np.ndarray) -> dict | None:
    """Подогнать модель-кандидата по её обозначению."""
    if token in CURVES:
        return fit_curve(CURVES[token], x, y)
    if token in ("linear", *X_TRANSFORMS) or (token.startswith("poly") and token[4:].isdigit()):
        return _linear_candidate(token, x, y)
    raise ValueError(f"Неизвестная модель {token!r}: linear, {', '.join(X_TRANSFORMS)}, polyK, "
                     f"{', '.join(CURVES)}")


def _best(rows: list[dict]) -> dict[str, str]:
    """Лучшая модель по R²adj (max), AIC и SBC (min)."""
    ok = [r for r in rows if r["fitted"]]
    if not ok:
        return {}
    best = {"adj_r_squared": max(ok, key=lambda r: r["adj_r_squared"])["model"]}
    best.update({k: min(ok, key=lambda r, k=k: r[k])["model"] for k in ("aic", "sbc")})
    return best


def curve_comparison(
    x: list[float],
    y: list[float],
    models: str = "linear, log10, poly2, poly3, exponential, inverse_linear",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Сводка качества подгонки моделей для одних данных (Table 17.12)."""
    tokens = [t.strip().lower() for t in models.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        raise ValueError("Перечисли модели через запятую")
    frame, result, ok = _prepare("curve_comparison", x, y, alpha, transform_kind, offset, 4,
                                 models=tokens)
    if not ok:
        return result
    x_arr, y_arr = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    rows = []
    for token in tokens:
        fit = _candidate(token, x_arr, y_arr)
        row = {"model": token, "fitted": fit is not None and fit["df"] > 0}
        if row["fitted"]:
            row.update({"p": fit["p"], "ms_resid": fit["s2"],
                        **goodness(fit["rss"], tot, len(y_arr), fit["p"])})
        rows.append(row)
    result.parameters.update({"models_table": rows, "best": _best(rows)})
    failed = [r["model"] for r in rows if not r["fitted"]]
    if failed:
        result.add_warning("curve_models_failed", f"Не подогнаны: {', '.join(failed)}",
                           params={"models": ", ".join(failed)})
    return result
