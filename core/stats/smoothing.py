"""Сглаживание: регрессионные сплайны, сглаживающий сплайн и loess (гл. 17.1.2, стр. 460).

Книга называет их альтернативой полиномам, ограничивающей «шероховатость» кривой
и избегающей переподгонки (Ruppert et al., 2003), без примеров. ``smooth_curve``:

- ``bspline`` — регрессионный B-сплайн степени k с внутренними узлами по
  квантилям x: обычная МЛР (SE, ДИ, F-тест «сплайн против прямой», R²adj, AIC,
  SBC, неадекватность при повторах x).
- ``smoothing_spline`` — кубический сглаживающий сплайн (штрафная регрессия по
  B-сплайнам с узлами в x — критерий scipy make_smoothing_spline), штраф λ по
  обобщённой кросс-валидации GCV = n·RSS/(n − tr S)²; эффективные df = tr S,
  σ̂² = RSS/(n − tr S), поточечные SE по матрице сглаживания S.
- ``loess`` — локально-взвешенная линейная регрессия (statsmodels lowess) с
  робастными итерациями; без формальной инференции.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import BSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

from core.report.schema import Severity, StatResult
from core.stats.curves import _finish, _prepare, fit_basis
from core.stats.linear_regression import parse_numbers

SMOOTHERS = ("bspline", "smoothing_spline", "loess")
_GRID = 101
_MAX_UNIQUE = 500


# --- Регрессионный B-сплайн ------------------------------------------------------


def bspline_knots(x: np.ndarray, n_knots: int, degree: int) -> np.ndarray:
    """Узловой вектор: границы кратности k + 1, внутренние узлы — квантили различных x."""
    inner = np.quantile(np.unique(x), np.linspace(0, 1, n_knots + 2)[1:-1]) if n_knots else []
    lo, hi = float(x.min()), float(x.max())
    return np.r_[[lo] * (degree + 1), inner, [hi] * (degree + 1)]


def bspline_basis(knots: np.ndarray, degree: int) -> Callable[[np.ndarray], np.ndarray]:
    """Базис B-сплайна как функция x (значения за границами прижимаются к границам)."""
    lo, hi = knots[0], knots[-1]

    def basis(v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, float), lo, hi)
        return BSpline.design_matrix(v, knots, degree).toarray()
    return basis


def _bspline(frame: pd.DataFrame, n_knots: int, degree: int, x_predict: str,
             result: StatResult) -> None:
    """МЛР по B-сплайну, F против прямой, кривая с ДИ."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    basis = bspline_basis(bspline_knots(x, n_knots, degree), degree)
    fit = fit_basis(basis(x), y)
    line = fit_basis(np.column_stack([np.ones_like(x), x]), y)
    ddf = line["df"] - fit["df"]
    f = (line["rss"] - fit["rss"]) / ddf / fit["s2"] if ddf > 0 else float("nan")
    result.statistics.update({"F_vs_line": float(f), "df_vs_line": ddf,
                              "p_vs_line": float(stats.f.sf(f, ddf, fit["df"])) if ddf else 1.0,
                              "n_basis": fit["p"]})
    _finish(frame, fit, [], result)
    _curve_output(frame, lambda g: _linear_band(basis, fit, g, result), x_predict, result)


def _linear_band(basis: Callable[[np.ndarray], np.ndarray], fit: dict, grid: np.ndarray, result: StatResult) -> dict:
    """μ̂ и ДИ среднего по базису."""
    d = basis(grid)
    mean = d @ fit["beta"]
    se = np.sqrt(np.einsum("ij,jk,ik->i", d, fit["cov"], d))
    crit = float(stats.t.ppf(1 - result.parameters["alpha"] / 2, fit["df"]))
    return {"fitted": mean, "lower": mean - crit * se, "upper": mean + crit * se, "se": se}


# --- Сглаживающий сплайн ----------------------------------------------------------


def _aggregate(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Средние по различным x, их численности и внутригрупповая SS (повторы x)."""
    g = frame.groupby("x")["y"]
    xs = np.asarray(g.mean().index, float)
    within = float(((frame["y"] - g.transform("mean")) ** 2).sum())
    return xs, g.mean().to_numpy(float), g.size().to_numpy(float), within


class PenalizedSpline:
    """Кубический сглаживающий сплайн как штрафная регрессия по B-сплайнам с узлами в x:
    min Σ w(ȳ − f)² + λ∫f''² — тот же критерий, что у scipy make_smoothing_spline."""

    def __init__(self, xs: np.ndarray, ym: np.ndarray, w: np.ndarray) -> None:
        self.knots = np.r_[[xs[0]] * 4, xs[1:-1], [xs[-1]] * 4]
        self.xs, self.ym, self.w = xs, ym, w
        self.b = BSpline.design_matrix(xs, self.knots, 3).toarray()
        self.btwb = self.b.T @ (w[:, None] * self.b)
        self.btwy = self.b.T @ (w * ym)
        self.omega = self._penalty()

    def _penalty(self) -> np.ndarray:
        """Ω_jk = ∫B_j''B_k'': B'' кусочно-линейны, 2-точечная квадратура Гаусса точна."""
        nodes, weights = np.polynomial.legendre.leggauss(2)
        a, b = self.xs[:-1], self.xs[1:]
        pts = ((b - a)[:, None] * (nodes + 1) / 2 + a[:, None]).ravel()
        wts = ((b - a)[:, None] * weights / 2).ravel()
        eye = np.eye(self.b.shape[1])
        d2 = np.column_stack([BSpline(self.knots, eye[j], 3).derivative(2)(pts)
                              for j in range(eye.shape[0])])
        return d2.T @ (wts[:, None] * d2)

    def solve(self, lam: float) -> dict:
        """Коэффициенты, A⁻¹ и эффективные df = tr(A⁻¹BᵀWB)."""
        a_inv = np.linalg.inv(self.btwb + lam * self.omega)
        coef = a_inv @ self.btwy
        return {"lam": lam, "coef": coef, "a_inv": a_inv,
                "edf": float(np.trace(a_inv @ self.btwb)), "fitted": self.b @ coef}

    def basis(self, v: np.ndarray) -> np.ndarray:
        """Базис в новых точках (за границами — прижатие к границе)."""
        v = np.clip(np.asarray(v, float), self.xs[0], self.xs[-1])
        return BSpline.design_matrix(v, self.knots, 3).toarray()


def _gcv(ps: PenalizedSpline, within: float, n: int, log_lam: float) -> float:
    """GCV(λ) = n·RSS/(n − tr S)² на исходных наблюдениях."""
    sol = ps.solve(float(np.exp(log_lam)))
    rss = within + float((ps.w * (ps.ym - sol["fitted"]) ** 2).sum())
    return n * rss / (n - sol["edf"]) ** 2


def choose_lambda(ps: PenalizedSpline, within: float, n: int) -> float:
    """λ̂ по минимуму GCV: сетка по log λ, затем уточнение."""
    span = float(np.ptp(ps.xs))
    grid = np.linspace(np.log(1e-10 * span ** 3), np.log(1e3 * span ** 3), 61)
    values = [_gcv(ps, within, n, g) for g in grid]
    k = int(np.argmin(values))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    best = optimize.minimize_scalar(lambda g: _gcv(ps, within, n, g), bounds=(lo, hi),
                                    method="bounded", options={"xatol": 1e-8})
    return float(np.exp(best.x))


def _smoothing_spline(frame: pd.DataFrame, lam: float, x_predict: str,
                      result: StatResult) -> None:
    """Сглаживающий сплайн: λ (GCV), эффективные df, σ̂², поточечные SE."""
    xs, ym, w, within = _aggregate(frame)
    if xs.size < 5 or xs.size > _MAX_UNIQUE:
        result.add_warning("smooth_bad_size", f"Нужно 5–{_MAX_UNIQUE} различных значений x.",
                           Severity.ERROR)
        return
    n = len(frame)
    ps = PenalizedSpline(xs, ym, w)
    cap = 1e3 * float(np.ptp(xs)) ** 3  # дальше — практически прямая, а матрица плохо обусловлена
    if lam > cap:
        result.add_warning("smooth_lambda_capped", f"λ уменьшен до {cap:.3g}: при большем "
                           "штрафе кривая уже совпадает с прямой МНК.", params={"cap": f"{cap:.3g}"})
    sol = ps.solve(choose_lambda(ps, within, n) if lam <= 0 else min(lam, cap))
    rss = within + float((w * (ym - sol["fitted"]) ** 2).sum())
    df = n - sol["edf"]
    s2 = rss / df
    tot = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    result.statistics.update({"lambda": sol["lam"], "edf": sol["edf"], "ms_resid": s2,
                              "df_resid": df, "r_squared": 1 - rss / tot,
                              "gcv": n * rss / df ** 2})
    middle = sol["a_inv"] @ ps.btwb @ sol["a_inv"]
    crit = float(stats.t.ppf(1 - result.parameters["alpha"] / 2, df))

    def band(grid: np.ndarray) -> dict:
        bg = ps.basis(grid)
        fit = bg @ sol["coef"]
        se = np.sqrt(s2 * np.einsum("ij,jk,ik->i", bg, middle, bg))
        return {"fitted": fit, "lower": fit - crit * se, "upper": fit + crit * se, "se": se}

    _residuals(frame, ps.basis(frame["x"].to_numpy(float)) @ sol["coef"], result)
    _curve_output(frame, band, x_predict, result)


# --- loess ---------------------------------------------------------------------------


def _loess(frame: pd.DataFrame, frac: float, iterations: int, x_predict: str,
           result: StatResult) -> None:
    """Локальная линейная регрессия с робастными весами (статистика — описательная)."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    fitted = lowess(y, x, frac=frac, it=int(iterations), return_sorted=False)
    order = np.argsort(x)

    def band(grid: np.ndarray) -> dict:
        return {"fitted": np.interp(grid, x[order], fitted[order])}

    rss = float(((y - fitted) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    result.statistics.update({"frac": frac, "iterations": int(iterations),
                              "r_squared": 1 - rss / tot, "rmse": float(np.sqrt(rss / y.size))})
    _residuals(frame, fitted, result)
    _curve_output(frame, band, x_predict, result)
    result.add_warning("loess_descriptive", "Loess — описательная кривая: SE и тесты не "
                       "вычисляются.", Severity.INFO)


# --- Общие выходы ---------------------------------------------------------------------


def _residuals(frame: pd.DataFrame, fitted: np.ndarray, result: StatResult) -> None:
    """Остатки против подгонки и x."""
    r = frame["y"].to_numpy(float) - fitted
    n = r.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    result.parameters["plots"] = {
        "residual_vs_fitted": {"fitted": fitted.tolist(), "residuals": r.tolist()},
        "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
        "std_residuals": (r / r.std(ddof=1)).tolist() if n > 1 else r.tolist(),
        "residual_vs_x": {"x": frame["x"].tolist(), "residuals": r.tolist()}}


def _curve_output(frame: pd.DataFrame, band: Callable[[np.ndarray], dict], x_predict: str, result: StatResult) -> None:
    """Кривая на сетке, предсказания, точки данных; предупреждение об экстраполяции."""
    lo, hi = float(frame["x"].min()), float(frame["x"].max())
    grid = np.linspace(lo, hi, _GRID)
    b = band(grid)
    result.parameters["curve"] = {"x": grid.tolist(), **{k: np.asarray(v).tolist()
                                                        for k, v in b.items() if k != "se"}}
    points = np.asarray(parse_numbers(x_predict), float)
    if points.size:
        bp = band(points)
        result.parameters["predictions"] = [
            {"x": float(p), **{k: float(np.asarray(v)[i]) for k, v in bp.items()}}
            for i, p in enumerate(points)]
        if np.any((points < lo) | (points > hi)):
            result.add_warning("curve_extrapolation", "Сглаживание не экстраполирует: точки вне "
                               f"[{lo:g}; {hi:g}] прижаты к границе.",
                               params={"points": x_predict, "low": f"{lo:g}", "high": f"{hi:g}"})
    result.parameters["points"] = {"x": frame["x"].tolist(), "y": frame["y"].tolist()}


def smooth_curve(
    x: list[float],
    y: list[float],
    method: str = "bspline",
    n_knots: int = 3,
    degree: int = 3,
    lam: float = 0.0,
    frac: float = 0.667,
    iterations: int = 3,
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Гибкая кривая без переподгонки: B-сплайн, сглаживающий сплайн или loess (стр. 460)."""
    n_knots, degree = int(n_knots), int(degree)
    if method not in SMOOTHERS or not 0 <= n_knots <= 20 or not 1 <= degree <= 3 or \
            not 0 < frac <= 1 or lam < 0:
        raise ValueError(f"method ∈ {SMOOTHERS}, n_knots ∈ [0, 20], degree ∈ [1, 3], "
                         "0 < frac ≤ 1, lam ≥ 0")
    n_par = n_knots + degree + 1 if method == "bspline" else 4
    frame, result, ok = _prepare("smooth_curve", x, y, alpha, transform_kind, offset, n_par + 1,
                                 smoother=method)
    if not ok:
        return result
    if method == "bspline":
        _bspline(frame, n_knots, degree, x_predict, result)
    elif method == "smoothing_spline":
        _smoothing_spline(frame, lam, x_predict, result)
    else:
        _loess(frame, frac, iterations, x_predict, result)
    return result
