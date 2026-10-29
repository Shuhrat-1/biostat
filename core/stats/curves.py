"""Криволинейные зависимости от одной переменной в рамках МЛР (гл. 17.1 книги).

- ``x_transform_regression`` — ПЛР по w = f(x + c): log10, ln, √, x², exp, 1/x
  (Fig. 17.1); предсказания и ДИ на исходной шкале x, сравнение всех
  допустимых преобразований по R²adj, AIC, SBC (Example 17.1A).
- ``polynomial_regression`` — полином порядка q в простых, центрированных или
  ортогональных степенях: последовательная ANOVA (Table 17.3), параметры в
  простых степенях (Table 17.4), VIF, маргинальный тест старшего члена,
  неадекватность при повторах x, кривая с ДИ на плотной сетке (Example 17.1B).
- ``trigonometric_regression`` — α + Σ_k [β_k sin(2πkt/ω) + β'_k cos(2πkt/ω)]:
  амплитуда γ = √(β² + β'²), фаза θ = atan2(−β', β) в [0, 2π) и в единицах
  времени, моменты максимума и минимума; SE γ и θ — дельта-методом
  (Example 17.2, Table 17.6).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.factorial import _new_result, _settings
from core.stats.linear_regression import parse_numbers
from core.stats.multiple_regression import goodness
from core.stats.regression_checks import _indexed_frame, _transform_column

X_TRANSFORMS = ("log10", "ln", "sqrt", "square", "exp", "reciprocal")
POLY_BASES = ("raw", "centred", "orthogonal")
_MAX_DEGREE = 10
_MAX_HARMONICS = 5
_GRID = 101
_VIF_LIMIT = 10.0

Basis = Callable[[np.ndarray], np.ndarray]


# --- Общий движок: МНК по базису f(x) -------------------------------------------


def _prepare(method: str, x: list[float], y: list[float], alpha: float, transform_kind: str,
             offset: float, n_params: int, **extra: object) -> tuple[pd.DataFrame, StatResult, bool]:
    """Пары без пропусков, трансформация отклика и проверка объёма."""
    settings = _settings(alpha, transform_kind, offset)
    frame, dropped = _indexed_frame(x, y)
    result = _new_result(method, frame, settings, **extra)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено пар с пропусками: {dropped}",
                           Severity.INFO)
    ok = _transform_column(frame, "y", transform_kind, offset, result)
    if ok and len(frame) < n_params + 1:
        result.add_warning("n_too_small", "Наблюдений должно быть больше числа параметров",
                           Severity.ERROR)
        ok = False
    return frame, result, ok


def fit_basis(design: np.ndarray, y: np.ndarray) -> dict:
    """МНК по SVD (устойчиво и для полиномов высокого порядка): β̂, cov, RSS, s²."""
    u, sv, vt = np.linalg.svd(design, full_matrices=False)
    rank = int((sv > sv[0] * 1e-12).sum())
    inv_s = np.where(np.arange(sv.size) < rank, 1 / sv, 0.0)
    beta = vt.T @ (inv_s * (u.T @ y))
    resid = y - design @ beta
    df = len(y) - design.shape[1]
    rss = float(resid @ resid)
    s2 = rss / df if df > 0 else float("nan")
    return {"beta": beta, "cov": s2 * (vt.T * inv_s ** 2) @ vt, "rss": rss, "df": df,
            "s2": s2, "rank": rank, "p": design.shape[1], "fitted": design @ beta,
            "resid": resid, "design": design}


def sequential_table(design: np.ndarray, y: np.ndarray, terms: list[tuple[str, list[int]]],
                     fit: dict) -> list[dict]:
    """Инкрементные SS групп столбцов после [1] (тип I); F против ResMS полной модели."""
    rows, cols = [], [0]
    previous = float(((y - y.mean()) ** 2).sum())
    for name, idx in terms:
        cols = cols + idx
        current = fit_basis(design[:, cols], y)["rss"]
        ss, df = previous - current, len(idx)
        f = ss / df / fit["s2"]
        rows.append({"term": f"+ {name}", "df": df, "ss": ss, "ms": ss / df, "F": f,
                     "p": float(stats.f.sf(f, df, fit["df"]))})
        previous = current
    rows.append({"term": "Residual", "df": fit["df"], "ss": fit["rss"], "ms": fit["s2"]})
    tot = float(((y - y.mean()) ** 2).sum())
    rows.append({"term": "Total", "df": len(y) - 1, "ss": tot, "ms": tot / (len(y) - 1)})
    return rows


def parameter_table(fit: dict, names: list[str], alpha: float) -> list[dict]:
    """Оценка, SE, t, P и ДИ параметров базиса."""
    se = np.sqrt(np.diag(fit["cov"]))
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = []
    for j, name in enumerate(names):
        t = fit["beta"][j] / se[j] if se[j] > 0 else float("nan")
        rows.append({"term": name, "estimate": float(fit["beta"][j]), "se": float(se[j]),
                     "t": float(t), "p": float(2 * stats.t.sf(abs(t), fit["df"])),
                     "ci_lower": float(fit["beta"][j] - t_crit * se[j]),
                     "ci_upper": float(fit["beta"][j] + t_crit * se[j])})
    return rows


def predict_rows(basis: Basis, fit: dict, points: np.ndarray, alpha: float) -> list[dict]:
    """μ̂(x) с ДИ среднего и нового наблюдения в точках x (исходная шкала)."""
    if points.size == 0:
        return []
    d = basis(points)
    mean = d @ fit["beta"]
    var = np.einsum("ij,jk,ik->i", d, fit["cov"], d)
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    se_m, se_o = np.sqrt(var), np.sqrt(var + fit["s2"])
    return [{"x": float(points[i]), "fitted": float(mean[i]), "se_mean": float(se_m[i]),
             "mean_lower": float(mean[i] - t_crit * se_m[i]),
             "mean_upper": float(mean[i] + t_crit * se_m[i]), "se_new": float(se_o[i]),
             "new_lower": float(mean[i] - t_crit * se_o[i]),
             "new_upper": float(mean[i] + t_crit * se_o[i])} for i in range(points.size)]


def curve_band(basis: Basis, fit: dict, lo: float, hi: float, alpha: float) -> dict:
    """Кривая и ДИ среднего на плотной сетке — проверка формы и переподгонки (стр. 459)."""
    rows = predict_rows(basis, fit, np.linspace(lo, hi, _GRID), alpha)
    return {"x": [r["x"] for r in rows], "fitted": [r["fitted"] for r in rows],
            "lower": [r["mean_lower"] for r in rows], "upper": [r["mean_upper"] for r in rows]}


def lack_of_fit(x: np.ndarray, y: np.ndarray, fit: dict) -> dict | None:
    """Неадекватность против чистой ошибки при повторённых x (12.8); None — нет повторов."""
    frame = pd.DataFrame({"x": x, "y": y})
    groups = frame.groupby("x")["y"]
    pe_df = len(frame) - groups.ngroups
    lof_df = fit["df"] - pe_df
    if pe_df < 1 or lof_df < 1:
        return None
    pe_ss = float(((frame["y"] - groups.transform("mean")) ** 2).sum())
    lof_ss = fit["rss"] - pe_ss
    f = (lof_ss / lof_df) / (pe_ss / pe_df)
    return {"lack_of_fit_df": lof_df, "lack_of_fit_ss": lof_ss, "pure_error_df": pe_df,
            "pure_error_ss": pe_ss, "F": float(f), "p": float(stats.f.sf(f, lof_df, pe_df))}


def _finish(frame: pd.DataFrame, fit: dict, params: list[dict], result: StatResult) -> None:
    """Качество подгонки, оценки, стандартизованные остатки и неадекватность."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    tot = float(((y - y.mean()) ** 2).sum())
    result.statistics.update({"df_resid": fit["df"], "ms_resid": fit["s2"], "n_params": fit["p"],
                              **goodness(fit["rss"], tot, len(y), fit["p"])})
    alpha = result.parameters["alpha"]
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))
    _finish_plots(frame, fit, result)
    lof = lack_of_fit(x, y, fit)
    result.parameters["lack_of_fit"] = lof
    if lof is not None:
        result.statistics.update({"F_lack_of_fit": lof["F"], "p_lack_of_fit": lof["p"]})
        if lof["p"] < alpha:
            result.add_warning("curve_lack_of_fit", "Отклонения средних по уровням x от кривой "
                               "значимы: модель не описывает зависимость (гл. 12.8).")


def _finish_plots(frame: pd.DataFrame, fit: dict, result: StatResult) -> None:
    """Стандартизованные остатки r = e/√(s²(1 − h)), h — рычаги плана (или якобиана)."""
    d = fit["design"]
    h = np.einsum("ij,jk,ik->i", d, np.linalg.pinv(d.T @ d), d)
    r = fit["resid"] / np.sqrt(fit["s2"] * np.clip(1 - h, 1e-12, None))
    n = r.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    result.parameters["plots"] = {
        "residual_vs_fitted": {"fitted": fit["fitted"].tolist(), "residuals": r.tolist()},
        "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
        "std_residuals": r.tolist(),
        "residual_vs_x": {"x": frame["x"].tolist(), "residuals": r.tolist()}}


def _model_test(rows: list[dict], n_terms: int, fit: dict, result: StatResult) -> None:
    """F всей модели: SS первых n_terms строк против ResMS."""
    ss = sum(r["ss"] for r in rows[:n_terms])
    df = sum(r["df"] for r in rows[:n_terms])
    f = ss / df / fit["s2"]
    result.p_value = float(stats.f.sf(f, df, fit["df"]))
    result.statistics.update({"F_model": float(f), "p_model": result.p_value})


def _prediction_output(basis: Basis, fit: dict, frame: pd.DataFrame, x_predict: str,
                       result: StatResult) -> None:
    """Предсказания в точках пользователя, кривая с ДИ, предупреждение об экстраполяции."""
    alpha = result.parameters["alpha"]
    lo, hi = float(frame["x"].min()), float(frame["x"].max())
    points = np.asarray(parse_numbers(x_predict), float)
    result.parameters["predictions"] = predict_rows(basis, fit, points, alpha)
    result.parameters["curve"] = curve_band(basis, fit, lo, hi, alpha)
    result.parameters["points"] = {"x": frame["x"].tolist(), "y": frame["y"].tolist()}
    outside = [float(p) for p in points if not lo <= p <= hi]
    if outside:
        result.add_warning("curve_extrapolation", "Предсказание вне диапазона x "
                           f"[{lo:g}; {hi:g}]: {', '.join(f'{p:g}' for p in outside)}. "
                           "Форма кривой за пределами данных ничем не ограничена.",
                           params={"points": ", ".join(f"{p:g}" for p in outside),
                                   "low": f"{lo:g}", "high": f"{hi:g}"})


# --- x_transform_regression -----------------------------------------------------


def x_transform(values: np.ndarray, kind: str, offset: float = 0.0) -> np.ndarray:
    """w = f(x + c) для преобразований Fig. 17.1; ValueError вне области."""
    v = np.asarray(values, float) + offset
    if kind in ("log10", "ln", "reciprocal") and np.any(v <= 0):
        raise ValueError(f"{kind}: нужно x + c > 0")
    if kind == "sqrt" and np.any(v < 0):
        raise ValueError("sqrt: нужно x + c ≥ 0")
    funcs = {"log10": np.log10, "ln": np.log, "sqrt": np.sqrt, "square": np.square,
             "exp": np.exp, "reciprocal": np.reciprocal}
    if kind not in funcs:
        raise ValueError(f"x_transform ∈ {X_TRANSFORMS}")
    return funcs[kind](v)


def _transform_basis(kind: str, offset: float) -> Basis:
    """[1, f(x + c)] как функция исходного x."""
    return lambda x: np.column_stack([np.ones(len(x)), x_transform(x, kind, offset)])


def _transform_alternatives(frame: pd.DataFrame, offset: float, chosen: str) -> list[dict]:
    """R²adj, AIC, SBC для всех преобразований, допустимых на данных (стр. 452)."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    tot = float(((y - y.mean()) ** 2).sum())
    rows = []
    for kind in ("none", *X_TRANSFORMS):
        try:
            design = (np.column_stack([np.ones(len(x)), x]) if kind == "none"
                      else _transform_basis(kind, offset)(x))
        except ValueError:
            continue
        if not np.all(np.isfinite(design)) or np.ptp(design[:, 1]) == 0:
            continue
        fit = fit_basis(design, y)
        rows.append({"x_transform": kind, "chosen": kind == chosen, "ms_resid": fit["s2"],
                     **goodness(fit["rss"], tot, len(y), 2)})
    return rows


def _transform_design(basis: Basis, frame: pd.DataFrame, result: StatResult) -> np.ndarray | None:
    """План [1, w]; None, если x + c вне области или w не варьирует."""
    try:
        design = basis(frame["x"].to_numpy(float))
    except ValueError as exc:
        result.add_warning("x_transform_domain", f"x: {exc}", Severity.ERROR)
        return None
    if not np.all(np.isfinite(design)) or np.ptp(design[:, 1]) == 0:
        result.add_warning("no_x_variation", "Преобразованный x не варьирует (или бесконечен)",
                           Severity.ERROR)
        return None
    return design


def x_transform_regression(
    x: list[float],
    y: list[float],
    x_transform_kind: str = "log10",
    x_offset: float = 0.0,
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """ПЛР по преобразованному x; ДИ предсказаний одинаковы на шкалах w и x (Example 17.1A)."""
    if x_transform_kind not in X_TRANSFORMS:
        raise ValueError(f"x_transform_kind ∈ {X_TRANSFORMS}")
    frame, result, ok = _prepare("x_transform_regression", x, y, alpha, transform_kind, offset,
                                 3, x_transform=x_transform_kind, x_offset=x_offset)
    if not ok:
        return result
    basis = _transform_basis(x_transform_kind, x_offset)
    design = _transform_design(basis, frame, result)
    if design is None:
        return result
    y_arr = frame["y"].to_numpy(float)
    fit = fit_basis(design, y_arr)
    rows = sequential_table(design, y_arr, [(f"{x_transform_kind}(x)", [1])], fit)
    params = parameter_table(fit, ["[1]", f"{x_transform_kind}(x)"], alpha)
    result.parameters.update({"anova_table": rows, "parameter_tests": params,
                              "alternatives": _transform_alternatives(frame, x_offset,
                                                                      x_transform_kind),
                              "w": design[:, 1].tolist()})
    _model_test(rows, 1, fit, result)
    _finish(frame, fit, params, result)
    _prediction_output(basis, fit, frame, x_predict, result)
    return result


# --- polynomial_regression ------------------------------------------------------


def polynomial_basis(x: np.ndarray, degree: int, kind: str) -> Basis:
    """Базис [1, φ1(x) … φq(x)]: простые степени, центрированные или ортогональные (стр. 458)."""
    centre = float(np.mean(x)) if kind != "raw" else 0.0

    def powers(v: np.ndarray) -> np.ndarray:
        return np.vander(np.asarray(v, float) - centre, degree + 1, increasing=True)

    if kind != "orthogonal":
        return powers
    # Q = V R⁻¹ ортонормирован на данных; тот же R⁻¹ переводит новые точки.
    _, r = np.linalg.qr(powers(x))
    r_inv = np.linalg.inv(r)
    return lambda v: powers(v) @ r_inv


def _power_names(degree: int, kind: str) -> list[str]:
    """Имена столбцов: [1], x, x^2 … или (x − x̄)^k, P1 … Pq."""
    if kind == "orthogonal":
        return ["[1]", *[f"P{k}" for k in range(1, degree + 1)]]
    base = "x" if kind == "raw" else "(x - mean)"
    return ["[1]", base, *[f"{base}^{k}" for k in range(2, degree + 1)]]


def vif_from_design(design: np.ndarray, names: list[str]) -> list[dict]:
    """VIF_j = j-й диагональный элемент обратной корреляционной матрицы (стр. 362)."""
    if design.shape[1] < 3:
        return []
    corr = np.corrcoef(design[:, 1:], rowvar=False)
    vif = np.diag(np.linalg.pinv(corr))
    return [{"variable": n, "vif": float(v)} for n, v in zip(names[1:], vif)]


def _check_degree(frame: pd.DataFrame, degree: int, result: StatResult) -> bool:
    """Порядок < числа различных x; k − 1 проходит через средние (переподгонка, стр. 459)."""
    distinct = frame["x"].nunique()
    if degree >= distinct:
        result.add_warning("poly_degree_too_high", f"Порядок {degree} требует не меньше "
                           f"{degree + 1} различных значений x (есть {distinct}).",
                           Severity.ERROR, params={"degree": degree, "distinct": distinct})
        return False
    if degree == distinct - 1:
        result.add_warning("poly_interpolates", "Полином порядка k − 1 проходит через средние "
                           "при каждом x: переподгонка, интерполяция ненадёжна.",
                           params={"degree": degree})
    return True


def _poly_warnings(rows: list[dict], vifs: list[dict], result: StatResult) -> None:
    """Старший член незначим — понизить порядок; VIF > 10 — центрировать/ортогонализовать."""
    top = rows[len(rows) - 3]
    if top["p"] > result.parameters["alpha"]:
        result.add_warning("poly_top_not_significant", f"Старший член {top['term'][2:]} "
                           f"незначим (P = {top['p']:.3f}): достаточно порядка на 1 ниже.",
                           params={"term": top["term"][2:], "p": round(top["p"], 4)})
    high = [f"{v['variable']} ({v['vif']:.0f})" for v in vifs if v["vif"] > _VIF_LIMIT]
    if high:
        result.add_warning("poly_collinearity", "VIF > 10: " + ", ".join(high) + ". Центрируй x "
                           "или используй ортогональные полиномы.",
                           params={"variables": ", ".join(high)})


def polynomial_regression(
    x: list[float],
    y: list[float],
    degree: int = 2,
    basis: str = "raw",
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Полином порядка q: степени добавляются по возрастанию, младшие сохраняются (17.1.2)."""
    degree = int(degree)
    if not 1 <= degree <= _MAX_DEGREE or basis not in POLY_BASES:
        raise ValueError(f"degree ∈ [1, {_MAX_DEGREE}], basis ∈ {POLY_BASES}")
    frame, result, ok = _prepare("polynomial_regression", x, y, alpha, transform_kind, offset,
                                 degree + 2, degree=degree, basis=basis)
    if not ok or not _check_degree(frame, degree, result):
        return result
    x_arr, y_arr = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    phi = polynomial_basis(x_arr, degree, basis)
    design = phi(x_arr)
    fit = fit_basis(design, y_arr)
    names = _power_names(degree, basis)
    rows = sequential_table(design, y_arr, [(n, [k]) for k, n in enumerate(names[1:], 1)], fit)
    raw_fit = fit_basis(polynomial_basis(x_arr, degree, "raw")(x_arr), y_arr)
    raw_params = parameter_table(raw_fit, _power_names(degree, "raw"), alpha)
    vifs = vif_from_design(design, names)
    result.parameters.update({
        "anova_table": rows, "parameter_tests": parameter_table(fit, names, alpha),
        "raw_parameters": raw_params, "vif": vifs, "x_mean": float(x_arr.mean())})
    _model_test(rows, degree, fit, result)
    _finish(frame, raw_fit, raw_params, result)
    _poly_warnings(rows, vifs, result)
    _prediction_output(phi, fit, frame, x_predict, result)
    return result


# --- trigonometric_regression ---------------------------------------------------


def trig_basis(period: float, harmonics: int) -> Basis:
    """[1, sin(2πkt/ω), cos(2πkt/ω)] для k = 1 … H (уравнение 17.1)."""
    def basis(t: np.ndarray) -> np.ndarray:
        arg = 2 * np.pi * np.asarray(t, float)[:, None] * np.arange(1, harmonics + 1) / period
        cols = np.empty((arg.shape[0], 2 * harmonics))
        cols[:, 0::2], cols[:, 1::2] = np.sin(arg), np.cos(arg)
        return np.column_stack([np.ones(arg.shape[0]), cols])
    return basis


def amplitude_phase(b_sin: float, b_cos: float, cov: np.ndarray) -> dict:
    """γ = √(β1² + β2²), θ = atan2(−β2, β1) ∈ [0, 2π); SE дельта-методом (стр. 461)."""
    gamma = float(np.hypot(b_sin, b_cos))
    theta = float(np.arctan2(-b_cos, b_sin) % (2 * np.pi))
    g_gamma = np.array([b_sin, b_cos]) / gamma
    g_theta = np.array([b_cos, -b_sin]) / gamma ** 2
    return {"amplitude": gamma, "se_amplitude": float(np.sqrt(g_gamma @ cov @ g_gamma)),
            "phase_rad": theta, "se_phase_rad": float(np.sqrt(g_theta @ cov @ g_theta))}


def _harmonic_rows(fit: dict, period: float, harmonics: int) -> list[dict]:
    """Амплитуда и фаза каждой гармоники; фаза в единицах времени = θω/(2πk)."""
    rows = []
    for k in range(1, harmonics + 1):
        j = 2 * k - 1
        ap = amplitude_phase(fit["beta"][j], fit["beta"][j + 1], fit["cov"][j:j + 2, j:j + 2])
        scale = period / (2 * np.pi * k)
        rows.append({"harmonic": k, **ap, "phase_time": ap["phase_rad"] * scale,
                     "se_phase_time": ap["se_phase_rad"] * scale})
    return rows


def _extremes(basis: Basis, fit: dict, period: float, first: dict, harmonics: int) -> dict:
    """Моменты max/min: для H = 1 — ω/4 + фаза и 3ω/4 + фаза (стр. 462), иначе по сетке."""
    if harmonics == 1:
        t_max = (period / 4 + first["phase_time"]) % period
        t_min = (3 * period / 4 + first["phase_time"]) % period
        se_t = first["se_phase_time"]
    else:
        grid = np.linspace(0, period, 3601)[:-1]
        curve = basis(grid) @ fit["beta"]
        t_max, t_min, se_t = float(grid[curve.argmax()]), float(grid[curve.argmin()]), None
    values = basis(np.array([t_max, t_min])) @ fit["beta"]
    return {"time_max": float(t_max), "value_max": float(values[0]), "time_min": float(t_min),
            "value_min": float(values[1]), "se_time": se_t}


def _trig_tables(basis: Basis, fit: dict, y: np.ndarray, period: float, harmonics: int,
                 result: StatResult) -> list[dict]:
    """ANOVA по гармоникам (2 df), параметры, амплитуды и фазы, экстремумы."""
    names = ["[1]", *[f"{f}{k}" for k in range(1, harmonics + 1) for f in ("Sin", "Cos")]]
    terms = [(f"harmonic {k}", [2 * k - 1, 2 * k]) for k in range(1, harmonics + 1)]
    rows = sequential_table(fit["design"], y, terms, fit)
    params = parameter_table(fit, names, result.parameters["alpha"])
    harm = _harmonic_rows(fit, period, harmonics)
    result.parameters.update({"anova_table": rows, "parameter_tests": params,
                              "harmonics_table": harm,
                              "extremes": _extremes(basis, fit, period, harm[0], harmonics)})
    _model_test(rows, harmonics, fit, result)
    result.statistics.update({"mean_level": float(fit["beta"][0]),
                              "amplitude": harm[0]["amplitude"],
                              "phase_time": harm[0]["phase_time"]})
    return params


def profile_rss(t: np.ndarray, y: np.ndarray, period: float, harmonics: int) -> float:
    """RSS линейной модели при фиксированном ω (модель частично линейна)."""
    return fit_basis(trig_basis(period, harmonics)(t), y)["rss"]


def _free_period(t: np.ndarray, y: np.ndarray, guess: float, harmonics: int,
                 result: StatResult) -> float:
    """ω̂: профиль RSS по сетке [ω₀/2, 2ω₀], затем уточнение (стр. 462 — модель нелинейна)."""
    grid = np.geomspace(guess / 2, guess * 2, 801)
    rss = np.array([profile_rss(t, y, w, harmonics) for w in grid])
    k = int(np.argmin(rss))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    best = optimize.minimize_scalar(lambda w: profile_rss(t, y, w, harmonics), bounds=(lo, hi),
                                    method="bounded", options={"xatol": 1e-10 * guess})
    if k in (0, grid.size - 1):
        result.add_warning("trig_period_bound", "Оценка периода на границе поиска [ω₀/2; 2ω₀]: "
                           "задай другое начальное ω₀.")
    result.parameters["period_fixed_rss"] = profile_rss(t, y, guess, harmonics)
    return float(best.x)


def _with_period_uncertainty(fit: dict, t: np.ndarray, y: np.ndarray, period: float,
                             harmonics: int, result: StatResult) -> dict:
    """Ковариация (α, β…, ω) по якобиану полной нелинейной модели; SE и df учитывают ω̂."""
    arg = 2 * np.pi * t[:, None] * np.arange(1, harmonics + 1) / period
    b_sin, b_cos = fit["beta"][1::2], fit["beta"][2::2]
    d_omega = ((b_sin * np.cos(arg) - b_cos * np.sin(arg)) * (-arg / period)).sum(axis=1)
    jac = np.column_stack([fit["design"], d_omega])
    df = len(y) - jac.shape[1]
    s2 = fit["rss"] / df
    cov = s2 * np.linalg.pinv(jac.T @ jac)
    se = float(np.sqrt(cov[-1, -1]))
    crit = float(stats.t.ppf(1 - result.parameters["alpha"] / 2, df))
    f = (result.parameters.pop("period_fixed_rss") - fit["rss"]) / s2
    result.parameters["period_estimate"] = {
        "estimate": period, "se": se, "ci_lower": period - crit * se, "ci_upper": period + crit * se,
        "df": df, "F_vs_fixed": float(f), "p_vs_fixed": float(stats.f.sf(f, 1, df))}
    result.statistics["period"] = period
    return {**fit, "cov": cov[:-1, :-1], "df": df, "s2": s2}


def trigonometric_regression(
    x: list[float],
    y: list[float],
    period: float = 12.0,
    harmonics: int = 1,
    x_predict: str = "",
    estimate_period: str = "no",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Периодическая регрессия; период ω известен или оценивается (17.1.3, стр. 462)."""
    harmonics = int(harmonics)
    if period <= 0 or not 1 <= harmonics <= _MAX_HARMONICS or estimate_period not in ("no", "yes"):
        raise ValueError(f"period > 0, harmonics ∈ [1, {_MAX_HARMONICS}], estimate_period ∈ "
                         "(no, yes)")
    extra = 1 if estimate_period == "yes" else 0
    frame, result, ok = _prepare("trigonometric_regression", x, y, alpha, transform_kind,
                                 offset, 2 * harmonics + 2 + extra, period=period,
                                 harmonics=harmonics, estimate_period=estimate_period)
    if not ok:
        return result
    t, y_arr = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    if extra:
        period = _free_period(t, y_arr, period, harmonics, result)
    basis = trig_basis(period, harmonics)
    design = basis(t)
    fit = fit_basis(design, y_arr)
    if extra and fit["rank"] == fit["p"]:
        fit = _with_period_uncertainty(fit, t, y_arr, period, harmonics, result)
    if fit["rank"] < fit["p"]:
        result.add_warning("trig_rank", "sin и cos линейно зависимы на этих временах: мало "
                           "различных фаз для заданного числа гармоник.", Severity.ERROR)
        return result
    params = _trig_tables(basis, fit, y_arr, period, harmonics, result)
    _finish(frame, fit, params, result)
    _prediction_output(basis, fit, frame, x_predict, result)
    result.parameters["curve"] = curve_band(basis, fit, 0.0, float(max(period, t.max())), alpha)
    return result
