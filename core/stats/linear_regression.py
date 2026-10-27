"""Модели для одной объясняющей переменной (гл. 12 книги).

- ``regression_details`` — ПЛР или регрессия через начало координат:
  таблица ANOVA (12.3, Tables 12.3, 12.8), t-тесты параметров против
  заданных значений (12.4), предсказания среднего и нового наблюдения
  (12.5), R² и adjusted R² или R²emp (12.6, 12.9.2), центрированная и
  стандартизованная параметризация (12.9.1), проверка неадекватности
  по чистой ошибке при повторённых x (12.8, Table 12.7).
- ``inverse_prediction`` — калибровка: x по новому y с пределами
  Филлера (12.9.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.blocking import _fill_residuals
from core.stats.factorial import _apply_transform, _new_result, _settings

_YES_NO = ("no", "yes")
_SCALINGS = ("none", "centred", "standardized")
_MIN_LOF_LEVELS = 3  # неадекватность проверяема, если уровней x больше параметров


# --- Общие части --------------------------------------------------------------


def parse_numbers(text: str) -> list[float]:
    """Числа через «;» или пробел; запятая допускается как десятичный знак."""
    parts = [p for chunk in (text or "").split(";") for p in chunk.split()]
    try:
        return [float(p.replace(",", ".")) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Не удалось разобрать числа: {text!r}") from exc


def _xy_frame(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> tuple[pd.DataFrame, int]:
    """Пары (x, y) без пропусков и число удалённых пар."""
    ax = np.asarray(x, dtype=float).ravel()
    ay = np.asarray(y, dtype=float).ravel()
    if ax.size != ay.size:
        raise ValueError("x и y должны быть одной длины")
    frame = pd.DataFrame({"x": ax, "y": ay})
    clean = frame.dropna().reset_index(drop=True)
    return clean, len(frame) - len(clean)


def _guard(frame: pd.DataFrame, dropped: int, origin: bool, result: StatResult) -> bool:
    """Достаточно ли точек и варьирует ли x."""
    if dropped:
        result.add_warning("missing_dropped", f"Удалено пар с пропусками: {dropped}",
                           Severity.INFO)
    if len(frame) < (2 if origin else 3):
        result.add_warning("n_too_small", "Слишком мало точек для регрессии", Severity.ERROR)
        return False
    flat = not frame["x"].any() if origin else np.ptp(frame["x"]) == 0
    if flat:
        result.add_warning("no_x_variation", "x не варьирует — наклон не определён",
                           Severity.ERROR)
        return False
    return True


def _summary(frame: pd.DataFrame, origin: bool) -> dict:
    """Суммы квадратов (исправленные или нет), оценки, s², ковариация (стр. 296–299, 315)."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    n = x.size
    xc, yc = (x, y) if origin else (x - x.mean(), y - y.mean())
    sxx, sxy, syy = float(xc @ xc), float(xc @ yc), float(yc @ yc)
    beta = sxy / sxx
    alpha = 0.0 if origin else float(y.mean() - beta * x.mean())
    df_res = n - 1 if origin else n - 2
    res_ss = syy - sxy ** 2 / sxx
    s2 = res_ss / df_res
    return {"n": n, "x_mean": float(x.mean()), "y_mean": float(y.mean()), "sxx": sxx,
            "sxy": sxy, "syy": syy, "beta": beta, "alpha": alpha, "df_res": df_res,
            "mod_ss": sxy ** 2 / sxx, "res_ss": res_ss, "s2": s2, "origin": origin,
            "tot_ss_corrected": float(((y - y.mean()) ** 2).sum())}


def _mean_variance(fit: dict, x0: np.ndarray) -> np.ndarray:
    """Var(μ̂(x0)) = s²(1/N + (x0 − x̄)²/SSxx) или s²·x0²/USSxx (стр. 302, 315)."""
    if fit["origin"]:
        return fit["s2"] * x0 ** 2 / fit["sxx"]
    return fit["s2"] * (1 / fit["n"] + (x0 - fit["x_mean"]) ** 2 / fit["sxx"])


# --- regression_details -------------------------------------------------------


def _anova(fit: dict) -> list[dict]:
    """Model / Residual / Total; F = ModMS/ResMS на (1, ResDF) (Tables 12.2, 12.8)."""
    res_ms = fit["s2"]
    f = fit["mod_ss"] / res_ms
    total_df = fit["n"] if fit["origin"] else fit["n"] - 1
    return [
        {"term": "Model", "stratum": "unit", "df": 1, "ss": fit["mod_ss"], "ms": fit["mod_ss"],
         "F": f, "p": float(stats.f.sf(f, 1, fit["df_res"]))},
        {"term": "Residual", "stratum": "unit", "df": fit["df_res"], "ss": fit["res_ss"],
         "ms": res_ms},
        {"term": "Total", "stratum": "unit", "df": total_df, "ss": fit["syy"]},
    ]


def _parameter_rows(fit: dict, nulls: dict[str, float], alpha: float) -> list[dict]:
    """Оценка, SE, t = (θ̂ − c)/SE, P и ДИ для α и β (стр. 300–301)."""
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df_res"]))
    se = {"slope": np.sqrt(fit["s2"] / fit["sxx"])}
    values = {"slope": fit["beta"]}
    if not fit["origin"]:
        se["intercept"] = np.sqrt(fit["s2"] * (1 / fit["n"] + fit["x_mean"] ** 2 / fit["sxx"]))
        values["intercept"] = fit["alpha"]
    rows = []
    for name in ("intercept", "slope"):
        if name not in values:
            continue
        t = (values[name] - nulls[name]) / se[name]
        rows.append({"parameter": name, "estimate": float(values[name]), "se": float(se[name]),
                     "null": nulls[name], "t": float(t),
                     "p": float(2 * stats.t.sf(abs(t), fit["df_res"])),
                     "ci_lower": float(values[name] - t_crit * se[name]),
                     "ci_upper": float(values[name] + t_crit * se[name])})
    return rows


def _goodness(fit: dict) -> dict:
    """R² и adjusted R² (стр. 305–306) или R²emp = 1 − ResUSS/TotSS (стр. 316)."""
    tot = fit["tot_ss_corrected"]
    if fit["origin"]:
        return {"r_squared_emp": 1 - fit["res_ss"] / tot if tot > 0 else float("nan")}
    r2 = fit["mod_ss"] / fit["syy"]
    adj = 1 - fit["s2"] / (fit["syy"] / (fit["n"] - 1))
    return {"r_squared": r2, "adj_r_squared": adj}


def _scaled(fit: dict, scaling: str) -> dict | None:
    """α* = ȳ; β* = s_x·β̂, SE(β*) = s_x·SE(β̂) (стр. 313–314)."""
    if scaling == "none" or fit["origin"]:
        return None
    se_slope = float(np.sqrt(fit["s2"] / fit["sxx"]))
    se_icept = float(np.sqrt(fit["s2"] / fit["n"]))
    out = {"scaling": scaling, "intercept": fit["y_mean"], "intercept_se": se_icept,
           "slope": fit["beta"], "slope_se": se_slope}
    if scaling == "standardized":
        s_x = float(np.sqrt(fit["sxx"] / (fit["n"] - 1)))
        out.update({"sd_x": s_x, "slope": fit["beta"] * s_x, "slope_se": se_slope * s_x})
    return out


def _predictions(fit: dict, points: list[float], alpha: float) -> list[dict]:
    """μ̂(x) с ДИ среднего и ДИ нового наблюдения (s² добавляется) (стр. 302–305)."""
    if not points:
        return []
    x0 = np.asarray(points, float)
    mean = fit["alpha"] + fit["beta"] * x0
    var_mean = _mean_variance(fit, x0)
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df_res"]))
    se_m, se_o = np.sqrt(var_mean), np.sqrt(var_mean + fit["s2"])
    return [{"x": float(x0[i]), "fitted": float(mean[i]), "se_mean": float(se_m[i]),
             "mean_lower": float(mean[i] - t_crit * se_m[i]),
             "mean_upper": float(mean[i] + t_crit * se_m[i]), "se_new": float(se_o[i]),
             "new_lower": float(mean[i] - t_crit * se_o[i]),
             "new_upper": float(mean[i] + t_crit * se_o[i])} for i in range(x0.size)]


def _lack_of_fit(frame: pd.DataFrame, fit: dict, rows: list[dict], result: StatResult) -> None:
    """Делит ResSS на неадекватность и чистую ошибку; F против PEMS (стр. 309–312, Table 12.7)."""
    groups = frame.groupby("x")["y"]
    levels = groups.ngroups
    pe_df = len(frame) - levels
    if fit["origin"] or levels < _MIN_LOF_LEVELS or pe_df < 1:
        result.add_warning("lack_of_fit_unavailable", "Проверка неадекватности требует "
                           "модели с α, ≥ 3 уровней x и повторённых значений x.")
        return
    pe_ss = float(((frame["y"] - groups.transform("mean")) ** 2).sum())
    lof_ss, lof_df = fit["res_ss"] - pe_ss, levels - 2
    pe_ms = pe_ss / pe_df
    f_m, f_l = rows[0]["ms"] / pe_ms, lof_ss / lof_df / pe_ms
    rows[0].update({"F": f_m, "p": float(stats.f.sf(f_m, 1, pe_df))})
    rows[1:2] = [
        {"term": "Residual", "stratum": "unit", "df": fit["df_res"], "ss": fit["res_ss"],
         "ms": fit["s2"]},
        {"term": "Lack of fit", "stratum": "residual", "df": lof_df, "ss": lof_ss,
         "ms": lof_ss / lof_df, "F": f_l, "p": float(stats.f.sf(f_l, lof_df, pe_df))},
        {"term": "Pure error", "stratum": "residual", "df": pe_df, "ss": pe_ss, "ms": pe_ms},
    ]
    result.statistics.update({
        "F_lack_of_fit": f_l, "p_lack_of_fit": rows[2]["p"], "ms_pure_error": pe_ms,
        "adj_r_squared_groups": 1 - pe_ms / (fit["syy"] / (fit["n"] - 1))})
    if rows[2]["p"] < result.parameters["alpha"]:
        result.add_warning("lack_of_fit", "Отклонения средних по уровням x от прямой "
                           "значимы: прямая не описывает зависимость (гл. 12.8).")


def _fill_statistics(fit: dict, rows: list[dict], params: list[dict], result: StatResult) -> None:
    """Плоские статистики, оценки параметров и таблицы в результат."""
    model = rows[0]
    result.p_value = model["p"]
    result.statistics.update({
        "F_model": model["F"], "p_model": model["p"], "df_resid": fit["df_res"],
        "ms_resid": fit["s2"], "ss_resid": fit["res_ss"], "ss_total": fit["syy"],
        "ss_xx": fit["sxx"], "ss_xy": fit["sxy"], "x_mean": fit["x_mean"],
        "y_mean": fit["y_mean"], **_goodness(fit)})
    for row in params:
        result.statistics[f"t_{row['parameter']}"] = row["t"]
        result.statistics[f"p_{row['parameter']}"] = row["p"]
        result.estimates.append(Estimate(
            name=row["parameter"], value=row["estimate"], ci_lower=row["ci_lower"],
            ci_upper=row["ci_upper"], ci_level=1 - result.parameters["alpha"],
            std_error=row["se"]))
    result.parameters.update({"anova_table": rows, "parameter_tests": params,
                              "line": {"intercept": fit["alpha"], "slope": fit["beta"]}})


def regression_details(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    through_origin: str = "no",
    slope_null: float = 0.0,
    intercept_null: float = 0.0,
    x_predict: str = "",
    scaling: str = "none",
    lack_of_fit: str = "no",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Подробный анализ модели y = α + βx (или y = βx) по главе 12."""
    settings = _settings(alpha, transform_kind, offset)
    if {through_origin, lack_of_fit} - set(_YES_NO) or scaling not in _SCALINGS:
        raise ValueError(f"through_origin, lack_of_fit ∈ {_YES_NO}, scaling ∈ {_SCALINGS}")
    origin = through_origin == "yes"
    frame, dropped = _xy_frame(x, y)
    result = _new_result("regression_details", frame, settings, through_origin=origin)
    if not _guard(frame, dropped, origin, result) or not _apply_transform(
            frame, transform_kind, offset, result):
        return result
    fit = _summary(frame, origin)
    result.parameters["points"] = {"x": frame["x"].tolist(), "y": frame["y"].tolist()}
    rows = _anova(fit)
    params = _parameter_rows(fit, {"slope": slope_null, "intercept": intercept_null}, alpha)
    _fill_statistics(fit, rows, params, result)
    if lack_of_fit == "yes":
        _lack_of_fit(frame, fit, rows, result)
    result.parameters["predictions"] = _predictions(fit, parse_numbers(x_predict), alpha)
    _flag_extrapolation(frame, result)
    result.parameters["scaled"] = _scaled(fit, scaling)
    design = frame[["x"]] if origin else sm.add_constant(frame[["x"]])
    _fill_residuals(sm.OLS(frame["y"], design).fit(), frame, result)
    return result


def _flag_extrapolation(frame: pd.DataFrame, result: StatResult) -> None:
    """Предупредить о предсказаниях вне наблюдённого диапазона x (стр. 304)."""
    lo, hi = float(frame["x"].min()), float(frame["x"].max())
    outside = [p["x"] for p in result.parameters["predictions"] if not lo <= p["x"] <= hi]
    if outside:
        result.add_warning(
            "extrapolation", f"Предсказание вне диапазона x [{lo:g}; {hi:g}]: "
            f"{', '.join(f'{v:g}' for v in outside)} — форма связи там не проверена.",
            params={"low": f"{lo:g}", "high": f"{hi:g}", "values": ", ".join(f"{v:g}" for v in outside)})


# --- Калибровка ---------------------------------------------------------------


def fieller_limits(fit: dict, y_new: float, replicates: int, alpha: float) -> dict:
    """x̂ = (ȳ_new − α̂)/β̂ и пределы Филлера; существуют при g < 1 (стр. 321)."""
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df_res"]))
    se_beta = np.sqrt(fit["s2"] / fit["sxx"])
    g = float((t_crit / (fit["beta"] / se_beta)) ** 2)
    x_hat = (y_new - fit["alpha"]) / fit["beta"]
    row = {"y_new": y_new, "x_hat": float(x_hat), "g": g,
           "lower": float("nan"), "upper": float("nan")}
    if g >= 1:
        return row
    d = x_hat - fit["x_mean"]
    centre = x_hat + g * d / (1 - g)
    half = (np.sqrt(fit["s2"]) * t_crit / (fit["beta"] * (1 - g))) * np.sqrt(
        d ** 2 / fit["sxx"] + (1 - g) * (1 / replicates + 1 / fit["n"]))
    row["lower"], row["upper"] = sorted((float(centre - half), float(centre + half)))
    return row


def inverse_prediction(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    y_new: str = "",
    replicates: int = 1,
    alpha: float = 0.05,
) -> StatResult:
    """Калибровка: оценить x по средним новых измерений y (r повторов каждое)."""
    settings = _settings(alpha, "none", 0.0)
    if int(replicates) < 1:
        raise ValueError("replicates ≥ 1")
    frame, dropped = _xy_frame(x, y)
    result = _new_result("inverse_prediction", frame, settings, replicates=int(replicates))
    values = parse_numbers(y_new)
    if not values:
        result.add_warning("calibration_no_values", "Укажи новые значения y через «;».",
                           Severity.ERROR)
        return result
    if not _guard(frame, dropped, False, result):
        return result
    fit = _summary(frame, False)
    result.parameters["points"] = {"x": frame["x"].tolist(), "y": frame["y"].tolist()}
    result.parameters["line"] = {"intercept": fit["alpha"], "slope": fit["beta"]}
    rows = [fieller_limits(fit, v, int(replicates), alpha) for v in values]
    _calibration_output(fit, rows, result)
    lo, hi = float(frame["x"].min()), float(frame["x"].max())
    outside = [f"{r['x_hat']:g}" for r in rows if not lo <= r["x_hat"] <= hi]
    if outside:
        result.add_warning("extrapolation", f"x̂ вне диапазона калибровки [{lo:g}; {hi:g}]: "
                           f"{', '.join(outside)}.",
                           params={"low": f"{lo:g}", "high": f"{hi:g}", "values": ", ".join(outside)})
    return result


def _calibration_output(fit: dict, rows: list[dict], result: StatResult) -> None:
    """Таблица калибровки, оценки x̂ и предупреждения о g ≥ 1 и экстраполяции."""
    level = 1 - result.parameters["alpha"]
    result.parameters["calibration"] = rows
    result.statistics.update({"slope": fit["beta"], "intercept": fit["alpha"],
                              "ms_resid": fit["s2"], "df_resid": fit["df_res"],
                              "g": rows[0]["g"]})
    for row in rows:
        lower, upper = (None if np.isnan(row[k]) else row[k] for k in ("lower", "upper"))
        result.estimates.append(Estimate(name=f"x({row['y_new']:g})", value=row["x_hat"],
                                         ci_lower=lower, ci_upper=upper, ci_level=level))
    if rows[0]["g"] >= 1:
        result.add_warning("fieller_undefined", "Наклон не отличается значимо от нуля "
                           "(g ≥ 1): пределы Филлера не существуют.",
                           params={"g": round(rows[0]["g"], 4)})
