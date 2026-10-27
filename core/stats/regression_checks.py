"""Проверка подгонки регрессии (гл. 13 книги).

- ``influence_measures`` — для любой матрицы плана: рычаги h_ii, простые,
  стандартизованные, предсказательные и удалённые остатки, Cook D и
  модифицированная статистика Кука C_i (13.2, 13.4).
- ``regression_diagnostics`` — метод каталога для ПЛР (или через начало):
  таблица по наблюдениям, пороги 2p/N, |r(i)| > 2, C_i > 2√((N − p)/N),
  перерасчёт без влиятельных точек, данные графиков (Figs 13.3, 13.5, 13.7).
- ``regression_cross_validation`` — разбиение данных, leave-1-out и
  k-fold: PB, MAD, MSEP, RMSE и их доли от среднего отклика (13.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.factorial import _NONE, _new_result, _settings
from core.stats.transforms import TRANSFORMS, transform

_YES_NO = ("no", "yes")
_SCHEMES = ("split", "loo", "kfold")
_RESID_CUTOFF = 2.0


# --- Расчёт мер влияния -------------------------------------------------------


def _ols(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """МНК: параметры, ковариация (без s²), s², ResDF."""
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    df = design.shape[0] - design.shape[1]
    s2 = float(((y - design @ beta) ** 2).sum() / df)
    return beta, xtx_inv, s2, df


def influence_measures(design: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Остатки всех видов, рычаги и статистики Кука по формулам стр. 333–336."""
    n, p = design.shape
    beta, xtx_inv, s2, _ = _ols(design, y)
    h = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    fitted = design @ beta
    e = y - fitted
    r = e / np.sqrt(s2 * (1 - h))
    s2_del = s2 * (n - p - r ** 2) / (n - p - 1)
    r_del = r * np.sqrt((n - p - 1) / (n - p - r ** 2))
    return {"beta": beta, "s2": s2, "fitted": fitted, "residual": e, "leverage": h,
            "std_residual": r, "prediction_residual": e / (1 - h), "s2_deleted": s2_del,
            "deletion_residual": r_del, "cook_d": r ** 2 / p * h / (1 - h),
            "modified_cook": np.sqrt((n - p) * r_del ** 2 / p * h / (1 - h))}


def half_normal_quantiles(n: int) -> np.ndarray:
    """Квантили полунормального графика: [50 + 50(i − 0.375)/(n + 0.25)]-й процентиль (стр. 104)."""
    return stats.norm.ppf(0.5 + 0.5 * (np.arange(1, n + 1) - 0.375) / (n + 0.25))


# --- Подготовка данных --------------------------------------------------------


def _indexed_frame(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> tuple[pd.DataFrame, int]:
    """Пары без пропусков с исходным номером строки (с 1)."""
    ax = np.asarray(x, dtype=float).ravel()
    ay = np.asarray(y, dtype=float).ravel()
    if ax.size != ay.size:
        raise ValueError("x и y должны быть одной длины")
    frame = pd.DataFrame({"row": np.arange(1, ax.size + 1), "x": ax, "y": ay})
    clean = frame.dropna().reset_index(drop=True)
    return clean, len(frame) - len(clean)


def _transform_column(frame: pd.DataFrame, column: str, kind: str, offset: float,
                      result: StatResult) -> bool:
    """Трансформировать x или y (Table 6.4); вне области — ошибка."""
    if kind == _NONE:
        return True
    try:
        frame[column] = transform(frame[column].to_numpy(float), kind, offset)
    except ValueError as exc:
        result.add_warning("transform_domain", f"{column}: {exc}", Severity.ERROR)
        return False
    return True


def _prepare(method: str, x: np.ndarray | list[float], y: np.ndarray | list[float],
             options: dict) -> tuple[pd.DataFrame, StatResult, bool]:
    """Очистить пары, трансформировать x и y, проверить объём данных."""
    if options["x_transform"] not in (_NONE, *TRANSFORMS):
        raise ValueError(f"x_transform ∈ {(_NONE, *TRANSFORMS)}")
    if options["through_origin"] not in _YES_NO:
        raise ValueError(f"through_origin ∈ {_YES_NO}")
    settings = _settings(options["alpha"], options["transform_kind"], options["offset"])
    frame, dropped = _indexed_frame(x, y)
    result = _new_result(method, frame, settings, through_origin=options["through_origin"],
                         x_transform=options["x_transform"], x_offset=options["x_offset"])
    if dropped:
        result.add_warning("missing_dropped", f"Удалено пар с пропусками: {dropped}",
                           Severity.INFO)
    ok = (_transform_column(frame, "y", options["transform_kind"], options["offset"], result)
          and _transform_column(frame, "x", options["x_transform"], options["x_offset"], result)
          and _enough(frame, options, result))
    return frame, result, ok


def _enough(frame: pd.DataFrame, options: dict, result: StatResult) -> bool:
    """Удалённые остатки требуют N − p − 1 ≥ 1, x должен варьировать."""
    p = 1 if options["through_origin"] == "yes" else 2
    if len(frame) < p + options.get("extra", 2):
        result.add_warning("n_too_small", "Слишком мало точек для регрессии", Severity.ERROR)
        return False
    flat = not frame["x"].any() if p == 1 else np.ptp(frame["x"]) == 0
    if flat:
        result.add_warning("no_x_variation", "x не варьирует — наклон не определён",
                           Severity.ERROR)
        return False
    return True


def _design(x: np.ndarray, origin: bool) -> np.ndarray:
    """Матрица плана: [1, x] или [x]."""
    x = np.asarray(x, float)
    return x[:, None] if origin else np.column_stack([np.ones_like(x), x])


def _options(**kwargs: object) -> dict:
    """Собрать опции модели в словарь."""
    return dict(kwargs)


# --- regression_diagnostics ---------------------------------------------------


def _observation_rows(frame: pd.DataFrame, m: dict, flags: dict[str, np.ndarray]) -> list[dict]:
    """Строка на наблюдение: номер, x, y, подгонка, остатки, h, D, C и отметки."""
    keys = ("fitted", "residual", "std_residual", "prediction_residual",
            "deletion_residual", "leverage", "cook_d", "modified_cook")
    rows = []
    for i in range(len(frame)):
        row = {"row": int(frame["row"][i]), "x": float(frame["x"][i]), "y": float(frame["y"][i])}
        row.update({k: float(m[k][i]) for k in keys})
        row.update({k: bool(v[i]) for k, v in flags.items()})
        rows.append(row)
    return rows


def _flags(m: dict, n: int, p: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Отметки по порогам книги: h > 2p/N, |r(i)| > 2, C > 2√((N − p)/N), D > 1."""
    limits = {"leverage": 2 * p / n, "deletion_residual": _RESID_CUTOFF,
              "modified_cook": float(2 * np.sqrt((n - p) / n)), "cook_d": 1.0}
    flags = {"high_leverage": m["leverage"] > limits["leverage"],
             "outlier": np.abs(m["deletion_residual"]) > limits["deletion_residual"],
             "influential": m["modified_cook"] > limits["modified_cook"],
             "cook_d_gt_1": m["cook_d"] > limits["cook_d"]}
    return flags, limits


def _refit_without(frame: pd.DataFrame, keep: np.ndarray, origin: bool, beta: np.ndarray) -> dict | None:
    """Изменение параметров при исключении влиятельных точек (Example 13.2C)."""
    p = 1 if origin else 2
    if keep.all() or keep.sum() < p + 2:
        return None
    sub = frame[keep]
    new, xtx_inv, s2, _ = _ols(_design(sub["x"], origin), sub["y"].to_numpy(float))
    se = np.sqrt(s2 * np.diag(xtx_inv))
    names = ["slope"] if origin else ["intercept", "slope"]
    return {"n_omitted": int((~keep).sum()), "parameters": [
        {"parameter": name, "full": float(beta[j]), "without": float(new[j]),
         "change": float(new[j] - beta[j]), "se_without": float(se[j])}
        for j, name in enumerate(names)]}


def _diagnostic_plots(frame: pd.DataFrame, m: dict) -> dict:
    """Ряды для графиков: удалённые остатки, рычаги от x, C от подгонки и полунормальный."""
    r_del = m["deletion_residual"]
    n = r_del.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    return {
        "residual_vs_fitted": {"fitted": m["fitted"].tolist(), "residuals": r_del.tolist()},
        "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r_del).tolist()},
        "deletion_residuals": r_del.tolist(),
        "leverage_vs_x": {"x": frame["x"].tolist(), "leverage": m["leverage"].tolist()},
        "cook_vs_fitted": {"fitted": m["fitted"].tolist(), "cook": m["modified_cook"].tolist()},
        "cook_half_normal": {"theoretical_quantiles": half_normal_quantiles(n).tolist(),
                             "sample_quantiles": np.sort(m["modified_cook"]).tolist()},
    }


def _diagnostic_warnings(flags: dict[str, np.ndarray], frame: pd.DataFrame,
                         result: StatResult) -> None:
    """Сообщения о выбросах и влиятельных точках с номерами строк."""
    n = len(frame)
    texts = {"outlier": "|удалённый остаток| > 2", "influential": "статистика Кука C выше порога"}
    for key, code in (("outlier", "deletion_outliers"), ("influential", "influential_points")):
        rows = frame["row"][flags[key]].tolist()
        if rows:
            listed = ", ".join(str(r) for r in rows[:20]) + ("…" if len(rows) > 20 else "")
            result.add_warning(code, f"{texts[key]}: строки {listed}. Удалять без найденной ошибки нельзя.", Severity.INFO,
                               params={"count": len(rows), "rows": listed,
                                       "expected": round(0.05 * n, 1)})


def regression_diagnostics(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    through_origin: str = "no",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    x_transform: str = "none",
    x_offset: float = 0.0,
) -> StatResult:
    """Остатки, рычаги и влияние наблюдений в ПЛР с порогами книги (гл. 13)."""
    options = _options(through_origin=through_origin, alpha=alpha, transform_kind=transform_kind,
                       offset=offset, x_transform=x_transform, x_offset=x_offset)
    frame, result, ok = _prepare("regression_diagnostics", x, y, options)
    if not ok:
        return result
    origin = through_origin == "yes"
    design = _design(frame["x"], origin)
    m = influence_measures(design, frame["y"].to_numpy(float))
    n, p = design.shape
    flags, limits = _flags(m, n, p)
    result.parameters.update({
        "observations": _observation_rows(frame, m, flags), "thresholds": limits, "p": p,
        "refit_without_influential": _refit_without(frame, ~flags["influential"], origin,
                                                    m["beta"]),
        "plots": _diagnostic_plots(frame, m),
        "points": {"x": frame["x"].tolist(), "y": frame["y"].tolist()},
        "line": {"intercept": 0.0 if origin else float(m["beta"][0]),
                 "slope": float(m["beta"][-1])}})
    result.statistics.update({f"n_{k}": int(v.sum()) for k, v in flags.items()})
    result.statistics.update({"max_leverage": float(m["leverage"].max()),
                              "max_modified_cook": float(m["modified_cook"].max()),
                              "max_abs_deletion_residual": float(np.abs(m["deletion_residual"]).max()),
                              "ms_resid": m["s2"], "df_resid": n - p})
    _diagnostic_warnings(flags, frame, result)
    return result


# --- regression_cross_validation ----------------------------------------------


def _folds(n: int, scheme: str, train_fraction: float, k_folds: int, seed: int) -> list[np.ndarray]:
    """Индексы проверочных наборов: один (split), по одному (loo) или k (kfold)."""
    if scheme == "loo":
        return [np.array([i]) for i in range(n)]
    order = np.random.default_rng(seed).permutation(n)
    if scheme == "split":
        n_train = round(train_fraction * n)
        return [np.sort(order[n_train:])]
    return [np.sort(part) for part in np.array_split(order, k_folds)]


def _discrepancies(frame: pd.DataFrame, folds: list[np.ndarray], origin: bool) -> pd.DataFrame:
    """Для каждого набора: подгонка на остальных, прогноз Ŷ и расхождение Y − Ŷ."""
    x, y = frame["x"].to_numpy(float), frame["y"].to_numpy(float)
    parts = []
    for k, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(frame)), test)
        beta = _ols(_design(x[train], origin), y[train])[0]
        pred = _design(x[test], origin) @ beta
        parts.append(pd.DataFrame({"row": frame["row"].to_numpy()[test], "x": x[test],
                                   "y": y[test], "predicted": pred, "discrepancy": y[test] - pred,
                                   "fold": k + 1}))
    return pd.concat(parts).sort_values("row").reset_index(drop=True)


def prediction_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """PB, MAD, MSEP, RMSE и их проценты от среднего отклика (стр. 339–340)."""
    d = np.asarray(observed, float) - np.asarray(predicted, float)
    out = {"pb": float(d.mean()), "mad": float(np.abs(d).mean()), "msep": float((d ** 2).mean())}
    out["rmse"] = float(np.sqrt(out["msep"]))
    mean = float(np.mean(observed))
    out["validation_mean"] = mean
    for key in ("pb", "mad", "rmse"):
        out[f"{key}_pct"] = 100 * out[key] / mean if mean else float("nan")
    return out


def _training_fit(frame: pd.DataFrame, test: np.ndarray, origin: bool, result: StatResult) -> None:
    """Параметры, SE, t, P и s обучающей выборки (Table 13.3)."""
    train = frame.drop(index=test)
    design = _design(train["x"], origin)
    beta, xtx_inv, s2, df = _ols(design, train["y"].to_numpy(float))
    se = np.sqrt(s2 * np.diag(xtx_inv))
    names = ["slope"] if origin else ["intercept", "slope"]
    rows = []
    for j, name in enumerate(names):
        t = beta[j] / se[j]
        rows.append({"parameter": name, "estimate": float(beta[j]), "se": float(se[j]),
                     "t": float(t), "p": float(2 * stats.t.sf(abs(t), df))})
        result.estimates.append(Estimate(name=f"train_{name}", value=float(beta[j]),
                                         std_error=float(se[j])))
    yt = train["y"].to_numpy(float)
    tot_ms = float(((yt - yt.mean()) ** 2).sum() / (len(yt) - 1))
    result.parameters["training"] = {"n": len(train), "parameters": rows, "s": float(np.sqrt(s2)),
                                     "adj_r_squared": None if origin else 1 - s2 / tot_ms}
    result.parameters["line"] = {"intercept": 0.0 if origin else float(beta[0]),
                                 "slope": float(beta[-1])}


def _check_cv_args(scheme: str, train_fraction: float, k_folds: int, n: int) -> None:
    """Допустимые схема, доля обучающей выборки и число блоков."""
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme ∈ {_SCHEMES}")
    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction ∈ [0.1, 0.9]")
    if scheme == "kfold" and not 2 <= int(k_folds) <= max(n, 2):
        raise ValueError("k_folds ∈ [2, N]")


def regression_cross_validation(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    scheme: str = "split",
    train_fraction: float = 0.5,
    k_folds: int = 5,
    seed: int = 1,
    through_origin: str = "no",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    x_transform: str = "none",
    x_offset: float = 0.0,
) -> StatResult:
    """Предсказательная способность ПЛР: расхождения на проверочных данных (13.5)."""
    options = _options(through_origin=through_origin, alpha=alpha, transform_kind=transform_kind,
                       offset=offset, x_transform=x_transform, x_offset=x_offset, extra=3)
    frame, result, ok = _prepare("regression_cross_validation", x, y, options)
    _check_cv_args(scheme, train_fraction, k_folds, len(frame))
    result.parameters.update({"scheme": scheme, "seed": int(seed)})
    if not ok:
        return result
    origin = through_origin == "yes"
    folds = _folds(len(frame), scheme, train_fraction, int(k_folds), int(seed))
    table = _discrepancies(frame, folds, origin)
    _cv_output(frame, table, folds, origin, result)
    return result


def _cv_output(frame: pd.DataFrame, table: pd.DataFrame, folds: list[np.ndarray],
               origin: bool, result: StatResult) -> None:
    """Сводки расхождений, PRESS, ResMS полной модели, таблица и графики."""
    summary = prediction_summary(table["y"], table["predicted"])
    full = influence_measures(_design(frame["x"], origin), frame["y"].to_numpy(float))
    summary.update({"n_validation": len(table), "ms_resid_full": full["s2"]})
    if result.parameters["scheme"] == "loo":  # PRESS = Σ ê(i)² (стр. 342)
        summary["press"] = float((table["discrepancy"] ** 2).sum())
    result.statistics.update(summary)
    result.parameters["discrepancies"] = table.to_dict("records")
    if result.parameters["scheme"] == "split":
        _training_fit(frame, folds[0], origin, result)
        result.parameters["points"] = {"x": table["x"].tolist(), "y": table["y"].tolist()}
    result.parameters["plots"] = {"discrepancy_vs_x": {
        "x": table["x"].tolist(), "discrepancy": table["discrepancy"].tolist()}}
