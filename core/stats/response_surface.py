"""Поверхности отклика от двух переменных (гл. 17.2 книги).

``response_surface`` — полиномиальная поверхность порядка q ≤ 3 в простых или
центрированных переменных (уравнения 17.3, 17.5, 17.6). Члены упорядочены по
степени: x1, x2, x1², x1.x2, x2², x1³, x1².x2, x1.x2², x2³ (Table 17.10).

Остаток — чистая ошибка: по повторам комбинаций (x1, x2) в данных или
заданный пользователем ResMS на df (если в таблице средние вариантов, как
в Table 17.7). Тогда отклонения средних от поверхности — Remainder
(неадекватность), а доля объяснённой вариации вариантов = 1 − RemMS/TrtMS
(стр. 465). Обратный отбор уважает маргинальность: проверяются только члены,
не входящие в другие члены модели (стр. 468, Example 17.3B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.factorial import _new_result, _settings
from core.stats.multiple_regression import parse_names
from core.stats.regression_checks import _transform_column

_SELECTIONS = ("none", "backward")
_CENTRES = ("none", "mean")
_SURFACE_GRID = 21
ORDERS = {"linear": 1, "interaction": 1, "quadratic": 2, "cubic": 3}

Term = tuple[int, int]


def surface_terms(order: str) -> list[Term]:
    """Степени (a, b) с 1 ≤ a + b ≤ q по степени, затем по убыванию степени x1;
    interaction — x1 + x2 + x1.x2 (уравнение 17.3)."""
    if order == "interaction":
        return [(1, 0), (0, 1), (1, 1)]
    q = ORDERS[order]
    return [(d - b, b) for d in range(1, q + 1) for b in range(d + 1)]


def term_name(term: Term, names: tuple[str, str]) -> str:
    """(2, 1) → «H^2.I»."""
    parts = [n if p == 1 else f"{n}^{p}" for n, p in zip(names, term) if p]
    return ".".join(parts)


def is_subterm(small: Term, big: Term) -> bool:
    """small маргинален big: все степени не больше и члены различны (стр. 468)."""
    return small != big and small[0] <= big[0] and small[1] <= big[1]


def _design(u: np.ndarray, v: np.ndarray, terms: list[Term]) -> np.ndarray:
    """[1, u^a·v^b …]."""
    return np.column_stack([np.ones_like(u)] + [u ** a * v ** b for a, b in terms])


# --- Данные -------------------------------------------------------------------


def _frame(y: list[float], columns: dict[str, list[float]], names: list[str],
           centre: str) -> tuple[pd.DataFrame, dict[str, float], int]:
    """Полные строки; u, v — переменные после центрирования (cH = H − H̄, Example 17.3B)."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "x1": np.asarray(columns[names[0]], float),
                          "x2": np.asarray(columns[names[1]], float)})
    clean = frame.dropna().reset_index(drop=True)
    shift = {"x1": 0.0, "x2": 0.0}
    if centre == "mean":
        shift = {k: float(clean[k].mean()) for k in shift}
    clean["u"], clean["v"] = clean["x1"] - shift["x1"], clean["x2"] - shift["x2"]
    return clean, shift, len(frame) - len(clean)


def _variables(columns: dict[str, list[float]], response_name: str, variables: str) -> list[str]:
    """Ровно две объясняющие переменные, отличные от отклика."""
    names = parse_names(variables) or [n for n in columns if n != response_name][:2]
    if len(names) != 2 or response_name in names or any(n not in columns for n in names):
        raise ValueError("Нужны ровно две числовые объясняющие переменные (variables = «x1, x2»)")
    return names


# --- Ошибка и подгонка ---------------------------------------------------------


def _error_term(frame: pd.DataFrame, weights: np.ndarray, error_ms: float,
                error_df: int) -> dict | None:
    """Чистая ошибка: заданная (ResMS, df) или по повторам комбинаций (x1, x2)."""
    cells = frame.groupby(["x1", "x2"])["y"]
    if error_ms > 0 and error_df > 0:
        return {"ms": float(error_ms), "df": int(error_df), "ss": float(error_ms * error_df),
                "source": "given"}
    pe_df = len(frame) - cells.ngroups
    if pe_df < 1:
        return None
    pe_ss = float((weights * (frame["y"] - cells.transform("mean")) ** 2).sum())
    return {"ms": pe_ss / pe_df, "df": pe_df, "ss": pe_ss, "source": "replicates"}


def _rss(frame: pd.DataFrame, w: np.ndarray, terms: list[Term]) -> tuple[float, np.ndarray, np.ndarray]:
    """Взвешенный МНК: RSS, β̂ и (XᵀWX)⁻¹."""
    design = _design(frame["u"].to_numpy(float), frame["v"].to_numpy(float), terms)
    xtwx = design.T @ (w[:, None] * design)
    inv = np.linalg.pinv(xtwx)
    beta = inv @ design.T @ (w * frame["y"].to_numpy(float))
    resid = frame["y"].to_numpy(float) - design @ beta
    return float((w * resid ** 2).sum()), beta, inv


def _f_row(term: str, ss: float, df: int, error: dict) -> dict:
    """SS, MS, F против чистой ошибки и P."""
    f = ss / df / error["ms"]
    return {"term": term, "df": df, "ss": ss, "ms": ss / df, "F": float(f),
            "p": float(stats.f.sf(f, df, error["df"]))}


def _sequential(frame: pd.DataFrame, w: np.ndarray, terms: list[Term], names: tuple[str, str],
                error: dict) -> tuple[list[dict], dict]:
    """Таблица 17.8/17.10: члены по порядку, Remainder, Residual, Total."""
    y = frame["y"].to_numpy(float)
    ybar = float((w * y).sum() / w.sum())
    previous = float((w * (y - ybar) ** 2).sum())
    summary = _variation(frame, previous, error)
    rows = []
    for k, term in enumerate(terms):
        current = _rss(frame, w, terms[:k + 1])[0]
        rows.append(_f_row(f"+ {term_name(term, names)}", previous - current, 1, error))
        previous = current
    remainder = _remainder_row(previous, len(terms), error, summary)
    rows += [remainder] if remainder else []
    rows.append({"term": "Residual", "df": error["df"], "ss": error["ss"], "ms": error["ms"]})
    rows.append({"term": "Total", "df": summary["total_df"], "ss": summary["total_ss"],
                 "ms": summary["total_ss"] / summary["total_df"]})
    return rows, summary


def _variation(frame: pd.DataFrame, tot: float, error: dict) -> dict:
    """SS вариантов (между комбинациями) и общая SS с чистой ошибкой (Table 17.8)."""
    within = error["ss"] if error["source"] == "replicates" else 0.0
    n_cells = frame.groupby(["x1", "x2"]).ngroups
    total_ss = tot + (error["ss"] if error["source"] == "given" else 0.0)
    return {"within": within, "trt_ss": tot - within, "n_cells": n_cells,
            "total_ss": total_ss, "total_df": n_cells - 1 + error["df"]}


def _remainder_row(rss: float, n_terms: int, error: dict, summary: dict) -> dict | None:
    """Отклонения средних вариантов от поверхности (неадекватность) против чистой ошибки."""
    rem_df = summary["n_cells"] - 1 - n_terms
    return _f_row("Remainder", rss - summary["within"], rem_df, error) if rem_df > 0 else None


# --- Отбор с маргинальностью ----------------------------------------------------


def eligible(terms: list[Term]) -> list[Term]:
    """Члены, не являющиеся подчленами других членов модели."""
    return [t for t in terms if not any(is_subterm(t, o) for o in terms)]


def _marginal(frame: pd.DataFrame, w: np.ndarray, terms: list[Term], term: Term,
              error: dict) -> dict:
    """Маргинальный F-тест члена: прирост RSS при его исключении."""
    full = _rss(frame, w, terms)[0]
    reduced = _rss(frame, w, [t for t in terms if t != term])[0]
    return _f_row("", reduced - full, 1, error)


def backward(frame: pd.DataFrame, w: np.ndarray, terms: list[Term], names: tuple[str, str],
             error: dict, alpha: float) -> tuple[list[Term], list[dict]]:
    """Исключать наименее значимый допустимый член, пока у всех P ≤ α (Example 17.3B)."""
    current, steps = list(terms), []
    while current:
        tests = {t: _marginal(frame, w, current, t, error) for t in eligible(current)}
        worst = max(tests, key=lambda t: tests[t]["p"])
        row = {"step": len(steps) + 1, "term": term_name(worst, names),
               "eligible": [term_name(t, names) for t in tests], "F": tests[worst]["F"],
               "p": tests[worst]["p"], "dropped": tests[worst]["p"] > alpha}
        steps.append(row)
        if not row["dropped"]:
            break
        current.remove(worst)
    return current, steps


# --- Параметры и сводки ---------------------------------------------------------


def _parameters(frame: pd.DataFrame, w: np.ndarray, terms: list[Term], names: tuple[str, str],
                error: dict, alpha: float) -> tuple[list[dict], np.ndarray]:
    """Оценки с SE = √(σ²·(XᵀWX)⁻¹), t на df чистой ошибки (Tables 17.9, 17.11)."""
    _, beta, inv = _rss(frame, w, terms)
    se = np.sqrt(error["ms"] * np.diag(inv))
    t_crit = float(stats.t.ppf(1 - alpha / 2, error["df"]))
    labels = ["[1]", *[term_name(t, names) for t in terms]]
    rows = []
    for j, label in enumerate(labels):
        t = beta[j] / se[j]
        rows.append({"term": label, "estimate": float(beta[j]), "se": float(se[j]),
                     "t": float(t), "p": float(2 * stats.t.sf(abs(t), error["df"])),
                     "ci_lower": float(beta[j] - t_crit * se[j]),
                     "ci_upper": float(beta[j] + t_crit * se[j])})
    return rows, beta


def _fit_summary(frame: pd.DataFrame, w: np.ndarray, terms: list[Term], error: dict,
                 summary: dict) -> dict:
    """ModSS, R²adj по всей вариации и доля вариации вариантов 1 − RemMS/TrtMS (стр. 465)."""
    rem = _remainder_row(_rss(frame, w, terms)[0], len(terms), error, summary)
    rem_ss = rem["ss"] if rem else 0.0
    rem_df = rem["df"] if rem else 0
    out = {"model_ss": summary["trt_ss"] - rem_ss, "model_df": len(terms),
           "adj_r_squared": 1 - ((rem_ss + error["ss"]) / (rem_df + error["df"]))
           / (summary["total_ss"] / summary["total_df"])}
    if rem:
        out.update({"F_remainder": rem["F"], "p_remainder": rem["p"],
                    "treatment_variation_explained":
                    1 - rem["ms"] / (summary["trt_ss"] / (summary["n_cells"] - 1))})
    return out


def _raw_coefficients(frame: pd.DataFrame, w: np.ndarray, terms: list[Term],
                      names: tuple[str, str]) -> list[dict]:
    """Та же модель в нецентрированных переменных (раскрытие скобок, стр. 469)."""
    raw = frame.assign(u=frame["x1"], v=frame["x2"])
    _, beta, _ = _rss(raw, w, terms)
    labels = ["[1]", *[term_name(t, names) for t in terms]]
    return [{"term": lab, "estimate": float(b)} for lab, b in zip(labels, beta)]


def _surface(frame: pd.DataFrame, beta: np.ndarray, terms: list[Term], shift: dict) -> dict:
    """Подогнанная поверхность на сетке 21 × 21 для контурного графика."""
    g1 = np.linspace(frame["x1"].min(), frame["x1"].max(), _SURFACE_GRID)
    g2 = np.linspace(frame["x2"].min(), frame["x2"].max(), _SURFACE_GRID)
    u, v = np.meshgrid(g1 - shift["x1"], g2 - shift["x2"], indexing="ij")
    z = _design(u.ravel(), v.ravel(), terms) @ beta
    return {"x1": g1.tolist(), "x2": g2.tolist(),
            "fitted": z.reshape(u.shape).tolist()}


def _plots(frame: pd.DataFrame, beta: np.ndarray, terms: list[Term], error: dict) -> dict:
    """Остатки (y − μ̂)/√σ² против подгонки, по x1 и x2, QQ."""
    fitted = _design(frame["u"].to_numpy(float), frame["v"].to_numpy(float), terms) @ beta
    r = (frame["y"].to_numpy(float) - fitted) / np.sqrt(error["ms"])
    n = r.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    return {"residual_vs_fitted": {"fitted": fitted.tolist(), "residuals": r.tolist()},
            "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
            "std_residuals": r.tolist(), "residual_vs_x1": frame["x1"].tolist(),
            "residual_vs_x2": frame["x2"].tolist()}


# --- response_surface -----------------------------------------------------------


def _setup(y: list[float], columns: dict[str, list[float]], response_name: str, variables: str,
           options: dict) -> tuple[pd.DataFrame, list[str], StatResult, bool]:
    """Проверка опций, полные строки, трансформация отклика."""
    if options["selection"] not in _SELECTIONS or options["centre"] not in _CENTRES:
        raise ValueError(f"selection ∈ {_SELECTIONS}, centre ∈ {_CENTRES}")
    if options["order"] not in ORDERS or options["replicates"] < 1:
        raise ValueError(f"order ∈ {tuple(ORDERS)}, replicates ≥ 1")
    names = _variables(columns, response_name, variables)
    settings = _settings(options["alpha"], options["transform_kind"], options["offset"])
    frame, shift, dropped = _frame(y, columns, names, options["centre"])
    result = _new_result("response_surface", frame, settings, response=response_name,
                         variables=names, order=options["order"], centre=shift,
                         selection=options["selection"])
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    ok = _transform_column(frame, "y", options["transform_kind"], options["offset"], result)
    return frame, names, result, ok


def _check_cells(frame: pd.DataFrame, order: str, error: dict | None, result: StatResult) -> bool:
    """Нужны чистая ошибка и вариантов больше, чем членов поверхности."""
    if error is None:
        result.add_warning("surface_no_pure_error", "Нет повторов комбинаций (x1, x2): задай "
                           "ResMS и df чистой ошибки из анализа исходных данных.", Severity.ERROR)
        return False
    if frame.groupby(["x1", "x2"]).ngroups <= len(surface_terms(order)):
        result.add_warning("surface_too_few_cells", "Различных комбинаций (x1, x2) должно быть "
                           "больше числа членов поверхности.", Severity.ERROR)
        return False
    return True


def response_surface(
    y: list[float],
    columns: dict[str, list[float]],
    response_name: str = "y",
    variables: str = "",
    order: str = "quadratic",
    centre: str = "mean",
    selection: str = "backward",
    error_ms: float = 0.0,
    error_df: int = 0,
    replicates: int = 1,
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Поверхность отклика порядка q (или x1 + x2 + x1.x2) с тестами против чистой ошибки (17.2, Example 17.3)."""
    options = {"order": order, "centre": centre, "selection": selection,
               "replicates": int(replicates), "alpha": alpha, "transform_kind": transform_kind,
               "offset": offset}
    frame, var_names, result, ok = _setup(y, columns, response_name, variables, options)
    w = np.full(len(frame), float(options["replicates"]))
    error = _error_term(frame, w, float(error_ms), int(error_df)) if ok else None
    if not ok or not _check_cells(frame, options["order"], error, result):
        return result
    prefix = "c" if centre == "mean" else ""
    names = (prefix + var_names[0], prefix + var_names[1])
    terms = surface_terms(options["order"])
    rows, summary = _sequential(frame, w, terms, names, error)
    final, steps = (backward(frame, w, terms, names, error, alpha) if selection == "backward"
                    else (terms, []))
    _report(frame, w, final, names, error, summary, tuple(var_names), result)
    result.parameters.update({"sequential": rows, "error": error, "selection_steps": steps,
                              "full_model": _fit_summary(frame, w, terms, error, summary)})
    return result


def _report(frame: pd.DataFrame, w: np.ndarray, terms: list[Term], names: tuple[str, str],
            error: dict, summary: dict, raw_names: tuple[str, str], result: StatResult) -> None:
    """Параметры итоговой модели, сводка подгонки, поверхность, остатки."""
    alpha = result.parameters["alpha"]
    params, beta = _parameters(frame, w, terms, names, error, alpha)
    fit = _fit_summary(frame, w, terms, error, summary)
    result.parameters.update({
        "selected_terms": [term_name(t, names) for t in terms], "parameter_tests": params,
        "raw_parameters": _raw_coefficients(frame, w, terms, raw_names),
        "surface": _surface(frame, beta, terms, result.parameters["centre"]),
        "plots": _plots(frame, beta, terms, error)})
    f_model = fit["model_ss"] / fit["model_df"] / error["ms"]
    result.p_value = float(stats.f.sf(f_model, fit["model_df"], error["df"]))
    result.statistics.update({"F_model": float(f_model), "p_model": result.p_value,
                              "ms_resid": error["ms"], "df_resid": error["df"], **fit})
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))
    if fit.get("p_remainder", 1.0) < alpha:
        result.add_warning("surface_lack_of_fit", "Средние вариантов значимо отклоняются от "
                           "поверхности: добавь члены более высокого порядка.")
