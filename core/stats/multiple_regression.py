"""Множественная линейная регрессия (гл. 14 книги).

- ``multiple_regression`` — модель [1] + x1 + … + xq: корреляции с P
  (Table 14.1), сводная ANOVA (14.3), последовательные SS типа I в
  заданном порядке и маргинальные SS типа III (14.5, 14.7), параметры
  (14.8), VIF (14.9), R², adjusted R², AIC, SBC (14.11), частные остатки,
  влиятельные точки, предсказания с ДИ.
- ``regression_subsets`` — все подмножества переменных (с обязательными
  членами): R², adjusted R², AIC, SBC, Cp Мэллоуза, маргинальные P
  (Tables 14.12, 14.16).
- ``stepwise_selection`` — прямой отбор, обратное исключение и пошаговые
  варианты с порогами Fin / Fout и обнаружением зацикливания (14.9.1).
- ``model_cross_validation`` — сравнение моделей-кандидатов по MSEP, MAD,
  PB на k-fold или разбиении (14.9.3).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.factorial import _new_result, _settings
from core.stats.regression_checks import (
    _diagnostic_warnings,
    _flags,
    _folds,
    _transform_column,
    influence_measures,
    prediction_summary,
)

_STRATEGIES = ("forward", "backward", "forward_stepwise", "backward_stepwise")
_VIF_LIMIT = 10.0
_MAX_SUBSET_VARIABLES = 12
_MAX_STEPS = 100


# --- Данные -------------------------------------------------------------------


def parse_names(text: str) -> list[str]:
    """Имена колонок через «,» или «;» (пробелы внутри имени допустимы)."""
    return [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]


def _select(columns: dict[str, list[float]], response_name: str, variables: str) -> list[str]:
    """Объясняющие переменные: заданные списком или все числовые, кроме отклика."""
    names = parse_names(variables) or [n for n in columns if n != response_name]
    unknown = [n for n in names if n not in columns]
    if unknown:
        raise ValueError(f"Нет числовых колонок: {', '.join(unknown)}")
    if response_name in names:
        raise ValueError("Отклик не может быть объясняющей переменной")
    return list(dict.fromkeys(names))


def _frame(y: list[float], columns: dict[str, list[float]], names: list[str]) -> tuple[pd.DataFrame, int]:
    """Полные строки (отклик и все переменные) с исходным номером строки."""
    data = {"row": np.arange(1, len(y) + 1), "y": np.asarray(y, float)}
    data.update({n: np.asarray(columns[n], float) for n in names})
    frame = pd.DataFrame(data)
    clean = frame.dropna().reset_index(drop=True)
    return clean, len(frame) - len(clean)


def _prepare(method: str, y: list[float], columns: dict[str, list[float]], response_name: str,
             variables: str, options: dict) -> tuple[pd.DataFrame, list[str], StatResult, bool]:
    """Выбор переменных, полные строки, трансформация отклика, проверка объёма."""
    names = _select(columns, response_name, variables)
    settings = _settings(options["alpha"], options["transform_kind"], options["offset"])
    frame, dropped = _frame(y, columns, names)
    result = _new_result(method, frame, settings, response=response_name, variables=names)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    ok = _transform_column(frame, "y", options["transform_kind"], options["offset"], result)
    if ok and len(frame) < len(names) + 3:
        result.add_warning("n_too_small", "Наблюдений должно быть больше числа параметров + 1",
                           Severity.ERROR)
        ok = False
    return frame, names, result, ok


# --- Подгонка -----------------------------------------------------------------


def fit_ols(frame: pd.DataFrame, names: list[str]) -> dict:
    """МНК для [1] + names: параметры, ковариация, RSS, df, s²; rank — ранг плана."""
    design = np.column_stack([np.ones(len(frame))] + [frame[n].to_numpy(float) for n in names])
    y = frame["y"].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    df = len(y) - design.shape[1]
    rss = float(resid @ resid)
    out = {"design": design, "beta": beta, "rank": int(rank), "p": design.shape[1],
           "rss": rss, "df": df, "fitted": design @ beta, "resid": resid, "names": names}
    out["s2"] = rss / df if df > 0 else float("nan")
    if rank == design.shape[1] and df > 0:
        out["cov"] = out["s2"] * np.linalg.inv(design.T @ design)
    return out


def _rss(frame: pd.DataFrame, names: list[str]) -> float:
    """RSS модели [1] + names."""
    return fit_ols(frame, names)["rss"]


def goodness(rss: float, tot_ss: float, n: int, p: int, ms_full: float | None = None) -> dict:
    """R², R²adj, AIC = N ln RSS + 2p, SBC = N ln RSS + p ln N, Cp (Table 14.11)."""
    out = {"r_squared": 1 - rss / tot_ss,
           "adj_r_squared": 1 - (rss / (n - p)) / (tot_ss / (n - 1)),
           "aic": n * np.log(rss) + 2 * p, "sbc": n * np.log(rss) + np.log(n) * p}
    if ms_full:
        out["cp"] = rss / ms_full + 2 * p - n
    return {k: float(v) for k, v in out.items()}


def _check_rank(fit: dict, result: StatResult) -> bool:
    """Совершенная (мульти)коллинеарность — параметры не единственны (стр. 361)."""
    if fit["rank"] < fit["p"]:
        result.add_warning("perfect_collinearity", "Переменные линейно зависимы: удали "
                           "одну из совпадающих (или суммирующихся) переменных.",
                           Severity.ERROR)
        return False
    return True


# --- Таблицы ANOVA ------------------------------------------------------------


def _summary_anova(fit: dict, tot_ss: float, n: int) -> list[dict]:
    """Model (p − 1 df), Residual (N − p), Total (N − 1) (Table 14.2)."""
    mod_ss, mod_df = tot_ss - fit["rss"], fit["p"] - 1
    f = (mod_ss / mod_df) / fit["s2"]
    return [{"term": "Model", "df": mod_df, "ss": mod_ss, "ms": mod_ss / mod_df, "F": f,
             "p": float(stats.f.sf(f, mod_df, fit["df"]))},
            {"term": "Residual", "df": fit["df"], "ss": fit["rss"], "ms": fit["s2"]},
            {"term": "Total", "df": n - 1, "ss": tot_ss}]


def _term_row(term: str, ss: float, fit: dict) -> dict:
    """Строка с 1 df: SS, F против ResMS полной модели, P."""
    f = ss / fit["s2"]
    return {"term": term, "df": 1, "ss": float(ss), "ms": float(ss), "F": float(f),
            "p": float(stats.f.sf(f, 1, fit["df"]))}


def sequential_rows(frame: pd.DataFrame, names: list[str], fit: dict) -> list[dict]:
    """Инкрементные SS (тип I) в заданном порядке; F против ResMS полной модели (14.4.1)."""
    rows, previous = [], float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    for k, name in enumerate(names):
        current = _rss(frame, names[:k + 1])
        rows.append(_term_row(f"+ {name}", previous - current, fit))
        previous = current
    return rows


def marginal_rows(frame: pd.DataFrame, names: list[str], fit: dict) -> list[dict]:
    """Маргинальные SS (тип III): исключение каждой переменной из полной модели (14.4.2)."""
    return [_term_row(f"- {name}", _rss(frame, [m for m in names if m != name]) - fit["rss"],
                      fit) for name in names]


# --- Параметры, VIF, предсказания ----------------------------------------------


def parameter_rows(fit: dict, alpha: float) -> list[dict]:
    """Оценка, SE, t, P и ДИ каждого параметра (Table 14.8)."""
    se = np.sqrt(np.diag(fit["cov"]))
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = []
    for j, name in enumerate(["[1]", *fit["names"]]):
        t = fit["beta"][j] / se[j]
        rows.append({"term": name, "estimate": float(fit["beta"][j]), "se": float(se[j]),
                     "t": float(t), "p": float(2 * stats.t.sf(abs(t), fit["df"])),
                     "ci_lower": float(fit["beta"][j] - t_crit * se[j]),
                     "ci_upper": float(fit["beta"][j] + t_crit * se[j])})
    return rows


def vif_rows(frame: pd.DataFrame, names: list[str]) -> list[dict]:
    """R²_l регрессии x_l на остальные и VIF_l = 1/(1 − R²_l) (стр. 362)."""
    if len(names) < 2:
        return []
    rows = []
    for name in names:
        sub = frame.assign(y=frame[name])
        rss = _rss(sub, [m for m in names if m != name])
        tot = float(((sub["y"] - sub["y"].mean()) ** 2).sum())
        r2 = 1 - rss / tot
        rows.append({"variable": name, "r_squared": float(r2),
                     "vif": float(1 / (1 - r2)) if r2 < 1 else float("inf")})
    return rows


def correlation_rows(frame: pd.DataFrame, names: list[str], response: str) -> list[dict]:
    """Попарные r и P (t на N − 2 df) для переменных и отклика (Table 14.1)."""
    labels = [*names, "y"]
    rows = []
    for a, b in combinations(labels, 2):
        r, p = stats.pearsonr(frame[a], frame[b])
        rows.append({"var1": a if a != "y" else response, "var2": b if b != "y" else response,
                     "r": float(r), "p": float(p)})
    return rows


def _parse_point(text: str, names: list[str]) -> dict[str, float]:
    """«Length=3.2, Hardness=-10» → словарь; все переменные модели обязательны."""
    point = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Ожидалось имя=значение: {part.strip()!r}")
        point[key.strip()] = float(value.strip())
    missing = [n for n in names if n not in point]
    if missing:
        raise ValueError(f"Не заданы значения: {', '.join(missing)}")
    return point


def predictions(fit: dict, frame: pd.DataFrame, text: str, alpha: float) -> list[dict]:
    """μ̂ с ДИ среднего и нового наблюдения для точек через «;» (14.5)."""
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = []
    for chunk in [c for c in (text or "").split(";") if c.strip()]:
        point = _parse_point(chunk, fit["names"])
        x0 = np.array([1.0] + [point[n] for n in fit["names"]])
        mean, var = float(x0 @ fit["beta"]), float(x0 @ fit["cov"] @ x0)
        se_m, se_o = np.sqrt(var), np.sqrt(var + fit["s2"])
        outside = [n for n in fit["names"]
                   if not frame[n].min() <= point[n] <= frame[n].max()]
        rows.append({"point": {n: point[n] for n in fit["names"]}, "fitted": mean,
                     "se_mean": se_m, "mean_lower": mean - t_crit * se_m,
                     "mean_upper": mean + t_crit * se_m, "se_new": se_o,
                     "new_lower": mean - t_crit * se_o, "new_upper": mean + t_crit * se_o,
                     "outside_range": outside})
    return rows


def _partial_residuals(frame: pd.DataFrame, fit: dict) -> dict:
    """e_li = ê_i + β̂_l·x_li и прямая β̂_l·x для каждой переменной (14.6)."""
    out = {}
    for j, name in enumerate(fit["names"], start=1):
        x = frame[name].to_numpy(float)
        b = float(fit["beta"][j])
        out[name] = {"x": x.tolist(), "partial": (fit["resid"] + b * x).tolist(), "slope": b}
    return out


# --- multiple_regression ------------------------------------------------------


def _mlr_tables(frame: pd.DataFrame, names: list[str], fit: dict, result: StatResult) -> None:
    """Сводная ANOVA, тип I и III, параметры, VIF, корреляции, качество подгонки."""
    n = len(frame)
    tot_ss = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    params = parameter_rows(fit, result.parameters["alpha"])
    vifs = vif_rows(frame, names)
    result.parameters.update({
        "anova_table": _summary_anova(fit, tot_ss, n),
        "sequential": sequential_rows(frame, names, fit),
        "marginal": marginal_rows(frame, names, fit), "parameter_tests": params,
        "vif": vifs, "correlations": correlation_rows(frame, names, result.parameters["response"])})
    model = result.parameters["anova_table"][0]
    result.p_value = model["p"]
    result.statistics.update({"F_model": model["F"], "p_model": model["p"],
                              "df_resid": fit["df"], "ms_resid": fit["s2"],
                              **goodness(fit["rss"], tot_ss, n, fit["p"])})
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - result.parameters["alpha"],
                                         std_error=row["se"]))
    _vif_warning(vifs, result)


def _vif_warning(vifs: list[dict], result: StatResult) -> None:
    """VIF > 10 — повод изучить коллинеарность, а не правило удаления (стр. 362)."""
    high = [f"{v['variable']} ({v['vif']:.1f})" for v in vifs if v["vif"] > _VIF_LIMIT]
    if high:
        result.add_warning("collinearity", f"VIF > 10: {', '.join(high)}", Severity.WARNING,
                           params={"variables": ", ".join(high)})


def _mlr_diagnostics(frame: pd.DataFrame, fit: dict, result: StatResult) -> None:
    """Стандартизованные остатки, графики, частные остатки и влиятельные точки."""
    m = influence_measures(fit["design"], frame["y"].to_numpy(float))
    n, p = fit["design"].shape
    flags, limits = _flags(m, n, p)
    r = m["std_residual"]
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    result.parameters["plots"] = {
        "residual_vs_fitted": {"fitted": m["fitted"].tolist(), "residuals": r.tolist()},
        "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
        "std_residuals": r.tolist(), "partial_residuals": _partial_residuals(frame, fit),
        "observed_vs_fitted": {"fitted": m["fitted"].tolist(), "observed": frame["y"].tolist()}}
    result.parameters["thresholds"] = limits
    result.statistics.update({f"n_{k}": int(v.sum()) for k, v in flags.items()})
    _diagnostic_warnings(flags, frame, result)


def multiple_regression(
    y: list[float],
    columns: dict[str, list[float]],
    response_name: str = "y",
    variables: str = "",
    x_predict: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Модель [1] + x1 + … + xq по главе 14; порядок variables задаёт тип I."""
    options = {"alpha": alpha, "transform_kind": transform_kind, "offset": offset}
    frame, names, result, ok = _prepare("multiple_regression", y, columns, response_name,
                                        variables, options)
    if not ok:
        return result
    fit = fit_ols(frame, names)
    if not _check_rank(fit, result):
        return result
    _mlr_tables(frame, names, fit, result)
    _mlr_diagnostics(frame, fit, result)
    result.parameters["predictions"] = predictions(fit, frame, x_predict, alpha)
    outside = sorted({n for row in result.parameters["predictions"] for n in row["outside_range"]})
    if outside:
        result.add_warning("extrapolation_multi", "Точки предсказания вне диапазона данных "
                           f"по: {', '.join(outside)}", params={"variables": ", ".join(outside)})
    return result


# --- regression_subsets -------------------------------------------------------


def _subset_row(frame: pd.DataFrame, base: list[str], extra: tuple[str, ...],
                candidates: list[str], ref: dict) -> dict:
    """Статистики подмодели и маргинальные P кандидатов в ней (Tables 14.12, 14.16)."""
    names = [*base, *extra]
    fit = fit_ols(frame, names)
    row = {"terms": names, "p": fit["p"],
           **goodness(fit["rss"], ref["tot_ss"], ref["n"], fit["p"], ref["ms_full"])}
    if fit["df"] > 0:
        for name in candidates:
            if name in extra:
                drop = _rss(frame, [m for m in names if m != name]) - fit["rss"]
                row[f"p_{name}"] = float(stats.f.sf(drop / fit["s2"], 1, fit["df"]))
    return row


def _best_models(rows: list[dict]) -> dict[str, list[str]]:
    """Лучшая подмодель по каждому критерию (R²adj — max; AIC, SBC — min)."""
    best = {"adj_r_squared": max(rows, key=lambda r: r["adj_r_squared"])}
    best.update({k: min(rows, key=lambda r, k=k: r[k]) for k in ("aic", "sbc")})
    return {k: v["terms"] for k, v in best.items()}


def regression_subsets(
    y: list[float],
    columns: dict[str, list[float]],
    response_name: str = "y",
    variables: str = "",
    always_include: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Все подмножества переменных-кандидатов поверх обязательных членов (14.9)."""
    options = {"alpha": alpha, "transform_kind": transform_kind, "offset": offset}
    frame, names, result, ok = _prepare("regression_subsets", y, columns, response_name,
                                        variables, options)
    base = [n for n in parse_names(always_include)]
    if any(b not in names for b in base):
        raise ValueError("Обязательные члены должны входить в список переменных")
    candidates = [n for n in names if n not in base]
    if len(candidates) > _MAX_SUBSET_VARIABLES:
        raise ValueError(f"Не больше {_MAX_SUBSET_VARIABLES} кандидатов (2^q моделей)")
    if not ok or not _check_rank(fit_ols(frame, names), result):
        return result
    full = fit_ols(frame, names)
    ref = {"n": len(frame), "ms_full": full["s2"],
           "tot_ss": float(((frame["y"] - frame["y"].mean()) ** 2).sum())}
    rows = [_subset_row(frame, base, extra, candidates, ref)
            for k in range(len(candidates) + 1) for extra in combinations(candidates, k)]
    result.parameters.update({"subsets": rows, "best": _best_models(rows),
                              "always_include": base, "candidates": candidates})
    result.statistics.update({"n_models": len(rows), "ms_resid_full": full["s2"]})
    return result


# --- stepwise_selection -------------------------------------------------------


def _incremental_f(frame: pd.DataFrame, smaller: list[str], larger: list[str]) -> tuple[float, float]:
    """F и P для одного добавленного члена: (RSS₀ − RSS₁)/s²₁ на (1, df₁)."""
    big = fit_ols(frame, larger)
    f = (_rss(frame, smaller) - big["rss"]) / big["s2"]
    return float(f), float(stats.f.sf(f, 1, big["df"]))


def _try_add(frame: pd.DataFrame, current: list[str], pool: list[str],
             f_in: float) -> dict | None:
    """Кандидат с наибольшим инкрементным F, если F > Fin и остаются df."""
    best = None
    for name in [n for n in pool if n not in current]:
        if len(frame) - len(current) - 2 < 1:
            break
        f, p = _incremental_f(frame, current, [*current, name])
        if best is None or f > best["F"]:
            best = {"action": "add", "variable": name, "F": f, "p": p}
    return best if best and best["F"] > f_in else None


def _try_drop(frame: pd.DataFrame, current: list[str], keep: list[str],
              f_out: float) -> dict | None:
    """Член с наименьшим маргинальным F, если F < Fout (обязательные не трогаем)."""
    best = None
    for name in [n for n in current if n not in keep]:
        f, p = _incremental_f(frame, [m for m in current if m != name], current)
        if best is None or f < best["F"]:
            best = {"action": "drop", "variable": name, "F": f, "p": p}
    return best if best and best["F"] < f_out else None


def _apply(current: list[str], step: dict) -> list[str]:
    """Новый состав модели после шага."""
    if step["action"] == "add":
        return [*current, step["variable"]]
    return [m for m in current if m != step["variable"]]


def _next_step(frame: pd.DataFrame, current: list[str], pool: list[str], keep: list[str],
               strategy: str, limits: tuple[float, float]) -> dict | None:
    """Один шаг; forward_stepwise сначала пробует исключить, backward_stepwise — добавить.

    Так прямой пошаговый после каждого добавления исключает, пока может, и
    лишь затем добавляет снова; обратный — зеркально (стр. 369–370).
    """
    f_in, f_out = limits
    add = lambda: _try_add(frame, current, pool, f_in)
    drop = lambda: _try_drop(frame, current, keep, f_out)
    if strategy == "forward":
        return add()
    if strategy == "backward":
        return drop()
    first, second = (drop, add) if strategy == "forward_stepwise" else (add, drop)
    return first() or second()


def _run_selection(frame: pd.DataFrame, pool: list[str], keep: list[str], strategy: str,
                   limits: tuple[float, float]) -> tuple[list[dict], list[str], list[str]]:
    """Шаги до остановки или повторения состава модели (зацикливание)."""
    current = list(pool) if strategy.startswith("backward") else list(keep)
    seen, steps = {frozenset(current)}, []
    for _ in range(_MAX_STEPS):
        step = _next_step(frame, current, pool, keep, strategy, limits)
        if step is None:
            return steps, current, []
        new = _apply(current, step)
        steps.append({**step, "terms": len(new)})
        if frozenset(new) in seen:
            cycling = [step["variable"]]
            return steps, [m for m in current if m not in cycling] if step["action"] == "add" \
                else new, cycling
        seen.add(frozenset(new))
        current = new
    return steps, current, []


def _check_strategy(strategy: str, f_in: float, f_out: float) -> None:
    """Допустимая стратегия и положительные пороги."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy ∈ {_STRATEGIES}")
    if f_in <= 0 or f_out <= 0:
        raise ValueError("Пороги Fin и Fout должны быть > 0")


def stepwise_selection(
    y: list[float],
    columns: dict[str, list[float]],
    response_name: str = "y",
    variables: str = "",
    strategy: str = "forward_stepwise",
    f_in: float = 2.0,
    f_out: float = 4.0,
    always_include: str = "",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Автоматический отбор переменных по F-тестам с порогами Fin и Fout (14.9.1)."""
    _check_strategy(strategy, f_in, f_out)
    options = {"alpha": alpha, "transform_kind": transform_kind, "offset": offset}
    frame, names, result, ok = _prepare("stepwise_selection", y, columns, response_name,
                                        variables, options)
    keep = parse_names(always_include)
    if any(k not in names for k in keep):
        raise ValueError("Обязательные члены должны входить в список переменных")
    if not ok or not _check_rank(fit_ols(frame, names), result):
        return result
    steps, final, cycling = _run_selection(frame, names, keep, strategy, (f_in, f_out))
    result.parameters.update({"strategy": strategy, "f_in": f_in, "f_out": f_out,
                              "steps": steps, "selected": final, "cycling": cycling})
    _selection_summary(frame, final, result)
    if cycling:
        result.add_warning("selection_cycle", f"Отбор зациклился на {', '.join(cycling)}: "
                           "переменная исключена — проверь её маргинальный F-тест.",
                           params={"variables": ", ".join(cycling)})
    result.add_warning("selection_caution", "Автоматический отбор — первый шаг: сравни "
                       "кандидатов, проверь остатки и биологический смысл.", Severity.INFO)
    return result


def _selection_summary(frame: pd.DataFrame, final: list[str], result: StatResult) -> None:
    """Параметры и качество итоговой модели отбора."""
    fit = fit_ols(frame, final)
    tot_ss = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    result.statistics.update(goodness(fit["rss"], tot_ss, len(frame), fit["p"]))
    result.statistics.update({"n_selected": len(final), "ms_resid": fit["s2"],
                              "df_resid": fit["df"]})
    if "cov" in fit:
        rows = parameter_rows(fit, result.parameters["alpha"])
        result.parameters["parameter_tests"] = rows
        result.estimates.extend(Estimate(name=r["term"], value=r["estimate"],
                                         std_error=r["se"]) for r in rows)


# --- model_cross_validation ---------------------------------------------------


def parse_models(text: str) -> list[list[str]]:
    """Модели через «|», переменные через «+» или «,»: «Length + Hardness | Length»."""
    models = [parse_names(m.replace("+", ",")) for m in (text or "").split("|")]
    return [m for m in models if m]


def _model_discrepancies(frame: pd.DataFrame, names: list[str], folds: list[np.ndarray]) -> np.ndarray:
    """Прогнозы Ŷ каждой проверочной части по модели, подогнанной на остальных."""
    pred = np.full(len(frame), np.nan)
    for test in folds:
        train = frame.drop(index=test)
        fit = fit_ols(train, names)
        design = np.column_stack([np.ones(len(test))] + [frame[n].to_numpy(float)[test]
                                                          for n in names])
        pred[test] = design @ fit["beta"]
    return pred


def model_cross_validation(
    y: list[float],
    columns: dict[str, list[float]],
    response_name: str = "y",
    models: str = "",
    scheme: str = "kfold",
    train_fraction: float = 0.5,
    k_folds: int = 5,
    seed: int = 1,
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
) -> StatResult:
    """Сравнение моделей-кандидатов по ошибке прогноза на проверочных данных (14.9.3)."""
    candidates = parse_models(models)
    if len(candidates) < 1:
        raise ValueError("Укажи модели: «x1 + x2 | x1»")
    names = list(dict.fromkeys(n for m in candidates for n in m))
    options = {"alpha": alpha, "transform_kind": transform_kind, "offset": offset}
    frame, names, result, ok = _prepare("model_cross_validation", y, columns, response_name,
                                        ", ".join(names), options)
    if scheme not in ("split", "loo", "kfold"):
        raise ValueError("scheme ∈ ('split', 'loo', 'kfold')")
    if not ok:
        return result
    folds = _folds(len(frame), scheme, train_fraction, int(k_folds), int(seed))
    rows = []
    for model in candidates:
        pred = _model_discrepancies(frame, model, folds)
        mask = ~np.isnan(pred)
        rows.append({"terms": model, **prediction_summary(frame["y"][mask], pred[mask])})
    best = min(rows, key=lambda r: r["msep"])
    result.parameters.update({"scheme": scheme, "seed": int(seed), "models": rows,
                              "best": best["terms"]})
    result.statistics.update({"best_msep": best["msep"], "n_models": len(rows)})
    return result
