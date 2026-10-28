"""Регрессия с группами (гл. 15.1–15.2 книги).

``regression_with_groups`` — отклик, одна или несколько количественных x и
фактор групп: модели «отдельные линии», «параллельные линии», «одна
линия», «нуль» (15.1.1); последовательные ANOVA в двух порядках (Tables
15.5, 15.6, 15.8); отбор по маргинальности от полной модели (15.1.2.4,
15.2); параметры итоговой модели в простой параметризации (отдельные α_j
и β_j, Tables 15.2, 15.10); разности сдвигов групп с ДИ и, для log-шкалы,
отношения (Example 15.2).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.factorial import _NONE, _natural_order, _new_result, _settings
from core.stats.multiple_regression import _select
from core.stats.regression_checks import _transform_column, influence_measures

_REFIT_DF = 10  # ResDF ≤ 10: F-тесты по пересчитанной модели (стр. 396)
_MIN_PER_GROUP = 3
_OVERLAP = 0.5  # эвристика BioStat: доля перекрытия диапазонов x между группами
_LOG_BASE = {"log10": 10.0, "ln": np.e}


# --- Члены модели и матрица плана ----------------------------------------------


def _term_label(term: tuple[str, ...]) -> str:
    """('x', 'grp') → 'x.grp'."""
    return ".".join(term)


def _columns(frame: pd.DataFrame, term: tuple[str, ...], levels: list[str]) -> np.ndarray:
    """Столбцы члена: x — сама переменная, grp — индикаторы (кроме первого), x.grp — x·индикатор."""
    dummies = np.column_stack([(frame["grp"] == lev).to_numpy(float) for lev in levels[1:]])
    if term == ("grp",):
        return dummies
    x = frame[term[0]].to_numpy(float)[:, None]
    return x if len(term) == 1 else x * dummies


def _design(frame: pd.DataFrame, terms: list[tuple[str, ...]], levels: list[str]) -> np.ndarray:
    """[1] + члены в first-level-zero параметризации (стр. 388–390)."""
    blocks = [np.ones((len(frame), 1))] + [_columns(frame, t, levels) for t in terms]
    return np.column_stack(blocks)


def _fit(frame: pd.DataFrame, terms: list[tuple[str, ...]], levels: list[str]) -> dict:
    """RSS, ранг и df модели."""
    design = _design(frame, terms, levels)
    y = frame["y"].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    rss = float(((y - design @ beta) ** 2).sum())
    return {"rss": rss, "rank": int(rank), "df": len(y) - int(rank)}


def _marginal_to(term: tuple[str, ...], terms: list[tuple[str, ...]]) -> bool:
    """Член входит в другой член модели (x и grp маргинальны к x.grp, стр. 399)."""
    return any(set(term) < set(other) for other in terms)


# --- Последовательные таблицы --------------------------------------------------


def _sequence(variables: list[str], group_first: bool) -> list[tuple[str, ...]]:
    """Порядок: переменные, группы, взаимодействия (или группы первыми)."""
    mains = [(v,) for v in variables]
    order = [("grp",), *mains] if group_first else [*mains, ("grp",)]
    return order + [(v, "grp") for v in variables]


def _reverse_sequence(variables: list[str], group_first: bool) -> list[tuple[str, ...]]:
    """Второй порядок: для одной x — группы и x местами (Table 15.6), иначе x в обратном порядке."""
    if len(variables) == 1:
        return _sequence(variables, not group_first)
    return _sequence(variables[::-1], group_first)


def sequential_table(frame: pd.DataFrame, terms: list[tuple[str, ...]], levels: list[str],
                     names: dict[str, str]) -> list[dict]:
    """Инкрементные SS и df; F против ResMS полной модели (Table 15.4)."""
    full = _fit(frame, terms, levels)
    res_ms = full["rss"] / full["df"]
    rows, prev = [], _fit(frame, [], levels)
    for k in range(len(terms)):
        cur = _fit(frame, terms[:k + 1], levels)
        df, ss = cur["rank"] - prev["rank"], prev["rss"] - cur["rss"]
        row = {"term": "+ " + _display(terms[k], names), "df": df, "ss": ss}
        if df > 0:
            f = ss / df / res_ms
            row.update({"ms": ss / df, "F": f, "p": float(stats.f.sf(f, df, full["df"]))})
        rows.append(row)
        prev = cur
    total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    rows += [{"term": "Residual", "df": full["df"], "ss": full["rss"], "ms": res_ms},
             {"term": "Total", "df": len(frame) - 1, "ss": total}]
    return rows


def _display(term: tuple[str, ...], names: dict[str, str]) -> str:
    """Имя члена с именами колонок пользователя."""
    return _term_label(tuple(names.get(t, t) for t in term))


# --- Отбор модели по маргинальности -------------------------------------------


def _drop_test(frame: pd.DataFrame, current: list[tuple[str, ...]], term: tuple[str, ...],
               levels: list[str], res: tuple[float, int]) -> dict:
    """F для исключения члена из текущей модели против заданного ResMS (s², df)."""
    big = _fit(frame, current, levels)
    small = _fit(frame, [t for t in current if t != term], levels)
    df = big["rank"] - small["rank"]
    f = (small["rss"] - big["rss"]) / df / res[0]
    return {"term": term, "df": df, "F": float(f), "p": float(stats.f.sf(f, df, res[1]))}


def select_model(frame: pd.DataFrame, terms: list[tuple[str, ...]], levels: list[str],
                 alpha: float) -> tuple[list[tuple[str, ...]], list[dict]]:
    """Обратное исключение немаргинальных членов с P > α (стр. 392–393, 397, 400)."""
    current, steps = list(terms), []
    full = _fit(frame, terms, levels)
    while current:
        fit = full if full["df"] > _REFIT_DF else _fit(frame, current, levels)
        res = (fit["rss"] / fit["df"], fit["df"])
        tests = [_drop_test(frame, current, t, levels, res)
                 for t in current if not _marginal_to(t, current)]
        worst = max(tests, key=lambda r: r["p"])
        steps.append({**worst, "dropped": worst["p"] > alpha})
        if worst["p"] <= alpha:
            break
        current.remove(worst["term"])
    return current, steps


# --- Параметры итоговой модели ------------------------------------------------


def _simple_design(frame: pd.DataFrame, terms: list[tuple[str, ...]], levels: list[str],
                   variables: list[str]) -> tuple[np.ndarray, list[str]]:
    """Простая параметризация: α_j по группам (или общий α), β или β_j по каждой x."""
    ind = np.column_stack([(frame["grp"] == lev).to_numpy(float) for lev in levels])
    has_grp = ("grp",) in terms
    cols, labels = ([ind], [f"α[{lev}]" for lev in levels]) if has_grp else (
        [np.ones((len(frame), 1))], ["α"])
    for v in variables:
        x = frame[v].to_numpy(float)[:, None]
        if (v, "grp") in terms:
            cols.append(x * ind)
            labels += [f"β_{v}[{lev}]" for lev in levels]
        elif (v,) in terms:
            cols.append(x)
            labels.append(f"β_{v}")
    return np.column_stack(cols), labels


def final_parameters(frame: pd.DataFrame, terms: list[tuple[str, ...]], levels: list[str],
                     variables: list[str], alpha: float) -> dict:
    """Оценки, SE, t, P и ДИ итоговой модели, ковариация и R²adj (Tables 15.2, 15.10)."""
    design, labels = _simple_design(frame, terms, levels, variables)
    y = frame["y"].to_numpy(float)
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    df = len(y) - np.linalg.matrix_rank(design)
    s2 = float(((y - design @ beta) ** 2).sum() / df)
    cov = s2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    rows = [{"term": lab, "estimate": float(b), "se": float(s), "t": float(b / s),
             "p": float(2 * stats.t.sf(abs(b / s), df)), "ci_lower": float(b - t_crit * s),
             "ci_upper": float(b + t_crit * s)} for lab, b, s in zip(labels, beta, se)]
    tot_ms = float(((y - y.mean()) ** 2).sum() / (len(y) - 1))
    return {"rows": rows, "cov": cov, "labels": labels, "df": int(df), "s2": s2,
            "adj_r_squared": 1 - s2 / tot_ms, "t_crit": t_crit, "design": design}


def intercept_differences(params: dict, levels: list[str], kind: str) -> list[dict]:
    """α_i − α_j с SE и ДИ; на log-шкале — отношение и его ДИ (Example 15.2, стр. 405)."""
    idx = {lab: k for k, lab in enumerate(params["labels"])}
    if f"α[{levels[0]}]" not in idx:
        return []
    rows = []
    for a, b in combinations(levels, 2):
        i, j = idx[f"α[{b}]"], idx[f"α[{a}]"]
        cov = params["cov"]
        diff = params["rows"][i]["estimate"] - params["rows"][j]["estimate"]
        se = float(np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j]))
        half = params["t_crit"] * se
        row = {"contrast": f"{b} − {a}", "difference": diff, "se": se,
               "ci_lower": diff - half, "ci_upper": diff + half}
        if kind in _LOG_BASE:
            base = _LOG_BASE[kind]
            row.update({"ratio": base ** diff, "ratio_lower": base ** (diff - half),
                        "ratio_upper": base ** (diff + half)})
        rows.append(row)
    return rows


def _slope_ratios(params: dict, kind: str) -> list[dict]:
    """Множитель отклика на единицу x: base^β с ДИ (стр. 405)."""
    if kind not in _LOG_BASE:
        return []
    base = _LOG_BASE[kind]
    return [{"term": r["term"], "factor": base ** r["estimate"], "lower": base ** r["ci_lower"],
             "upper": base ** r["ci_upper"]} for r in params["rows"] if r["term"].startswith("β")]


# --- Стандартные модели для одной x --------------------------------------------


def standard_models(frame: pd.DataFrame, x: str, levels: list[str]) -> list[dict]:
    """R²adj и ModSS моделей: нуль, одна линия, отдельные группы, параллельные, отдельные линии."""
    specs = {"null": [], "single_line": [(x,)], "separate_groups": [("grp",)],
             "parallel_lines": [(x,), ("grp",)],
             "separate_lines": [(x,), ("grp",), (x, "grp")]}
    tot = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    n, rows = len(frame), []
    for name, terms in specs.items():
        fit = _fit(frame, terms, levels)
        adj = 1 - (fit["rss"] / fit["df"]) / (tot / (n - 1))
        rows.append({"model": name, "model_df": fit["rank"] - 1, "model_ss": tot - fit["rss"],
                     "adj_r_squared": float(adj)})
    return rows


# --- Подготовка данных -------------------------------------------------------


def _frame(y: list[float], columns: dict[str, list[float]], group: list[str],
           variables: list[str]) -> tuple[pd.DataFrame, int]:
    """Полные строки: отклик, переменные, метка группы."""
    data = {"y": np.asarray(y, float), "grp": pd.Series(group, dtype=object).replace("", np.nan)}
    data.update({v: np.asarray(columns[v], float) for v in variables})
    frame = pd.DataFrame(data)
    clean = frame.dropna().reset_index(drop=True)
    clean["grp"] = clean["grp"].astype(str)
    return clean, len(frame) - len(clean)


def _guard(frame: pd.DataFrame, levels: list[str], variables: list[str], result: StatResult) -> bool:
    """Нужно ≥ 2 групп, ≥ 3 наблюдений в группе и остаточные df у полной модели."""
    counts = frame["grp"].value_counts()
    if len(levels) < 2:
        result.add_warning("factor_one_level", "Нужно минимум 2 группы", Severity.ERROR)
        return False
    if counts.min() < _MIN_PER_GROUP:
        result.add_warning("group_too_small", f"В группе меньше {_MIN_PER_GROUP} наблюдений: "
                           f"{counts.idxmin()}", Severity.ERROR,
                           params={"group": str(counts.idxmin()), "min": _MIN_PER_GROUP})
        return False
    if len(frame) <= len(levels) * (len(variables) + 1):
        result.add_warning("n_too_small", "Нет остаточных df для модели отдельных линий",
                           Severity.ERROR)
        return False
    return True


def _overlap_warning(frame: pd.DataFrame, variables: list[str], result: StatResult) -> None:
    """Сравнивать группы разумно при сходных диапазонах x (стр. 396, 400)."""
    poor = []
    for v in variables:
        ranges = frame.groupby("grp")[v].agg(["min", "max"])
        for a, b in combinations(ranges.index, 2):
            lo, hi = max(ranges.loc[a, "min"], ranges.loc[b, "min"]), min(
                ranges.loc[a, "max"], ranges.loc[b, "max"])
            width = min(ranges.loc[a, "max"] - ranges.loc[a, "min"],
                        ranges.loc[b, "max"] - ranges.loc[b, "min"])
            if width > 0 and (hi - lo) / width < _OVERLAP:
                poor.append(f"{v}: {a}–{b}")
    if poor:
        result.add_warning("groups_poor_overlap", f"Диапазоны x групп мало перекрываются: "
                           f"{', '.join(poor)}", params={"pairs": ", ".join(poor)})


# --- Метод каталога -----------------------------------------------------------


def regression_with_groups(
    y: list[float],
    columns: dict[str, list[float]],
    group: list[str],
    response_name: str = "y",
    variables: str = "",
    group_first: str = "no",
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Регрессия с группами: выбор между отдельными, параллельными и одной линией (гл. 15)."""
    settings = _settings(alpha, transform_kind, offset)
    names_x = _select(columns, response_name, variables)
    frame, dropped = _frame(y, columns, group, names_x)
    group_name = (column_names or {}).get("group", "grp")
    result = _new_result("regression_with_groups", frame, settings, response=response_name,
                         variables=names_x, group=group_name)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    levels = _natural_order(frame["grp"])
    if not _transform_column(frame, "y", transform_kind, offset, result) or not _guard(
            frame, levels, names_x, result):
        return result
    _fill(frame, levels, names_x, group_name, group_first == "yes", result)
    return result


def _fill(frame: pd.DataFrame, levels: list[str], variables: list[str], group_name: str,
          group_first: bool, result: StatResult) -> None:
    """Таблицы, отбор, итоговая модель и предупреждения."""
    names = {"grp": group_name}
    alpha = result.parameters["alpha"]
    full = _sequence(variables, group_first)
    result.parameters["sequential"] = sequential_table(frame, full, levels, names)
    result.parameters["sequential_reverse"] = sequential_table(
        frame, _reverse_sequence(variables, group_first), levels, names)
    if len(variables) == 1:
        result.parameters["standard_models"] = standard_models(frame, variables[0], levels)
    final, steps = select_model(frame, full, levels, alpha)
    result.parameters["selection_steps"] = [
        {**s, "term": _display(s["term"], names)} for s in steps]
    result.parameters["selected_terms"] = [_display(t, names) for t in final]
    _final_model(frame, final, levels, variables, result)
    _overlap_warning(frame, variables, result)


def _final_model(frame: pd.DataFrame, final: list[tuple[str, ...]], levels: list[str],
                 variables: list[str], result: StatResult) -> None:
    """Параметры, разности сдвигов, отношения и плоские статистики итоговой модели."""
    alpha, kind = result.parameters["alpha"], result.parameters["transform"]
    params = final_parameters(frame, final, levels, variables, alpha)
    result.parameters["parameter_tests"] = params["rows"]
    result.parameters["model_type"] = _model_type(final, variables)
    if ("grp",) in final and not any(len(t) == 2 for t in final):
        result.parameters["intercept_differences"] = intercept_differences(params, levels, kind)
    result.parameters["slope_factors"] = _slope_ratios(params, kind if kind != _NONE else "")
    result.parameters["plots"] = _residual_plots(params["design"], frame)
    result.statistics.update({"adj_r_squared": params["adj_r_squared"], "df_resid": params["df"],
                              "ms_resid": params["s2"], "n_groups": len(levels)})
    for row in params["rows"]:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))


def _residual_plots(design: np.ndarray, frame: pd.DataFrame) -> dict:
    """Стандартизованные остатки итоговой модели с метками групп (Fig. 15.4)."""
    m = influence_measures(design, frame["y"].to_numpy(float))
    r = m["std_residual"]
    n = r.size
    blom = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    return {"residual_vs_fitted": {"fitted": m["fitted"].tolist(), "residuals": r.tolist()},
            "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
            "std_residuals": r.tolist(), "groups": frame["grp"].tolist()}


def _model_type(final: list[tuple[str, ...]], variables: list[str]) -> str:
    """Название итоговой модели для одной x (стр. 382–383); иначе «general»."""
    if len(variables) != 1:
        return "general"
    x = variables[0]
    key = frozenset(final)
    return {frozenset(): "null", frozenset({(x,)}): "single_line",
            frozenset({("grp",)}): "separate_groups",
            frozenset({(x,), ("grp",)}): "parallel_lines"}.get(key, "separate_lines")
