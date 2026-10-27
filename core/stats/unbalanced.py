"""Неортогональные факторные модели (гл. 11 книги).

``unbalanced_anova`` — ANOVA для факторов с неравной повторностью или
пустыми комбинациями (Example 11.2), с блоком и посторонним (extraneous)
фактором (Examples 11.4–11.5):

- проверка ортогональности по условию n_rs = n_r· n_·s / N (Eq. 11.1);
- последовательные таблицы (Type I) в заданном и обратном порядке,
  SS типа II; df по рангу матрицы плана (пустые ячейки учитываются);
- отбор модели сзади наперёд по маргинальности (11.2.4): на каждом шаге
  проверяются члены, не входящие в другие, удаляется наименее значимый;
- предсказания по итоговой модели: таблица ячеек и маргинальные средние
  с равными весами, SE и SED из ковариации оценок; неоцениваемые ячейки — None;
- необязательное сравнение с алгоритмом Хили–Уэстмакотта (11.4).

Блок и посторонний фактор подгоняются первыми и не удаляются при отборе.
"""

from __future__ import annotations

import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.blocking import _fill_residuals
from core.stats.designs import _rss
from core.stats.factorial import (
    _apply_transform,
    _back_transform_table,
    _design_error,
    _frame,
    _natural_order,
    _new_result,
    _rows_to_statistics,
    _settings,
)

_NONE = "none"
_KEYS = ("ex", "fa", "fb", "fc")
_SLOTS = {"ex": ("extraneous", "Extraneous"), "fa": ("factor_a", "A"),
          "fb": ("factor_b", "B"), "fc": ("factor_c", "C"), "block": ("block", "Block")}
_TOL = 1e-9


# --- Члены и подгонка ---------------------------------------------------------


def _model_terms(keys: list[str], model: str) -> list[tuple[str, ...]]:
    """Члены по возрастанию порядка; для additive — только главные эффекты."""
    sizes = range(1, len(keys) + 1) if model == "full" else [1]
    return [t for size in sizes for t in combinations(keys, size)]


def _label(term: tuple[str, ...], names: dict[str, str]) -> str:
    return ".".join(names[k] for k in term)


def _fit_rss(frame: pd.DataFrame, terms: list[tuple[str, ...]], base: list[tuple[str, ...]]
             ) -> tuple[float, int]:
    """RSS и ранг модели [1] + base + terms."""
    return _rss(frame, base + list(terms))


def _increment(frame: pd.DataFrame, before: list, after: list, base: list) -> tuple[float, int]:
    """Прирост модельной SS и df при переходе от before к after (стр. 266)."""
    rss0, r0 = _fit_rss(frame, before, base)
    rss1, r1 = _fit_rss(frame, after, base)
    return max(rss0 - rss1, 0.0), r1 - r0


def _f_row(term: str, kind: str, ss: float, df: int, s2: float, res_df: int) -> dict:
    """Строка таблицы: SS, df, MS и F против заданного остатка."""
    row = {"term": term, "kind": kind, "df": int(df), "ss": float(ss)}
    if df > 0:
        f = ss / df / s2
        row.update({"ms": ss / df, "F": float(f), "p": float(stats.f.sf(f, df, res_df))})
    return row


# --- Ортогональность ----------------------------------------------------------


def orthogonality(frame: pd.DataFrame, cols: list[str], names: dict[str, str]) -> list[dict]:
    """Для каждой пары факторов: таблица повторности и условие Eq. 11.1."""
    out = []
    for a, b in combinations(cols, 2):
        table = pd.crosstab(frame[a], frame[b]).reindex(
            index=_natural_order(frame[a]), columns=_natural_order(frame[b]))
        n = table.to_numpy(float)
        expected = np.outer(n.sum(1), n.sum(0)) / n.sum()
        out.append({"factors": [names[a], names[b]], "orthogonal": bool(np.allclose(n, expected)),
                    "empty_cells": int((n == 0).sum()),
                    "counts": {str(r): {str(c): int(v) for c, v in row.items()}
                               for r, row in table.iterrows()}})
    return out


# --- Таблицы ANOVA ------------------------------------------------------------


def _sequential_table(frame: pd.DataFrame, order: list[tuple[str, ...]], base: list,
                      names: dict[str, str], res: tuple[float, int]) -> list[dict]:
    """Type I: каждый член добавляется к предыдущим (Table 11.11)."""
    s2, res_df = res[0] / res[1], res[1]
    rows = []
    for i, term in enumerate(order):
        ss, df = _increment(frame, order[:i], order[: i + 1], base)
        rows.append(_f_row(_label(term, names), "sequential", ss, df, s2, res_df))
    return rows


def _type2_table(frame: pd.DataFrame, terms: list[tuple[str, ...]], base: list,
                 names: dict[str, str], res: tuple[float, int]) -> list[dict]:
    """Type II: член добавляется к модели из всех других членов того же или меньшего порядка."""
    s2, res_df = res[0] / res[1], res[1]
    rows = []
    for term in terms:
        others = [t for t in terms if len(t) <= len(term) and t != term]
        ss, df = _increment(frame, others, others + [term], base)
        rows.append(_f_row(_label(term, names), "type2", ss, df, s2, res_df))
    return rows


def _structure_rows(frame: pd.DataFrame, base: list, names: dict[str, str],
                    res: tuple[float, int]) -> list[dict]:
    """Строки блока: подгоняется первым (11.6.1)."""
    if not base:
        return []
    rss0, _ = _rss(frame, [])
    rss1, rank = _rss(frame, base)
    ss, df = rss0 - rss1, rank - 1
    return [_f_row(names["block"], "structure", ss, df, res[0] / res[1], res[1])]


# --- Отбор модели -------------------------------------------------------------


def _droppable(terms: list[tuple[str, ...]], protected: set) -> list[tuple[str, ...]]:
    """Члены, не входящие ни в один другой оставшийся член и не защищённые."""
    return [t for t in terms if t not in protected
            and not any(set(t) < set(u) for u in terms)]


def backward_selection(frame: pd.DataFrame, terms: list[tuple[str, ...]], base: list,
                       protected: set, alpha: float, names: dict[str, str]
                       ) -> tuple[list[tuple[str, ...]], list[dict]]:
    """Удаление сзади наперёд по маргинальности на уровне alpha (стр. 270)."""
    current, steps = list(terms), []
    while True:
        rss, rank = _fit_rss(frame, current, base)
        res_df = len(frame) - rank
        tests = []
        for term in _droppable(current, protected):
            reduced = [t for t in current if t != term]
            ss, df = _increment(frame, reduced, current, base)
            tests.append(_f_row(_label(term, names), "drop", ss, df, rss / res_df, res_df))
        candidates = [r for r in tests if r["df"] > 0 and r["p"] >= alpha]
        steps.append({"model": [_label(t, names) for t in current], "tests": tests})
        if not candidates:
            return current, steps
        worst = max(candidates, key=lambda r: r["p"])
        steps[-1]["dropped"] = worst["term"]
        current = [t for t in current if _label(t, names) != worst["term"]]


# --- Предсказания -------------------------------------------------------------


def _formula(terms: list[tuple[str, ...]], base: list) -> str:
    parts = [":".join(f"C({k})" for k in t) for t in base + terms]
    return "y ~ " + (" + ".join(parts) if parts else "1")


def _fit_final(frame: pd.DataFrame, terms: list, base: list) -> dict:
    """МНК итоговой модели (псевдообратная): β, ковариация, s², df, план."""
    y, x = patsy.dmatrices(_formula(terms, base), frame, return_type="dataframe")
    xv = x.to_numpy(float)
    pinv = np.linalg.pinv(xv)
    beta = pinv @ y.to_numpy(float).ravel()
    resid = y.to_numpy(float).ravel() - xv @ beta
    rank = int(np.linalg.matrix_rank(xv))
    s2 = float(resid @ resid / (len(frame) - rank))
    return {"beta": beta, "cov": s2 * np.linalg.pinv(xv.T @ xv), "s2": s2,
            "df": len(frame) - rank, "info": x.design_info, "proj": pinv @ xv}


def _grid(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Все комбинации уровней факторов (включая отсутствующие в данных)."""
    levels = [_natural_order(frame[c]) for c in cols]
    grid = pd.DataFrame(list(product(*levels)), columns=cols)
    grid["y"] = 0.0
    return grid


def _averaged_rows(frame: pd.DataFrame, fit: dict, cols: list[str], keys: list[str]
                   ) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Строки плана, усреднённые с равными весами по прочим факторам и блокам."""
    grid = _grid(frame, cols)
    design = np.asarray(patsy.build_design_matrices([fit["info"]], grid)[0])
    estimable = np.all(np.abs(design - design @ fit["proj"]) < 1e-6, axis=1)
    cells, rows, ok = [], [], []
    for cell, idx in grid.groupby(keys[0] if len(keys) == 1 else keys,
                                   sort=False).groups.items():
        cells.append(cell if isinstance(cell, tuple) else (cell,))
        rows.append(design[idx].mean(axis=0))
        ok.append(bool(estimable[idx].all()))
    return cells, np.vstack(rows), np.array(ok)


def _prediction_table(frame: pd.DataFrame, fit: dict, cols: list[str], keys: list[str],
                      names: dict[str, str], alpha: float) -> dict:
    """Предсказания члена с SE, ДИ и попарными SED (Table 11.15, 11.24)."""
    cells, rows, ok = _averaged_rows(frame, fit, cols, keys)
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    labels = [" × ".join(f"{names[k]}={lv}" for k, lv in zip(keys, cell)) for cell in cells]
    out_rows = []
    for label, row, good in zip(labels, rows, ok):
        mean = float(row @ fit["beta"]) if good else None
        se = float(np.sqrt(row @ fit["cov"] @ row)) if good else None
        out_rows.append({"label": label, "mean": mean, "se": se, "estimable": bool(good),
                         "ci_lower": None if mean is None else mean - t_crit * se,
                         "ci_upper": None if mean is None else mean + t_crit * se})
    comps = _comparisons(labels, rows, ok, fit, t_crit)
    seds = [c["sed"] for c in comps]
    return {"term": ".".join(names[k] for k in keys), "rows": out_rows, "comparisons": comps,
            "t_crit": t_crit, "df": fit["df"],
            "sed_min": min(seds) if seds else None, "sed_max": max(seds) if seds else None}


def _comparisons(labels: list[str], rows: np.ndarray, ok: np.ndarray, fit: dict,
                 t_crit: float) -> list[dict]:
    """Попарные разности оцениваемых предсказаний: SED, t, P, ДИ."""
    out = []
    for i, j in combinations(range(len(labels)), 2):
        if not (ok[i] and ok[j]):
            continue
        d = rows[j] - rows[i]
        diff, sed = float(d @ fit["beta"]), float(np.sqrt(d @ fit["cov"] @ d))
        out.append({"group1": labels[i], "group2": labels[j], "mean_diff": diff, "sed": sed,
                    "lsd": t_crit * sed, "t": diff / sed,
                    "p_unadjusted": float(2 * stats.t.sf(abs(diff / sed), fit["df"])),
                    "ci_lower": diff - t_crit * sed, "ci_upper": diff + t_crit * sed})
    return out


# --- Хили–Уэстмакотт ----------------------------------------------------------


def healy_westmacott(frame: pd.DataFrame, keys: list[str], names: dict[str, str]
                     ) -> list[dict] | None:
    """Пропуски заменяются средними ячеек до полной равной повторности (11.4).

    Возвращает приближённую ANOVA (ResDF уменьшены на число пропусков) или
    None, если есть пустые ячейки. SE по такой таблице занижены.
    """
    counts = frame.groupby(keys).size()
    n_cells = int(np.prod([frame[k].nunique() for k in keys]))
    if len(counts) != n_cells:
        return None
    target = int(counts.max())
    filled = [frame]
    for cell, n in counts.items():
        cell = cell if isinstance(cell, tuple) else (cell,)
        mean = float(frame.loc[(frame[keys] == pd.Series(cell, index=keys)).all(axis=1), "y"].mean())
        filled += [pd.DataFrame([dict(zip(keys, cell)) | {"y": mean}])] * (target - int(n))
    full = pd.concat(filled, ignore_index=True)
    missing = len(full) - len(frame)
    terms = _model_terms(keys, "full")
    rss, rank = _rss(full, terms)
    res = (rss, len(full) - rank - missing)
    rows = _sequential_table(full, terms, [], names, res)
    rows.append({"term": "Residual", "kind": "sequential", "df": res[1], "ss": rss,
                 "ms": rss / res[1]})
    return rows


# --- Метод --------------------------------------------------------------------


def unbalanced_anova(
    values: list[float],
    factor_a: list[str],
    factor_b: list[str] | None = None,
    factor_c: list[str] | None = None,
    block: list[str] | None = None,
    extraneous: list[str] | None = None,
    model: str = "full",
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
    compare_healy_westmacott: str = "no",
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Неортогональная ANOVA с отбором модели и предсказаниями (гл. 11)."""
    settings = _checked_settings(alpha, transform_kind, offset, model)
    given = {"ex": extraneous, "fa": factor_a, "fb": factor_b, "fc": factor_c}
    keys, names = [k for k in _KEYS if given[k] is not None], _display_names(column_names)
    frame = _frame(values, given | {"block": block})
    result = _new_result("unbalanced_anova", frame, settings, model=model,
                         factors=[names[k] for k in keys])
    if not (_guard(frame, keys, model, block is not None, result)
            and _apply_transform(frame, transform_kind, offset, result)):
        return result
    ctx = {"frame": frame, "keys": keys, "names": names, "settings": settings,
           "base": [("block",)] if block is not None else [], "model": model}
    _analyse(ctx, result)
    if compare_healy_westmacott == "yes":
        _add_healy_westmacott(ctx, result)
    return result


def _checked_settings(alpha: float, transform_kind: str, offset: float, model: str) -> dict:
    """Настройки анализа; model ∈ {full, additive}."""
    if model not in ("full", "additive"):
        raise ValueError("model ∈ ('full', 'additive')")
    return _settings(alpha, transform_kind, offset)


def _display_names(column_names: dict[str, str] | None) -> dict[str, str]:
    """Имена колонок из моста или умолчания A, B, C, Block, Extraneous."""
    return {k: (column_names or {}).get(slot, default) for k, (slot, default) in _SLOTS.items()}


def _guard(frame: pd.DataFrame, keys: list[str], model: str, has_block: bool,
           result: StatResult) -> bool:
    """≥ 2 уровней у каждого фактора и остаточные df полной модели > 0."""
    if any(frame[k].nunique() < 2 for k in keys):
        return _design_error(result, "factor_one_level", "У фактора меньше 2 уровней")
    _, rank = _rss(frame, ([("block",)] if has_block else []) + _model_terms(keys, model))
    if len(frame) - rank < 1:
        return _design_error(result, "n_too_small",
                             "Нет остаточных df: упростите модель (additive) или добавьте данные")
    return True


def _analyse(ctx: dict, result: StatResult) -> None:
    """Таблицы полной модели, отбор, предсказания итоговой модели, остатки."""
    frame, keys, names, base = ctx["frame"], ctx["keys"], ctx["names"], ctx["base"]
    terms = _model_terms(keys, ctx["model"])
    rss, rank = _fit_rss(frame, terms, base)
    res = (rss, len(frame) - rank)
    rows = _structure_rows(frame, base, names, res) + _sequential_table(frame, terms, base, names, res)
    rows.append({"term": "Residual", "kind": "residual", "df": res[1], "ss": rss, "ms": rss / res[1]})
    _rows_to_statistics(rows, float(((frame["y"] - frame["y"].mean()) ** 2).sum()), result)
    result.parameters.update(_extra_tables(ctx, terms, res))
    protected = {("ex",)} if "ex" in keys else set()
    final, steps = backward_selection(frame, terms, base, protected,
                                      ctx["settings"]["alpha"], names)
    result.parameters.update({"selection_steps": steps,
                              "final_terms": [_label(t, names) for t in final]})
    _predictions(ctx, final, result)
    _orthogonality_note(result)


def _extra_tables(ctx: dict, terms: list, res: tuple[float, int]) -> dict:
    """Обратный порядок (для 2 факторов), SS типа II и таблицы повторности."""
    frame, keys, names, base = ctx["frame"], ctx["keys"], ctx["names"], ctx["base"]
    out = {"type2": _type2_table(frame, terms, base, names, res),
           "orthogonality": orthogonality(frame, keys + (["block"] if base else []), names)}
    mains = [t for t in terms if len(t) == 1]
    if len(mains) == 2:
        reverse = mains[::-1] + [t for t in terms if len(t) > 1]
        out["sequential_reverse"] = _sequential_table(frame, reverse, base, names, res)
    return out


def _predictions(ctx: dict, final: list, result: StatResult) -> None:
    """Таблица ячеек и таблицы максимальных членов итоговой модели."""
    frame, keys, names, base = ctx["frame"], ctx["keys"], ctx["names"], ctx["base"]
    fit = _fit_final(frame, final, base)
    cols = keys + (["block"] if base else [])
    alpha = ctx["settings"]["alpha"]
    maximal = [list(t) for t in final if not any(set(t) < set(u) for u in final)]
    groups = [keys] + [m for m in maximal if m != keys]
    tables = [_prediction_table(frame, fit, cols, g, names, alpha) for g in groups]
    _finish_tables(tables, ctx["settings"], result)
    result.statistics.update({"df_resid_final": fit["df"], "ms_resid_final": fit["s2"]})
    if any(not r["estimable"] for r in tables[0]["rows"]):
        result.add_warning("prediction_not_estimable",
                           "Часть комбинаций не оценивается итоговой моделью (пустые ячейки "
                           "при взаимодействии): их предсказания пропущены.", Severity.INFO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fill_residuals(smf.ols(_formula(final, base), data=frame).fit(), frame, result)


def _finish_tables(tables: list[dict], settings: dict, result: StatResult) -> None:
    """Обратное преобразование и средние таблиц — в estimates."""
    kind = settings["transform_kind"]
    level = 1 - settings["alpha"]
    for table in tables:
        estimable = [r for r in table["rows"] if r["estimable"]]
        if kind != _NONE:
            _back_transform_table({"rows": estimable, "comparisons": table["comparisons"]},
                                  kind, settings["offset"])
        prefix = "back_" if kind != _NONE else ""
        for row in estimable:
            result.estimates.append(Estimate(
                name=row["label"], value=row[f"{prefix}mean"], ci_lower=row[f"{prefix}ci_lower"],
                ci_upper=row[f"{prefix}ci_upper"], ci_level=level,
                std_error=None if kind != _NONE else row["se"]))
    result.parameters["tables"] = tables


def _orthogonality_note(result: StatResult) -> None:
    """INFO: какие пары факторов неортогональны (порядок подгонки важен)."""
    pairs = [" × ".join(o["factors"]) for o in result.parameters["orthogonality"]
             if not o["orthogonal"]]
    if pairs:
        result.add_warning("nonorthogonal_design",
                           "Факторы неортогональны: SS зависят от порядка подгонки — сравни "
                           f"последовательные таблицы ({', '.join(pairs)}).", Severity.INFO,
                           params={"pairs": ", ".join(pairs)})


def _add_healy_westmacott(ctx: dict, result: StatResult) -> None:
    """Приближённая таблица Хили–Уэстмакотта для сравнения (без блока)."""
    rows = None if ctx["base"] or ctx["model"] != "full" else healy_westmacott(
        ctx["frame"], ctx["keys"], ctx["names"])
    if rows is None:
        result.add_warning("healy_westmacott_unavailable",
                           "Хили–Уэстмакотт доступен только для полной факторной модели без "
                           "блока и без пустых ячеек.", Severity.INFO)
        return
    result.parameters["healy_westmacott"] = rows
    result.add_warning("healy_westmacott_approximate",
                       "Таблица Хили–Уэстмакотта приближённая: SS вариантов завышены, SE "
                       "занижены; при многих пропусках она преувеличивает значимость.",
                       Severity.INFO)
