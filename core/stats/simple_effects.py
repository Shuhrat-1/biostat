"""Простые эффекты двухфакторного ANOVA (Witte, гл. 18.9; пакет W4).

Простой эффект — влияние одного фактора при одном уровне другого.
F_se = MS_se / MS_within (ф. 18.5), SS_se = Σ n·(ȳ_ячейки − ȳ_уровня)² (ф. 18.6 для
любых n), знаменатель — общий MS_within всего опыта с df_within = N − cr.

Считаются оба набора (A при каждом уровне B и B при каждом уровне A) без
гейтинга по значимости взаимодействия (решение владельца по гл. 16): книга
советует смотреть один набор, лучше описывающий взаимодействие (стр. 355).
Рядом с P — P по Бонферрони в пределах набора [предл.]: книга множественность
простых эффектов не корректирует.

Пары ячеек внутри каждого простого эффекта — Тьюки(–Крамер) на MS_within с
q(число ячеек, df_within) и d = разность/√MS_within (стр. 357).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Severity, StatResult
from core.stats.factorial import factorial_anova


def _frame(values: list[float], factor_a: list[str], factor_b: list[str]) -> pd.DataFrame:
    """Выровнять колонки; строки с пропуском убрать."""
    if not len(values) == len(factor_a) == len(factor_b):
        raise ValueError("Колонки должны быть одной длины")
    frame = pd.DataFrame({"y": pd.to_numeric(pd.Series(values), errors="coerce"),
                          "a": pd.Series(factor_a).astype(str).str.strip(),
                          "b": pd.Series(factor_b).astype(str).str.strip()})
    bad = frame["a"].isin(["", "nan", "None"]) | frame["b"].isin(["", "nan", "None"])
    return frame[~bad].dropna().reset_index(drop=True)


def _cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Средние, n и SS внутри ячеек (a, b)."""
    grouped = frame.groupby(["a", "b"], sort=False)["y"]
    out = grouped.agg(mean="mean", n="size").reset_index()
    out["ss"] = grouped.apply(lambda s: float(((s - s.mean()) ** 2).sum())).to_numpy()
    return out


def _simple_row(sub: pd.DataFrame, ms_w: float, df_w: int) -> dict:
    """Один простой эффект: SS_se, df, MS, F_se, P."""
    grand = float(np.sum(sub["mean"] * sub["n"]) / sub["n"].sum())
    ss = float(np.sum(sub["n"] * (sub["mean"] - grand) ** 2))
    df = len(sub) - 1
    f = ss / df / ms_w
    return {"ss": ss, "df": df, "ms": ss / df, "F": f, "p": float(stats.f.sf(f, df, df_w))}


def _pairs(sub: pd.DataFrame, by: str, at: str, ms_w: float, df_w: int,
           alpha: float) -> list[dict]:
    """Тьюки–Крамер для ячеек одного простого эффекта; d = разность/√MS_within."""
    k = len(sub)
    q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df_w))
    rows, recs = [], sub.to_dict("records")
    for i in range(k):
        for j in range(i + 1, k):
            c1, c2 = recs[i], recs[j]
            diff = float(c2["mean"] - c1["mean"])
            se = float(np.sqrt(ms_w / 2 * (1 / c1["n"] + 1 / c2["n"])))
            p = float(stats.studentized_range.sf(abs(diff) / se, k, df_w))
            rows.append({"group1": f"{c1[by]} @ {at}", "group2": f"{c2[by]} @ {at}",
                         "mean_diff": diff, "p_adj": p, "ci_lower": diff - q_crit * se,
                         "ci_upper": diff + q_crit * se, "reject": bool(p < alpha),
                         "cohens_d": diff / np.sqrt(ms_w)})
    return rows


def _family(cells: pd.DataFrame, effect: str, at: str, names: dict[str, str],
            fit: dict, alpha: float) -> tuple[list[dict], list[dict]]:
    """Набор простых эффектов фактора effect при каждом уровне фактора at."""
    rows, pairs = [], []
    for level in dict.fromkeys(cells[at]):
        sub = cells[cells[at] == level]
        if len(sub) < 2:
            continue
        row = _simple_row(sub, fit["ms"], fit["df"])
        rows.append({"family": f"{names[effect]} | {names[at]}", "effect_of": names[effect],
                     "at": names[at], "level": level, **row})
        for pair in _pairs(sub, effect, level, fit["ms"], fit["df"], alpha):
            pairs.append({"family": rows[-1]["family"], "level": level, **pair})
    bonf = np.minimum(np.array([r["p"] for r in rows]) * len(rows), 1.0)
    for row, p_b in zip(rows, bonf):
        row["p_bonferroni"] = float(p_b)
    return rows, pairs


def _guard(frame: pd.DataFrame, cells: pd.DataFrame, result: StatResult) -> bool:
    """Нужны ≥ 2 уровня каждого фактора, все ячейки и df_within ≥ 1."""
    n_a, n_b = frame["a"].nunique(), frame["b"].nunique()
    if n_a < 2 or n_b < 2:
        result.add_warning("too_few_groups", "У каждого фактора нужно ≥ 2 уровня", Severity.ERROR)
        return False
    if len(cells) < n_a * n_b:
        result.add_warning("simple_empty_cell",
                           "Есть пустые ячейки A × B: простые эффекты не сравнимы между уровнями.",
                           Severity.ERROR, params={"missing": n_a * n_b - len(cells)})
        return False
    if len(frame) - len(cells) < 1 or cells["ss"].sum() == 0:
        result.add_warning("n_too_small",
                           "Нет разброса внутри ячеек: нужно ≥ 2 разных наблюдений хотя бы в "
                           "одной ячейке (MS_within).", Severity.ERROR)
        return False
    return True


def simple_effects(
    values: list[float],
    factor_a: list[str],
    factor_b: list[str],
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Простые эффекты A при каждом уровне B и B при каждом уровне A (Witte, гл. 18.9)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")
    cn = column_names or {}
    names = {"a": cn.get("factor_a", "A"), "b": cn.get("factor_b", "B")}
    frame = _frame(values, factor_a, factor_b)
    cells = _cells(frame)
    result = StatResult(method="simple_effects", n=len(frame),
                        parameters={"alpha": alpha, "factors": [names["a"], names["b"]]})
    if not _guard(frame, cells, result):
        return result
    df_w = len(frame) - len(cells)
    fit = {"ms": float(cells["ss"].sum()) / df_w, "df": df_w}
    a_rows, a_pairs = _family(cells, "a", "b", names, fit, alpha)
    b_rows, b_pairs = _family(cells, "b", "a", names, fit, alpha)
    result.statistics = {"ms_within": fit["ms"], "df_within": df_w,
                         **_interaction(values, factor_a, factor_b)}
    result.parameters.update({
        "simple_effects": a_rows + b_rows,
        "comparisons": a_pairs + b_pairs,
        "cell_means": _cell_rows(cells, names, fit),
    })
    return result


def _cell_rows(cells: pd.DataFrame, names: dict[str, str], fit: dict) -> list[dict]:
    """Средние ячеек с ДИ по общему MS_within (t на df_within); метка «A=a × B=b»."""
    t_crit = float(stats.t.ppf(0.975, fit["df"]))
    rows = []
    for r in cells.to_dict("records"):
        half = t_crit * float(np.sqrt(fit["ms"] / r["n"]))
        rows.append({"a": r["a"], "b": r["b"], "n": int(r["n"]), "mean": float(r["mean"]),
                     "label": f"{names['a']}={r['a']} × {names['b']}={r['b']}",
                     "ci_lower": float(r["mean"]) - half, "ci_upper": float(r["mean"]) + half})
    return rows


def _interaction(values: list[float], factor_a: list[str], factor_b: list[str]) -> dict:
    """F и P взаимодействия из factorial_anova — для контекста, без гейтинга."""
    res = factorial_anova(values, factor_a=factor_a, factor_b=factor_b,
                          column_names={"factor_a": "A", "factor_b": "B"})
    row = next((r for r in res.parameters.get("anova_table", []) if r["term"] == "A.B"), None)
    return {"F_interaction": row["F"], "p_interaction": row["p"]} if row else {}
