"""Дизайны со сложной блочной структурой (гл. 9 книги).

- ``latin_square_anova`` — латинский квадрат: страты Row, Column, Row.Column;
  один фактор вариантов или факторная пара A*B (Example 9.1).
- ``split_plot_anova`` — расщеплённые делянки: A на целых делянках, B на
  субделянках; страты Block, Block.WholePlot, Block.WholePlot.Subplot; SE и
  SED по стратам, df Саттертуэйта для сравнений через страты (Example 9.5).
- ``bibd_anova`` — сбалансированный неполный блочный дизайн: внутриблочный
  анализ, межблочная информация о вариантах в страте блоков, эффективность
  EF = λt/(nu), скорректированные средние и SED = √(2s²/(n·EF)) (9.3).

Суммы квадратов считаются как разности остаточных SS вложенных моделей
(np.linalg.lstsq по индикаторам), df — как прирост ранга матрицы плана.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.blocking import _fill_residuals
from core.stats.factorial import (
    _apply_transform,
    _back_transform_table,
    _design_error,
    _effects,
    _fill_tables,
    _frame,
    _natural_order,
    _new_result,
    _no_residual,
    _rows_to_statistics,
    _select_prediction_terms,
    _settings,
    _tables_to_estimates,
    _terms,
)

_NONE = "none"


# --- Общие вычисления ---------------------------------------------------------


def _design_matrix(frame: pd.DataFrame, terms: list[tuple[str, ...]]) -> np.ndarray:
    """Константа и индикаторы уровней (или комбинаций уровней) каждого члена."""
    blocks = [np.ones((len(frame), 1))]
    for term in terms:
        labels = frame[list(term)].astype(str).agg("|".join, axis=1)
        blocks.append(pd.get_dummies(labels).to_numpy(float))
    return np.hstack(blocks)


def _rss(frame: pd.DataFrame, terms: list[tuple[str, ...]]) -> tuple[float, int]:
    """Остаточная SS и ранг модели y ~ [1] + члены (МНК с вырожденностью)."""
    x = _design_matrix(frame, terms)
    y = frame["y"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return float(resid @ resid), int(np.linalg.matrix_rank(x))


def _sequential(frame: pd.DataFrame, steps: list[tuple[str, list[tuple[str, ...]]]]
                ) -> dict[str, tuple[float, int]]:
    """SS и df каждого шага: разность RSS и ранга при последовательном добавлении."""
    out, prev_rss, prev_rank = {}, *_rss(frame, [])
    terms: list[tuple[str, ...]] = []
    for name, added in steps:
        terms = terms + added
        rss, rank = _rss(frame, terms)
        out[name] = (prev_rss - rss, rank - prev_rank)
        prev_rss, prev_rank = rss, rank
    out["Residual"] = (prev_rss, len(frame) - prev_rank)
    return out


def _line(term: str, stratum: str, ss: float, df: int, denom: tuple[float, float] | None
          ) -> dict:
    """Строка ANOVA; F и P — против указанного остатка (MS, df), если он есть."""
    row = {"term": term, "stratum": stratum, "df": int(df), "ss": float(ss),
           "ms": float(ss / df) if df else 0.0}
    if denom is not None and df and denom[1] > 0 and denom[0] > 0:
        f = row["ms"] / denom[0]
        row.update({"F": float(f), "p": float(stats.f.sf(f, df, denom[1]))})
    return row


def _total_ss(frame: pd.DataFrame) -> float:
    return float(((frame["y"] - frame["y"].mean()) ** 2).sum())


def _names(column_names: dict[str, str] | None, defaults: dict[str, str]) -> dict[str, str]:
    """Отображаемые имена: имена колонок из моста или умолчания."""
    column_names = column_names or {}
    return {key: column_names.get(slot, default) for key, (slot, default) in
            ((k, v.split(":")) for k, v in defaults.items())}


# --- Латинский квадрат --------------------------------------------------------


def latin_square_anova(
    values: list[float],
    factor_a: list[str],
    row: list[str],
    column: list[str],
    factor_b: list[str] | None = None,
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """ANOVA латинского квадрата (9.1): ряды и колонки — перекрёстные блоки.

    С ``factor_b`` варианты — комбинации A × B, член вариантов разбивается на
    A, B, A.B; члены для предсказания — по маргинальности (гл. 8).
    """
    settings = _settings(alpha, transform_kind, offset)
    names = _names(column_names, {"fa": "factor_a:A", "fb": "factor_b:B",
                                  "row": "row:Row", "col": "column:Column"})
    keys = ["fa"] + (["fb"] if factor_b is not None else [])
    frame = _frame(values, {"fa": factor_a, "fb": factor_b, "row": row, "col": column})
    result = _new_result("latin_square_anova", frame, settings,
                         factors=[names[k] for k in keys])
    if not (_latin_guard(frame, keys, result)
            and _apply_transform(frame, transform_kind, offset, result)):
        return result
    _latin_fit(frame, keys, names, result)
    if result.statistics.get("df_resid", 0) > 0:
        _predict_by_marginality(frame, keys, names, settings, result)
    return result


def _latin_guard(frame: pd.DataFrame, keys: list[str], result: StatResult) -> bool:
    """t рядов = t колонок = t вариантов, каждая ячейка — одна единица, каждый вариант
    ровно раз в каждом ряду и каждой колонке (стр. 210)."""
    trt = frame[keys].astype(str).agg("|".join, axis=1)
    t = trt.nunique()
    shape_ok = frame["row"].nunique() == t == frame["col"].nunique() and len(frame) == t * t
    once = (frame.groupby(["row", "col"]).size().eq(1).all()
            and pd.crosstab(frame["row"], trt).eq(1).all().all()
            and pd.crosstab(frame["col"], trt).eq(1).all().all())
    if not (shape_ok and once):
        return _design_error(result, "latin_square_invalid",
                             "Не латинский квадрат: нужно t рядов, t колонок, t вариантов и "
                             "каждый вариант ровно раз в каждом ряду и колонке")
    if len(keys) == 2 and frame["fa"].nunique() * frame["fb"].nunique() != t:
        return _design_error(result, "factorial_unbalanced",
                             "Варианты должны быть всеми комбинациями A × B")
    if t < 3:
        return _design_error(result, "n_too_small", "Квадрат 2 × 2 не даёт остаточных df")
    return True


def _latin_fit(frame: pd.DataFrame, keys: list[str], names: dict[str, str],
               result: StatResult) -> None:
    """Страты Row, Column и Row.Column с членами вариантов (Tables 9.3–9.5)."""
    trt_terms = _terms(keys)
    seq = _sequential(frame, [("Row", [("row",)]), ("Column", [("col",)])]
                      + [(".".join(names[k] for k in t), [t]) for t in trt_terms])
    res_ss, res_df = seq.pop("Residual")
    if _no_residual(res_ss, frame):
        _design_error(result, "no_residual_variation", "Остаточной вариации нет")
        return
    denom = (res_ss / res_df, res_df)
    rows = [_line(names["row"], "row", *seq.pop("Row"), denom),
            _line(names["col"], "column", *seq.pop("Column"), denom)]
    rows += [_line(term, "unit", ss, df, denom) for term, (ss, df) in seq.items()]
    rows.append(_line("Residual", "unit", res_ss, res_df, None))
    _rows_to_statistics(rows, _total_ss(frame), result)
    result.parameters["effects"] = _effects(
        frame, [("row",), ("col",)] + trt_terms, names)
    formula = "y ~ C(row) + C(col) + " + "*".join(f"C({k})" for k in keys)
    _fill_residuals(smf.ols(formula, data=frame).fit(), frame, result)


def _predict_by_marginality(frame: pd.DataFrame, keys: list[str], names: dict[str, str],
                            settings: dict, result: StatResult) -> None:
    """Члены для предсказания (стр. 166–167) и их таблицы средних на ResMS."""
    rows = {r["term"]: r for r in result.parameters["anova_table"]}
    terms = _terms(keys)
    p_values = {t: rows[".".join(names[k] for k in t)]["p"] for t in terms}
    chosen = _select_prediction_terms(terms, p_values, settings["alpha"])
    result.parameters["prediction_terms"] = [".".join(names[k] for k in t) for t in chosen]
    if chosen:
        _fill_tables(frame, [list(t) for t in chosen], names, settings, result)
    else:
        result.add_warning("no_significant_terms",
                           "Ни один член не значим: предсказание — общее среднее",
                           Severity.INFO)


# --- Расщеплённые делянки -----------------------------------------------------


def split_plot_anova(
    values: list[float],
    factor_a: list[str],
    factor_b: list[str],
    block: list[str] | None = None,
    whole_plot: list[str] | None = None,
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Расщеплённые делянки (9.2): A — на целых делянках, B — на субделянках.

    Целая делянка — ``whole_plot`` (внутри блока) или пара (блок, уровень A).
    """
    settings = _settings(alpha, transform_kind, offset)
    names = _names(column_names, {"fa": "factor_a:A", "fb": "factor_b:B",
                                  "block": "block:Block", "wp": "whole_plot:WholePlot"})
    frame = _frame(values, {"fa": factor_a, "fb": factor_b, "block": block, "wp": whole_plot})
    result = _new_result("split_plot_anova", frame, settings,
                         factors=[names["fa"], names["fb"]])
    if block is None and whole_plot is None:
        _design_error(result, "split_plot_no_units",
                      "Укажи колонку блоков или колонку целых делянок")
        return result
    frame["WP"] = _whole_plot_ids(frame, block is not None, whole_plot is not None)
    if (_split_guard(frame, block is not None, result)
            and _apply_transform(frame, transform_kind, offset, result)):
        strata = _split_fit(frame, names, block is not None, result)
        if strata is not None:
            _split_predict(frame, names, strata, settings, result)
    return result


def _whole_plot_ids(frame: pd.DataFrame, has_block: bool, has_wp: bool) -> pd.Series:
    """Метка целой делянки: блок/делянка, делянка или блок/уровень A."""
    second = frame["wp"] if has_wp else frame["fa"]
    return frame["block"] + "/" + second if has_block else second


def _split_guard(frame: pd.DataFrame, has_block: bool, result: StatResult) -> bool:
    """A постоянен в целой делянке; в каждой делянке каждый уровень B ровно раз;
    у всех уровней A одинаковое число делянок (в каждом блоке); df > 0."""
    if frame["fa"].nunique() < 2 or frame["fb"].nunique() < 2:
        return _design_error(result, "factor_one_level", "У фактора меньше 2 уровней")
    wp_a = frame.groupby("WP")["fa"].nunique()
    cells = pd.crosstab(frame["WP"], frame["fb"])
    keys = ["fa"] + (["block"] if has_block else [])
    per_a = frame.drop_duplicates("WP").groupby(keys).size()
    n_cells = frame["fa"].nunique() * (frame["block"].nunique() if has_block else 1)
    if wp_a.max() > 1 or not cells.eq(1).all().all() or per_a.nunique() != 1 \
            or len(per_a) != n_cells:
        return _design_error(result, "split_plot_invalid",
                             "Не стандартные расщеплённые делянки: в целой делянке должен "
                             "быть один уровень A и каждый уровень B ровно один раз, у всех "
                             "уровней A — одинаковое число целых делянок")
    return True


def _split_fit(frame: pd.DataFrame, names: dict[str, str], has_block: bool,
               result: StatResult) -> dict | None:
    """Три страты (Table 9.11): SS по разностям RSS, F против остатка своей страты."""
    steps = ([("Block", [("block",)])] if has_block else []) + [
        ("A", [("fa",)]), ("WholePlot", [("WP",)]), ("B", [("fb",)]),
        ("A.B", [("fa", "fb")])]
    seq = _sequential(frame, steps)
    (wp_ss, wp_df), (res_ss, res_df) = seq["WholePlot"], seq["Residual"]
    if wp_df < 1 or res_df < 1 or _no_residual(res_ss, frame):
        _design_error(result, "n_too_small",
                      "Нет остаточных df в страте целых делянок или субделянок")
        return None
    sw2, s2 = wp_ss / wp_df, res_ss / res_df
    a, b = names["fa"], names["fb"]
    rows = [_line("Block", "block", *seq["Block"], (sw2, wp_df))] if has_block else []
    rows += [_line(a, "whole_plot", *seq["A"], (sw2, wp_df)),
             _line("WholePlot residual", "whole_plot", wp_ss, wp_df, (s2, res_df)),
             _line(b, "subplot", *seq["B"], (s2, res_df)),
             _line(f"{a}.{b}", "subplot", *seq["A.B"], (s2, res_df)),
             _line("Residual", "subplot", res_ss, res_df, None)]
    _rows_to_statistics(rows, _total_ss(frame), result)
    result.statistics.update({"ms_wholeplot": sw2, "df_wholeplot": wp_df})
    _split_residuals(frame, result)
    return {"sw2": sw2, "wp_df": wp_df, "s2": s2, "res_df": res_df}


def _split_residuals(frame: pd.DataFrame, result: StatResult) -> None:
    """Остатки страты субделянок (модель с целыми делянками как фактором)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ C(WP) + C(fb) + C(fa):C(fb)", data=frame).fit()
        _fill_residuals(fit, frame, result)


def satterthwaite_df(sw2: float, s2: float, t_b: int, wp_df: float, res_df: float) -> float:
    """df для (s_w² + (t_B − 1)s²): формула Саттертуэйта (Eq. 9.2)."""
    num = (sw2 + (t_b - 1) * s2) ** 2
    return float(num / (sw2**2 / wp_df + ((t_b - 1) * s2) ** 2 / res_df))


def _split_predict(frame: pd.DataFrame, names: dict[str, str], strata: dict,
                   settings: dict, result: StatResult) -> None:
    """Члены по маргинальности и таблицы средних с SE и df своих страт (Table 9.13)."""
    rows = {r["term"]: r for r in result.parameters["anova_table"]}
    a, b = names["fa"], names["fb"]
    p_values = {("fa",): rows[a]["p"], ("fb",): rows[b]["p"], ("fa", "fb"): rows[f"{a}.{b}"]["p"]}
    chosen = _select_prediction_terms(list(p_values), p_values, settings["alpha"])
    result.parameters["prediction_terms"] = [".".join(names[k] for k in t) for t in chosen]
    tables = [_split_table(frame, list(t), names, strata, settings["alpha"]) for t in chosen]
    if settings["transform_kind"] != _NONE:
        for table in tables:
            _back_transform_table(table, settings["transform_kind"], settings["offset"])
    result.parameters["tables"] = tables
    _tables_to_estimates(tables, settings["transform_kind"] != _NONE,
                         1 - settings["alpha"], result)


def _split_variances(keys: list[str], strata: dict, t_a: int, t_b: int, m: int) -> dict:
    """Дисперсии и df среднего и разностей для члена (стр. 225–227)."""
    sw2, s2, wp_df, res_df = strata["sw2"], strata["s2"], strata["wp_df"], strata["res_df"]
    if keys == ["fa"]:
        v = sw2 / (t_b * m)
        return {"mean": (v, wp_df), "diff_same_a": None, "diff": (2 * v, wp_df)}
    if keys == ["fb"]:
        v = s2 / (t_a * m)
        return {"mean": (v, res_df), "diff_same_a": None, "diff": (2 * v, res_df)}
    combined = (sw2 + (t_b - 1) * s2) / (t_b * m)
    df = satterthwaite_df(sw2, s2, t_b, wp_df, res_df)
    return {"mean": (combined, df), "diff_same_a": (2 * s2 / m, res_df),
            "diff": (2 * combined, df)}


def _split_table(frame: pd.DataFrame, keys: list[str], names: dict[str, str],
                 strata: dict, alpha: float) -> dict:
    """Средние члена с SE, ДИ и попарными сравнениями на нужных df."""
    t_a, t_b = frame["fa"].nunique(), frame["fb"].nunique()
    m = frame["WP"].nunique() // t_a
    var = _split_variances(keys, strata, t_a, t_b, m)
    means = frame.groupby(keys)["y"].mean()
    levels = [_natural_order(frame[k]) for k in keys]
    cells = [c for c in pd.MultiIndex.from_product(levels)] if len(keys) > 1 else levels[0]
    rows = [_split_mean_row(cell, keys, names, float(means.loc[cell]), var["mean"], alpha)
            for cell in cells]
    comps = [_split_comparison(rows[i], rows[j], cells[i], cells[j], keys, var, alpha)
             for i, j in combinations(range(len(cells)), 2)]
    return {"term": ".".join(names[k] for k in keys), "rows": rows, "comparisons": comps,
            "se_df": var["mean"][1]}


def _split_mean_row(cell, keys: list[str], names: dict[str, str], mean: float,
                    var: tuple[float, float], alpha: float) -> dict:
    """Среднее с SE и ДИ на df своей страты (или Саттертуэйта)."""
    levels = cell if isinstance(cell, tuple) else (cell,)
    se = float(np.sqrt(var[0]))
    t_crit = float(stats.t.ppf(1 - alpha / 2, var[1]))
    return {"label": " × ".join(f"{names[k]}={lv}" for k, lv in zip(keys, levels)),
            "mean": mean, "se": se, "df": var[1],
            "ci_lower": mean - t_crit * se, "ci_upper": mean + t_crit * se}


def _split_comparison(row_i: dict, row_j: dict, cell_i, cell_j, keys: list[str],
                      var: dict, alpha: float) -> dict:
    """Разность средних: SED и df — внутри одного уровня A или через уровни A."""
    same_a = len(keys) == 2 and cell_i[0] == cell_j[0]
    v, df = var["diff_same_a"] if same_a else var["diff"]
    sed = float(np.sqrt(v))
    diff = row_j["mean"] - row_i["mean"]
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    return {"group1": row_i["label"], "group2": row_j["label"], "mean_diff": diff,
            "sed": sed, "df": df, "lsd": t_crit * sed, "t": diff / sed,
            "p_unadjusted": float(2 * stats.t.sf(abs(diff / sed), df)),
            "ci_lower": diff - t_crit * sed, "ci_upper": diff + t_crit * sed}


# --- Сбалансированный неполный блочный дизайн ---------------------------------


def bibd_anova(
    values: list[float],
    group: list[str],
    block: list[str],
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Внутриблочный анализ BIBD (9.3) с межблочной SS вариантов в страте блоков."""
    settings = _settings(alpha, transform_kind, offset)
    names = _names(column_names, {"G": "group:Treatment", "block": "block:Block"})
    frame = _frame(values, {"G": group, "block": block})
    result = _new_result("bibd_anova", frame, settings, factor=names["G"])
    design = _bibd_design(frame, result)
    if design is None or not _apply_transform(frame, transform_kind, offset, result):
        return result
    result.parameters["design"] = {k: v for k, v in design.items() if k != "incidence"}
    if _bibd_fit(frame, names, design, result):
        _bibd_means(frame, design, settings, result)
    return result


def _bibd_design(frame: pd.DataFrame, result: StatResult) -> dict | None:
    """t, n, m, u, λ, EF; проверка t·n = m·u и n(u − 1) = λ(t − 1) (стр. 233–234)."""
    incidence = pd.crosstab(frame["block"], frame["G"])
    t, m = incidence.shape[1], incidence.shape[0]
    sizes, reps = incidence.sum(axis=1), incidence.sum(axis=0)
    concurrence = incidence.T @ incidence
    pairs = concurrence.to_numpy()[np.triu_indices(t, 1)]
    if incidence.to_numpy().max() > 1 or sizes.nunique() != 1 or reps.nunique() != 1 \
            or len(set(pairs)) != 1 or t < 3:
        _design_error(result, "bibd_invalid",
                      "Не BIBD: вариант не должен повторяться в блоке, блоки одного размера, "
                      "равная повторность и одинаковое число совместных появлений пар (λ)")
        return None
    u, n, lam = int(sizes.iloc[0]), int(reps.iloc[0]), int(pairs[0])
    if u >= t:
        _design_error(result, "bibd_complete_blocks",
                      "Блоки полные: это RCBD, используй блочный ANOVA")
        return None
    return {"t": t, "n": n, "m": m, "u": u, "lambda": lam, "N": len(frame),
            "efficiency_factor": lam * t / (n * u), "resolvable_possible": t % u == 0,
            "incidence": incidence}


def _bibd_fit(frame: pd.DataFrame, names: dict[str, str], design: dict,
              result: StatResult) -> bool:
    """Страты Block (варианты + остаток) и Block.Unit (варианты + остаток), Table 9.21."""
    seq = _sequential(frame, [("Block", [("block",)]), ("Treatment", [("G",)])])
    res_ss, res_df = seq["Residual"]
    if res_df < 1 or _no_residual(res_ss, frame):
        _design_error(result, "n_too_small", "Нет остаточных df внутри блоков")
        return False
    blk_ss = seq["Block"][0]
    btr_ss = _between_block_treatment_ss(frame, design)
    t, m = design["t"], design["m"]
    blk_res = (max(blk_ss - btr_ss, 0.0), m - t)
    denom_unit = (res_ss / res_df, res_df)
    denom_blk = (blk_res[0] / blk_res[1], blk_res[1]) if blk_res[1] > 0 else None
    rows = [_line(f"{names['G']} (between blocks)", "block", btr_ss, t - 1, denom_blk)]
    if blk_res[1] > 0:
        rows.append(_line("Block residual", "block", *blk_res, denom_unit))
    rows += [_line(names["G"], "unit", *seq["Treatment"], denom_unit),
            _line("Residual", "unit", res_ss, res_df, None)]
    _rows_to_statistics(rows, _total_ss(frame), result)
    if blk_res[1] == 0:
        result.add_warning("bibd_no_block_residual",
                           "m = t: в страте блоков нет остаточных df, межблочный F не "
                           "строится; объединение информации — смешанная модель (гл. 16)",
                           Severity.INFO)
    _fill_residuals(smf.ols("y ~ C(block) + C(G)", data=frame).fit(), frame, result)
    return True


def _between_block_treatment_ss(frame: pd.DataFrame, design: dict) -> float:
    """Межблочная SS вариантов: регрессия средних блоков на долю вариантов в блоке."""
    incidence = design["incidence"].to_numpy(float)
    block_means = frame.groupby("block")["y"].mean().reindex(design["incidence"].index)
    y = block_means.to_numpy(float)
    x = np.hstack([np.ones((len(y), 1)), incidence / design["u"]])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return float(design["u"] * (np.sum((y - y.mean()) ** 2) - resid @ resid))


def _bibd_means(frame: pd.DataFrame, design: dict, settings: dict,
                result: StatResult) -> None:
    """Внутриблочные средние ȳ + (ȳ_j − B̄_j)/EF, межблочные эффекты, SED, LSD (9.3.1)."""
    ef, n = design["efficiency_factor"], design["n"]
    grand = float(frame["y"].mean())
    levels = _natural_order(frame["G"])
    trt_mean = frame.groupby("G")["y"].mean()
    block_mean = frame.groupby("block")["y"].mean()
    inc = design["incidence"]
    b_bar = {g: float(block_mean[inc.index[inc[g] > 0]].mean()) for g in levels}
    adjusted = {g: grand + (float(trt_mean[g]) - b_bar[g]) / ef for g in levels}
    between = {g: (b_bar[g] - grand) / (1 - ef) for g in levels} if ef < 1 else None
    s2, df = result.statistics["ms_resid"], result.statistics["df_resid"]
    sed = float(np.sqrt(2 * s2 / (n * ef)))
    t_crit = float(stats.t.ppf(1 - settings["alpha"] / 2, df))
    comps = [_bibd_comparison(gi, gj, adjusted, sed, t_crit, df)
             for gi, gj in combinations(levels, 2)]
    result.statistics.update({"sed": sed, "lsd": t_crit * sed, "efficiency_factor": ef})
    result.parameters.update({
        "adjusted_means": adjusted, "unadjusted_means": trt_mean.reindex(levels).to_dict(),
        "between_block_effects": between, "comparisons": comps})
    for g in levels:
        result.estimates.append(Estimate(name=g, value=adjusted[g]))


def _bibd_comparison(gi: str, gj: str, adjusted: dict, sed: float, t_crit: float,
                     df: int) -> dict:
    """Разность скорректированных средних, SED = √(2s²/(n·EF)), ДИ и P на ResDF."""
    diff = adjusted[gj] - adjusted[gi]
    return {"group1": gi, "group2": gj, "mean_diff": diff, "sed": sed, "lsd": t_crit * sed,
            "t": diff / sed, "p_unadjusted": float(2 * stats.t.sf(abs(diff / sed), df)),
            "ci_lower": diff - t_crit * sed, "ci_upper": diff + t_crit * sed}
