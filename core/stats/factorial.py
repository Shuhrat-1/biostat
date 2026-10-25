"""Факторные и вложенные структуры вариантов (гл. 8.1–8.5 книги).

- ``factorial_anova`` — перекрёстная структура A*B или A*B*C (8.2–8.3):
  главные эффекты и взаимодействия, выбор членов для предсказания по
  принципу маргинальности (стр. 166–167), таблицы средних с SE, SED, LSD.
- ``nested_anova`` — вложенная структура Group/Member (8.4) и контроль
  против остальных вариантов как Type/Treatment (8.5), с разбиением
  вложенной суммы квадратов по группам (Example 8.3B).

Оба метода требуют ортогональности: полный набор комбинаций с равной
повторностью в каждом блоке (иначе — гл. 11). Блок необязателен и
выносится в свою страту (multi-stratum ANOVA, гл. 7.5). Суммы квадратов —
последовательные из statsmodels; при ортогональности они однозначны.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from core.report.schema import Estimate, Severity, StatResult
from core.stats.anova import _lsd_comparisons
from core.stats.blocking import _fill_residuals
from core.stats.transforms import TRANSFORMS, back_transform, transform

_NONE = "none"
_LOG_BASE = {"log10": 10.0, "ln": np.e}
# Исключённое взаимодействие с P ниже этого порога отмечается (эвристика BioStat
# по мотивам Example 8.1C, где P = 0.058 обсуждается как «пограничный» случай).
_BORDERLINE = 0.10


# --- Общая подготовка данных -------------------------------------------------


def _frame(values: list[float], factors: dict[str, list[str] | None]) -> pd.DataFrame:
    """Отклик и факторы в одну таблицу; строки с пропуском в любой колонке удаляются."""
    data = {"y": values} | {k: v for k, v in factors.items() if v is not None}
    if len({len(v) for v in data.values()}) != 1:
        raise ValueError("Колонки должны быть одной длины")
    frame = pd.DataFrame(data)
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    for col in frame.columns.drop("y"):
        frame[col] = frame[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    return frame.dropna().reset_index(drop=True)


def _natural_order(column: pd.Series) -> list[str]:
    """Уровни по порядку: числовые коды по значению (5 < 10 < 20), остальные по алфавиту."""
    def key(level: str) -> tuple[int, float, str]:
        try:
            return (0, float(level.replace(",", ".")), level)
        except ValueError:
            return (1, 0.0, level)
    return sorted(column.unique(), key=key)


def _settings(alpha: float, transform_kind: str, offset: float) -> dict:
    """Проверить уровень значимости и вид трансформации, собрать настройки."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должен быть в (0, 1)")
    if transform_kind not in (_NONE, *TRANSFORMS):
        raise ValueError(f"transform_kind ∈ {(_NONE, *TRANSFORMS)}")
    return {"alpha": alpha, "transform_kind": transform_kind, "offset": offset}


def _new_result(method: str, frame: pd.DataFrame, settings: dict, **extra: object) -> StatResult:
    """Пустой результат с настройками анализа в parameters."""
    return StatResult(method=method, n=len(frame), parameters={
        "alpha": settings["alpha"], "transform": settings["transform_kind"],
        "offset": settings["offset"], **extra})


def _apply_transform(frame: pd.DataFrame, kind: str, offset: float, result: StatResult) -> bool:
    """Трансформировать отклик (гл. 6); при значениях вне области — ошибка."""
    if kind == _NONE:
        return True
    try:
        frame["y"] = transform(frame["y"].to_numpy(float), kind, offset)
    except ValueError as exc:
        result.add_warning("transform_domain", str(exc), Severity.ERROR)
        return False
    return True


def _design_error(result: StatResult, code: str, message: str) -> bool:
    """Записать ошибку дизайна и вернуть False."""
    result.add_warning(code, message, Severity.ERROR)
    return False


def _no_residual(ssr: float, frame: pd.DataFrame) -> bool:
    """Остаточная SS — ноль с точностью до округления."""
    total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    return ssr <= 1e-12 * max(total, 1.0)


def _is_balanced(frame: pd.DataFrame, keys: list[str], result: StatResult) -> bool:
    """Все комбинации уровней (с блоком) есть и повторены одинаково (стр. 155, 172)."""
    sizes = frame.groupby(keys).size()
    n_cells = int(np.prod([frame[k].nunique() for k in keys]))
    if len(sizes) != n_cells or sizes.nunique() != 1:
        return _design_error(
            result, "factorial_unbalanced",
            "Не все комбинации уровней присутствуют с одинаковой повторностью "
            "(в каждом блоке): структура неортогональна, нужен анализ гл. 11",
        )
    return True


# --- Таблица ANOVA ------------------------------------------------------------


def _anova_rows(
    table: pd.DataFrame, terms: dict[str, str], has_block: bool
) -> list[dict]:
    """Строки ANOVA по стратам: блок против остатка, члены вариантов, остаток."""
    res_ms, res_df = float(table.loc["Residual", "mean_sq"]), int(table.loc["Residual", "df"])
    trt_ss = sum(float(table.loc[f, "sum_sq"]) for f in terms.values())
    rows = []
    if has_block:
        rows.append(_row("Block", "block", table.loc["C(block)"], res_ms, res_df, None))
    for name, formula_term in terms.items():
        rows.append(_row(name, "unit", table.loc[formula_term], res_ms, res_df, trt_ss))
    rows.append({"term": "Residual", "stratum": "unit", "df": res_df,
                 "ss": float(table.loc["Residual", "sum_sq"]), "ms": res_ms})
    return rows


def _row(name: str, stratum: str, line: pd.Series, res_ms: float, res_df: int,
         trt_ss: float | None) -> dict:
    """Одна строка ANOVA: df, SS, MS, F против остатка, P, доля в TrtSS."""
    df, ss = int(line["df"]), float(line["sum_sq"])
    f = float((ss / df) / res_ms)
    row = {"term": name, "stratum": stratum, "df": df, "ss": ss, "ms": ss / df,
           "F": f, "p": float(stats.f.sf(f, df, res_df))}
    if trt_ss:
        row["ss_share"] = ss / trt_ss
    return row


def _rows_to_statistics(rows: list[dict], total_ss: float, result: StatResult) -> None:
    """Плоские статистики для отображения: F и P каждого члена, остаток, итог."""
    for row in rows:
        if row["term"] == "Residual":
            result.statistics.update({"df_resid": row["df"], "ms_resid": row["ms"],
                                      "ss_resid": row["ss"]})
        else:
            result.statistics.update({f"F_{row['term']}": row["F"], f"p_{row['term']}": row["p"]})
    result.statistics["ss_total"] = total_ss
    result.parameters["anova_table"] = rows


# --- Таблицы средних ----------------------------------------------------------


def _means_table(
    frame: pd.DataFrame, keys: list[str], names: dict[str, str], s2: float, df: int,
    alpha: float,
) -> dict:
    """Средние ячеек члена с SE, ДИ, попарными SED/LSD на df остатка (стр. 160–162)."""
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    labels, groups = [], []
    ordered = frame.assign(**{k: pd.Categorical(frame[k], _natural_order(frame[k])) for k in keys})
    for levels, cell in ordered.groupby(keys, sort=True, observed=True)["y"]:
        levels = levels if isinstance(levels, tuple) else (levels,)
        labels.append(" × ".join(f"{names[k]}={lv}" for k, lv in zip(keys, levels)))
        groups.append(cell.to_numpy(float))
    rows = [_mean_row(lab, g, s2, t_crit) for lab, g in zip(labels, groups)]
    table = {"term": ".".join(names[k] for k in keys), "rows": rows, "t_crit": t_crit,
             "comparisons": _lsd_comparisons(groups, labels, s2, df, t_crit)}
    if len({g.size for g in groups}) == 1:
        sed = float(np.sqrt(2 * s2 / groups[0].size))
        table.update({"sed": sed, "lsd": t_crit * sed})
    return table


def _mean_row(label: str, cell: np.ndarray, s2: float, t_crit: float) -> dict:
    """Среднее ячейки, SE = √(s²/n) и ДИ."""
    mean, se = float(cell.mean()), float(np.sqrt(s2 / cell.size))
    return {"label": label, "n": int(cell.size), "mean": mean, "se": se,
            "ci_lower": mean - t_crit * se, "ci_upper": mean + t_crit * se}


def _back_transform_table(table: dict, kind: str, offset: float) -> None:
    """Средние и границы ДИ на исходной шкале; для log — отношения (гл. 6.3–6.4)."""
    for row in table["rows"]:
        for key in ("mean", "ci_lower", "ci_upper"):
            row[f"back_{key}"] = float(back_transform(row[key], kind, offset))
    if kind not in _LOG_BASE:
        return
    base = _LOG_BASE[kind]
    for comp in table["comparisons"]:
        comp.update({"ratio": float(base ** comp["mean_diff"]),
                     "ratio_ci_lower": float(base ** comp["ci_lower"]),
                     "ratio_ci_upper": float(base ** comp["ci_upper"])})
    if "lsd" in table:
        table["lsd_factor"] = float(base ** table["lsd"])


def _tables_to_estimates(tables: list[dict], transformed: bool, level: float,
                         result: StatResult) -> None:
    """Средние таблиц предсказания — в estimates (на исходной шкале при трансформации)."""
    prefix = "back_" if transformed else ""
    for table in tables:
        for row in table["rows"]:
            result.estimates.append(Estimate(
                name=row["label"], value=row[f"{prefix}mean"],
                ci_lower=row[f"{prefix}ci_lower"], ci_upper=row[f"{prefix}ci_upper"],
                ci_level=level, std_error=None if transformed else row["se"],
            ))


def _fill_tables(frame: pd.DataFrame, term_keys: list[list[str]], names: dict[str, str],
                 settings: dict, result: StatResult) -> None:
    """Таблицы средних выбранных членов, их обратное преобразование и estimates."""
    s2, df = result.statistics["ms_resid"], result.statistics["df_resid"]
    tables = [_means_table(frame, keys, names, s2, df, settings["alpha"]) for keys in term_keys]
    kind = settings["transform_kind"]
    if kind != _NONE:
        for table in tables:
            _back_transform_table(table, kind, settings["offset"])
    result.parameters["tables"] = tables
    _tables_to_estimates(tables, kind != _NONE, 1 - settings["alpha"], result)


# --- Факторная структура ------------------------------------------------------


def _terms(keys: list[str]) -> list[tuple[str, ...]]:
    """Все члены A*B(*C) по возрастанию порядка: A, B, C, A.B, A.C, B.C, A.B.C."""
    return [t for size in range(1, len(keys) + 1) for t in combinations(keys, size)]


def _select_prediction_terms(
    terms: list[tuple[str, ...]], p_values: dict[tuple[str, ...], float], alpha: float
) -> list[tuple[str, ...]]:
    """Отбор снизу вверх по маргинальности (стр. 166–167, Fig. 8.4).

    На каждом шаге проверяются только члены, не маргинальные к оставшимся;
    незначимые удаляются. Предсказание — по максимальным оставшимся членам.
    """
    remaining, kept = list(terms), set()
    while True:
        eligible = [t for t in remaining if t not in kept
                    and not any(set(t) < set(u) for u in remaining)]
        if not eligible:
            break
        for term in eligible:
            if p_values[term] < alpha:
                kept.add(term)
            else:
                remaining.remove(term)
    return [t for t in remaining if not any(set(t) < set(u) for u in remaining)]


def _effects(frame: pd.DataFrame, terms: list[tuple[str, ...]], names: dict[str, str]) -> dict:
    """Эффекты sum-to-zero по таблицам средних (стр. 156): средняя ячейки минус подчлены."""
    grand = float(frame["y"].mean())
    effects: dict[tuple[str, ...], dict[tuple[str, ...], float]] = {(): {(): grand}}
    for term in terms:
        means = frame.groupby(list(term))["y"].mean()
        effects[term] = {}
        for levels, mean in means.items():
            levels = levels if isinstance(levels, tuple) else (levels,)
            cell = dict(zip(term, levels))
            subs = [s for size in range(len(term)) for s in combinations(term, size)]
            effects[term][levels] = float(mean) - sum(
                effects[s][tuple(cell[k] for k in s)] for s in subs)
    return {".".join(names[k] for k in term): {
        " × ".join(map(str, lv)): v for lv, v in effects[term].items()} for term in terms}


def _flag_borderline(rows: list[dict], predicted: list[str], result: StatResult) -> None:
    """Исключённое взаимодействие с P < 0.10 — пометка (стр. 159–160, 171)."""
    for row in rows:
        is_interaction = "." in row["term"] and row["stratum"] == "unit"
        excluded = not any(set(row["term"].split(".")) <= set(p.split(".")) for p in predicted)
        if is_interaction and excluded and row.get("p", 1.0) < _BORDERLINE:
            result.add_warning(
                "interaction_borderline",
                f"Взаимодействие {row['term']} исключено из предсказания, но P = "
                f"{row['p']:.3f} близко к порогу: сравни его вклад (доля SS) с главными "
                f"эффектами и реши, можно ли им пренебречь.",
                Severity.INFO, params={"term": row["term"], "p": round(row["p"], 3)},
            )


def factorial_anova(
    values: list[float],
    factor_a: list[str],
    factor_b: list[str],
    factor_c: list[str] | None = None,
    block: list[str] | None = None,
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
    contrasts: str = "",
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Факторный ANOVA A*B(*C) с блоком (гл. 8.2–8.3); члены — по маргинальности.

    Средние и SE — по исходному ResMS (стр. 160). ``contrasts`` — ортогональные
    контрасты уровней A, разбивают A и его взаимодействия (Example 8.5).
    """
    settings = _settings(alpha, transform_kind, offset)
    factors = {"fa": factor_a, "fb": factor_b, "fc": factor_c}
    keys = [k for k, v in factors.items() if v is not None]
    names = _names(column_names, {"fa": "factor_a", "fb": "factor_b", "fc": "factor_c"}, keys)
    frame = _frame(values, factors | {"block": block})
    result = _new_result("factorial_anova", frame, settings,
                         factors=[names[k] for k in keys])
    if (_factorial_guard(frame, keys, block is not None, result)
            and _apply_transform(frame, transform_kind, offset, result)):
        _factorial_fit(frame, keys, names, block is not None, result)
        if result.statistics.get("df_resid", 0) > 0:
            _factorial_predict(frame, keys, names, settings, result)
            if contrasts.strip():
                _factorial_contrasts(frame, keys, names, contrasts, result)
    return result


def _factorial_contrasts(frame: pd.DataFrame, keys: list[str], names: dict[str, str],
                         spec: str, result: StatResult) -> None:
    """Строки контрастов фактора A и их взаимодействий — внутрь таблицы ANOVA."""
    from core.stats.contrasts import add_remainders, factorial_contrast_rows

    table = result.parameters["anova_table"]
    try:
        parts, estimates = factorial_contrast_rows(
            frame, keys, names, spec, result.statistics["ms_resid"],
            result.statistics["df_resid"])
    except ValueError as exc:
        result.add_warning("contrast_invalid", str(exc), Severity.ERROR)
        return
    for part in add_remainders(table, parts):
        pos = max(i for i, r in enumerate(table)
                  if r["term"] == part["parent"] or r.get("parent") == part["parent"])
        table.insert(pos + 1, part)
    result.parameters["contrasts"] = estimates


def _names(column_names: dict[str, str] | None, slots: dict[str, str],
           keys: list[str]) -> dict[str, str]:
    """Отображаемые имена факторов: имена колонок из моста или A, B, C (G, M)."""
    column_names = column_names or {}
    default = {"fa": "A", "fb": "B", "fc": "C", "G": "Group", "M": "Member"}
    return {k: column_names.get(slots[k], default[k]) for k in keys}


def _factorial_guard(frame: pd.DataFrame, keys: list[str], has_block: bool,
                     result: StatResult) -> bool:
    """≥ 2 уровней у каждого фактора, ортогональность, df остатка > 0."""
    if any(frame[k].nunique() < 2 for k in keys):
        return _design_error(result, "factor_one_level", "У фактора меньше 2 уровней")
    if not _is_balanced(frame, keys + (["block"] if has_block else []), result):
        return False
    n_cells = int(np.prod([frame[k].nunique() for k in keys]))
    n_blocks = frame["block"].nunique() if has_block else 1
    if len(frame) - n_cells - (n_blocks - 1) <= 0:
        return _design_error(result, "n_too_small",
                             "Нет повторностей: остаточных степеней свободы нет")
    return True


def _factorial_fit(frame: pd.DataFrame, keys: list[str], names: dict[str, str],
                   has_block: bool, result: StatResult) -> None:
    """Последовательная ANOVA statsmodels, строки по стратам, эффекты, остатки."""
    rhs = "*".join(f"C({k})" for k in keys)
    fit = smf.ols(f"y ~ {'C(block) + ' if has_block else ''}{rhs}", data=frame).fit()
    table = anova_lm(fit)
    if _no_residual(float(table.loc["Residual", "sum_sq"]), frame):
        _design_error(result, "no_residual_variation", "Остаточной вариации нет")
        return
    terms = {".".join(names[k] for k in t): ":".join(f"C({k})" for k in t) for t in _terms(keys)}
    rows = _anova_rows(table, terms, has_block)
    total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    _rows_to_statistics(rows, total, result)
    result.parameters["effects"] = _effects(frame, _terms(keys), names)
    _fill_residuals(fit, frame, result)


def _factorial_predict(frame: pd.DataFrame, keys: list[str], names: dict[str, str],
                       settings: dict, result: StatResult) -> None:
    """Члены для предсказания по маргинальности и их таблицы средних."""
    rows = {r["term"]: r for r in result.parameters["anova_table"]}
    terms = _terms(keys)
    p_values = {t: rows[".".join(names[k] for k in t)]["p"] for t in terms}
    chosen = _select_prediction_terms(terms, p_values, settings["alpha"])
    labels = [".".join(names[k] for k in t) for t in chosen]
    result.parameters["prediction_terms"] = labels
    if not chosen:
        result.add_warning("no_significant_terms",
                           "Ни один член не значим: предсказание — общее среднее",
                           Severity.INFO)
        return
    _fill_tables(frame, [list(t) for t in chosen], names, settings, result)
    _flag_borderline(result.parameters["anova_table"], labels, result)


# --- Вложенная структура ------------------------------------------------------


def nested_anova(
    values: list[float],
    group: list[str],
    member: list[str],
    block: list[str] | None = None,
    alpha: float = 0.05,
    transform_kind: str = _NONE,
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Вложенный ANOVA Group/Member с необязательным блоком (гл. 8.4–8.5).

    Член группы определяется парой (группа, член): нумерация может
    начинаться заново в каждой группе. Вложенная SS разбивается по группам.
    """
    settings = _settings(alpha, transform_kind, offset)
    names = _names(column_names, {"G": "group", "M": "member"}, ["G", "M"])
    frame = _frame(values, {"G": group, "M": member, "block": block})
    frame["M"] = frame["G"] + "/" + frame["M"]
    result = _new_result("nested_anova", frame, settings, group=names["G"], member=names["M"])
    if not (_nested_guard(frame, block is not None, result)
            and _apply_transform(frame, transform_kind, offset, result)):
        return result
    _nested_fit(frame, names, block is not None, result)
    if result.statistics.get("df_resid", 0) > 0:
        _fill_tables(frame, [["G"], ["M"]], names, settings, result)
        # В estimates — только средние групп; средние членов остаются в tables.
        result.estimates = [e for e in result.estimates if e.name.startswith(f"{names['G']}=")]
        result.parameters["effects"] = _nested_effects(frame, names)
    return result


def _nested_guard(frame: pd.DataFrame, has_block: bool, result: StatResult) -> bool:
    """≥ 2 групп, хотя бы одна группа с ≥ 2 членами, равная повторность, df > 0."""
    if frame["G"].nunique() < 2:
        return _design_error(result, "too_few_groups", "Нужно ≥ 2 групп")
    if frame.groupby("G")["M"].nunique().max() < 2:
        return _design_error(result, "nested_no_members",
                             "Ни в одной группе нет двух членов: вложенный член пуст")
    if not _is_balanced(frame, ["M"] + (["block"] if has_block else []), result):
        return False
    n_blocks = frame["block"].nunique() if has_block else 1
    if len(frame) - frame["M"].nunique() - (n_blocks - 1) <= 0:
        return _design_error(result, "n_too_small",
                             "Нет повторностей: остаточных степеней свободы нет")
    return True


def _nested_fit(frame: pd.DataFrame, names: dict[str, str], has_block: bool,
                result: StatResult) -> None:
    """SS групп и вложенного члена как разности RSS последовательных моделей."""
    base = "y ~ C(block)" if has_block else "y ~ 1"
    m0 = smf.ols(base, data=frame).fit()
    m1 = smf.ols(f"{base} + C(G)", data=frame).fit()
    m2 = smf.ols(f"{base} + C(M)", data=frame).fit()
    if _no_residual(m2.ssr, frame):
        _design_error(result, "no_residual_variation", "Остаточной вариации нет")
        return
    res_ms, res_df = m2.ssr / m2.df_resid, int(m2.df_resid)
    g_name, nest_name = names["G"], f"{names['G']}.{names['M']}"
    n_groups, n_members = frame["G"].nunique(), frame["M"].nunique()
    lines = []
    if has_block:
        lines.append(("Block", "block", m0.ess, frame["block"].nunique() - 1))
    lines += [(g_name, "unit", m0.ssr - m1.ssr, n_groups - 1),
              (nest_name, "unit", m1.ssr - m2.ssr, n_members - n_groups)]
    lines += _per_group_lines(frame, base, m2.ssr, nest_name)
    rows = [_row(n, s, pd.Series({"df": d, "sum_sq": ss}), res_ms, res_df, None)
            for n, s, ss, d in lines]
    rows.append({"term": "Residual", "stratum": "unit", "df": res_df, "ss": float(m2.ssr),
                 "ms": float(res_ms)})
    total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    _rows_to_statistics(rows, total, result)
    _fill_residuals(m2, frame, result)


def _per_group_lines(frame: pd.DataFrame, base: str, full_ssr: float,
                     nest_name: str) -> list[tuple[str, str, float, int]]:
    """Вложенная SS внутри каждой группы (Example 8.3B): слияние её членов в один уровень.

    Группа из одного члена не даёт вложенных df (стр. 177); если разбивать
    нечего (≤ 1 группы с ≥ 2 членами), строки не добавляются — они повторили бы
    общий вложенный член (Example 8.4B).
    """
    counts = frame.groupby("G")["M"].nunique()
    if (counts >= 2).sum() < 2:
        return []
    lines = []
    for grp in counts[counts >= 2].index:
        collapsed = frame.assign(Mc=np.where(frame["G"] == grp, "__group__", frame["M"]))
        ssr = smf.ols(f"{base} + C(Mc)", data=collapsed).fit().ssr
        lines.append((f"{nest_name}[{grp}]", "within_group", ssr - full_ssr,
                      int(counts[grp]) - 1))
    return lines


def _nested_effects(frame: pd.DataFrame, names: dict[str, str]) -> dict:
    """Родительские γ_r = ȳ_r − ȳ и вложенные δ_rs = ȳ_rs − ȳ_r (стр. 174)."""
    grand = float(frame["y"].mean())
    g_means = frame.groupby("G")["y"].mean()
    m_means = frame.groupby(["G", "M"])["y"].mean()
    return {
        names["G"]: {g: float(m - grand) for g, m in g_means.items()},
        f"{names['G']}.{names['M']}": {
            m: float(v - g_means[g]) for (g, m), v in m_means.items()
        },
    }
