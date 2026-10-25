"""Контрасты вариантов в ANOVA (гл. 8.6–8.7, 8.8.5 книги).

- ``treatment_contrasts`` — заданные пользователем контрасты ψ = Σ l_j μ_j
  (Example 8.4C–E): оценка, SE, t, ДИ, SS и F на 1 df, проверка
  ортогональности, остаток TrtSS; поправка Бонферрони или FDR для набора.
- ``polynomial_contrasts`` — ортогональные полиномиальные контрасты для
  количественных уровней (Example 8.6) с остатком (lack of fit) и
  коэффициентами полинома на исходной шкале уровней.

Один фактор вариантов, необязательный блок (RCBD). При блоках повторность
должна быть равной; без блоков (CRD) неравная повторность допустима.
Синтаксис контрастов: «A + C vs B; A vs C» (среднее против среднего) или
«F1=0.5, F2=-0.5, F3=-0.5, F4=0.5»; контрасты разделяются «;», имя
задаётся префиксом «имя: …».
"""

from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from core.report.schema import Estimate, Severity, StatResult
from core.stats.blocking import _fill_residuals
from core.stats.factorial import (
    _anova_rows,
    _apply_transform,
    _design_error,
    _frame,
    _is_balanced,
    _natural_order,
    _new_result,
    _no_residual,
    _rows_to_statistics,
    _settings,
)

_ADJUSTMENTS = ("none", "bonferroni", "fdr")
_POLY_NAMES = ("Linear", "Quadratic", "Cubic", "Quartic")
_TOL = 1e-9


# --- Общий однофакторный анализ ----------------------------------------------


def one_factor_fit(
    values: list[float], group: list[str], block: list[str] | None, settings: dict,
    method: str, column_names: dict[str, str] | None,
) -> tuple[StatResult, dict | None]:
    """ANOVA одного фактора (CRD или RCBD): таблица, остатки, средние уровней.

    Возвращает результат и сводку подгонки (уровни, средние, n, s², df, TrtSS)
    или None, если дизайн не подходит.
    """
    name = (column_names or {}).get("group", "Treatment")
    frame = _frame(values, {"G": group, "block": block})
    result = _new_result(method, frame, settings, factor=name)
    has_block = block is not None
    if not (_one_factor_guard(frame, has_block, result)
            and _apply_transform(frame, settings["transform_kind"], settings["offset"], result)):
        return result, None
    fit = smf.ols(f"y ~ {'C(block) + ' if has_block else ''}C(G)", data=frame).fit()
    table = anova_lm(fit)
    if _no_residual(float(table.loc["Residual", "sum_sq"]), frame):
        _design_error(result, "no_residual_variation", "Остаточной вариации нет")
        return result, None
    rows = _anova_rows(table, {name: "C(G)"}, has_block)
    _rows_to_statistics(rows, float(((frame["y"] - frame["y"].mean()) ** 2).sum()), result)
    _fill_residuals(fit, frame, result)
    return result, _summary(frame, name, rows, result)


def _one_factor_guard(frame: pd.DataFrame, has_block: bool, result: StatResult) -> bool:
    """≥ 2 уровней; при блоках — равная повторность в каждом блоке; df остатка > 0."""
    if frame["G"].nunique() < 2:
        return _design_error(result, "too_few_groups", "Нужно ≥ 2 уровней фактора")
    if has_block and not _is_balanced(frame, ["G", "block"], result):
        return False
    n_blocks = frame["block"].nunique() if has_block else 1
    if len(frame) - frame["G"].nunique() - (n_blocks - 1) <= 0:
        return _design_error(result, "n_too_small", "Нет повторностей: остаточных df нет")
    return True


def _summary(frame: pd.DataFrame, name: str, rows: list[dict], result: StatResult) -> dict:
    """Уровни в естественном порядке, их средние и повторности, s², df, TrtSS."""
    levels = _natural_order(frame["G"])
    grouped = frame.groupby("G")["y"]
    trt = next(r for r in rows if r["term"] == name)
    return {
        "name": name, "levels": levels,
        "means": grouped.mean().reindex(levels).to_numpy(float),
        "n": grouped.size().reindex(levels).to_numpy(float),
        "s2": result.statistics["ms_resid"], "df": result.statistics["df_resid"],
        "trt_ss": trt["ss"], "trt_df": trt["df"],
    }


def means_table(fit: dict, alpha: float) -> list[dict]:
    """Средние уровней с SE = √(s²/n) и ДИ на df остатка (стр. 84–85)."""
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = []
    for level, mean, n in zip(fit["levels"], fit["means"], fit["n"]):
        se = float(np.sqrt(fit["s2"] / n))
        rows.append({"level": level, "n": int(n), "mean": float(mean), "se": se,
                     "ci_lower": float(mean) - t_crit * se, "ci_upper": float(mean) + t_crit * se})
    return rows


# --- Разбор контрастов --------------------------------------------------------


def parse_contrasts(spec: str, levels: list[str]) -> list[tuple[str, np.ndarray]]:
    """Текст → [(имя, коэффициенты по уровням)]; ValueError при ошибке."""
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    if not parts:
        raise ValueError("Не задан ни один контраст")
    contrasts = []
    for i, part in enumerate(parts, 1):
        name, body = _split_name(part, i)
        coefs = _parse_sets(body, levels) if " vs " in f" {body} " else _parse_pairs(body, levels)
        if abs(coefs.sum()) > 1e-8:
            raise ValueError(f"Контраст «{name}»: сумма коэффициентов должна быть 0")
        if not np.any(coefs):
            raise ValueError(f"Контраст «{name}»: все коэффициенты нулевые")
        contrasts.append((name, coefs))
    return contrasts


def _split_name(part: str, index: int) -> tuple[str, str]:
    """«имя: тело» → (имя, тело); без имени — ψ1, ψ2…"""
    if ":" in part:
        name, body = part.split(":", 1)
        if name.strip():
            return name.strip(), body.strip()
    return f"ψ{index}", part


def _level_index(token: str, levels: list[str]) -> int:
    """Номер уровня по имени (без учёта регистра)."""
    lookup = {lv.lower(): i for i, lv in enumerate(levels)}
    key = token.strip().lower()
    if key not in lookup:
        raise ValueError(f"Уровень «{token.strip()}» не найден; есть: {', '.join(levels)}")
    return lookup[key]


def _parse_sets(body: str, levels: list[str]) -> np.ndarray:
    """«A + C vs B»: среднее первого набора минус среднее второго."""
    left, right = re.split(r"\s+vs\s+", body, maxsplit=1, flags=re.IGNORECASE)
    coefs = np.zeros(len(levels))
    for side, sign in ((left, 1.0), (right, -1.0)):
        idx = [_level_index(tok, levels) for tok in re.split(r"[+,]", side) if tok.strip()]
        if not idx:
            raise ValueError(f"Пустой набор уровней в «{body}»")
        coefs[idx] += sign / len(idx)
    return coefs


def _parse_pairs(body: str, levels: list[str]) -> np.ndarray:
    """«F1=0.5, F2=-0.5»: явные коэффициенты; не названные уровни — 0."""
    coefs = np.zeros(len(levels))
    for item in (x for x in body.split(",") if x.strip()):
        if "=" not in item:
            raise ValueError(f"Ожидалось «уровень=коэффициент» или «… vs …»: «{item.strip()}»")
        level, value = item.rsplit("=", 1)
        coefs[_level_index(level, levels)] = float(value.replace(" ", ""))
    return coefs


# --- Оценка контрастов --------------------------------------------------------


def _contrast_row(name: str, coefs: np.ndarray, fit: dict, t_crit: float) -> dict:
    """ψ̂ = Σ l ȳ, SE = √(s² Σ l²/n), t, P, ДИ; SS = ψ̂²/Σ(l²/n), F = SS/s² (стр. 183–184)."""
    psi = float(coefs @ fit["means"])
    var_factor = float(np.sum(coefs**2 / fit["n"]))
    se = float(np.sqrt(fit["s2"] * var_factor))
    t_stat = psi / se
    ss = psi**2 / var_factor
    scale = float(np.sqrt(np.sum(coefs**2)))
    return {"term": name, "coefficients": dict(zip(fit["levels"], coefs.tolist())),
            "estimate": psi, "se": se, "t": t_stat,
            # Стандартизованный контраст (Σ l² = 1), как его печатают пакеты (стр. 184).
            "estimate_standardized": psi / scale, "se_standardized": se / scale,
            "p": float(2 * stats.t.sf(abs(t_stat), fit["df"])),
            "ci_lower": psi - t_crit * se, "ci_upper": psi + t_crit * se,
            "df": 1, "ss": ss, "ms": ss, "F": ss / fit["s2"]}


def _orthogonality(contrasts: list[tuple[str, np.ndarray]], n: np.ndarray) -> list[dict]:
    """Пары неортогональных контрастов: Σ l_i l_k / n_j ≠ 0 (стр. 185)."""
    pairs = []
    for i in range(len(contrasts)):
        for k in range(i + 1, len(contrasts)):
            product = float(np.sum(contrasts[i][1] * contrasts[k][1] / n))
            if abs(product) > _TOL:
                pairs.append({"contrast1": contrasts[i][0], "contrast2": contrasts[k][0],
                              "product": product})
    return pairs


def _remainder(coef_matrix: np.ndarray, fit: dict) -> dict | None:
    """Остаток TrtSS после совместной подгонки контрастов (не зависит от порядка)."""
    psi = coef_matrix @ fit["means"]
    cov = (coef_matrix / fit["n"]) @ coef_matrix.T
    rank = int(np.linalg.matrix_rank(cov))
    df = fit["trt_df"] - rank
    if df <= 0:
        return None
    joint = float(psi @ np.linalg.pinv(cov) @ psi)
    ss = max(fit["trt_ss"] - joint, 0.0)
    f = ss / df / fit["s2"]
    return {"term": "Remainder", "df": df, "ss": ss, "ms": ss / df, "F": f,
            "p": float(stats.f.sf(f, df, fit["df"]))}


def _adjust(rows: list[dict], adjustment: str, alpha: float, fit: dict) -> None:
    """Поправка на множественность для набора контрастов (8.8.1, 8.8.2, 8.8.5)."""
    m = len(rows)
    raw = np.array([r["p"] for r in rows])
    if adjustment == "bonferroni":
        t_crit = float(stats.t.ppf(1 - alpha / (2 * m), fit["df"]))
        adjusted = np.minimum(raw * m, 1.0)
        for row in rows:
            row["ci_lower"] = row["estimate"] - t_crit * row["se"]
            row["ci_upper"] = row["estimate"] + t_crit * row["se"]
    elif adjustment == "fdr":
        adjusted = multipletests(raw, alpha=alpha, method="fdr_bh")[1]
    else:
        adjusted = raw
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(p_adj < alpha) if adjustment != "fdr" else bool(p_adj <= alpha)


def _contrasts_to_result(rows: list[dict], level: float, result: StatResult) -> None:
    """Оценки контрастов — в estimates, их F и P — в statistics."""
    for row in rows:
        result.estimates.append(Estimate(
            name=row["term"], value=row["estimate"], ci_lower=row["ci_lower"],
            ci_upper=row["ci_upper"], ci_level=level, std_error=row["se"]))
        result.statistics[f"F_{row['term']}"] = row["F"]
        result.statistics[f"p_{row['term']}"] = row["p"]


def _insert_partition(result: StatResult, name: str, parts: list[dict]) -> None:
    """Строки разбиения TrtSS — сразу после строки фактора в таблице ANOVA."""
    table = result.parameters["anova_table"]
    pos = next(i for i, r in enumerate(table) if r["term"] == name) + 1
    for offset, part in enumerate(parts):
        row = {k: part[k] for k in ("term", "df", "ss", "ms", "F", "p")}
        table.insert(pos + offset, row | {"stratum": "within_term", "parent": name})


def treatment_contrasts(
    values: list[float],
    group: list[str],
    block: list[str] | None = None,
    contrasts: str = "",
    alpha: float = 0.05,
    adjustment: str = "none",
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Контрасты вариантов с разбиением TrtSS (Example 8.4C–E).

    ψ̂ и ДИ — на шкале анализа (при трансформации — трансформированной).
    """
    if adjustment not in _ADJUSTMENTS:
        raise ValueError(f"adjustment ∈ {_ADJUSTMENTS}")
    settings = _settings(alpha, transform_kind, offset)
    result, fit = one_factor_fit(values, group, block, settings, "treatment_contrasts",
                                 column_names)
    if fit is None:
        return result
    try:
        parsed = parse_contrasts(contrasts, fit["levels"])
    except ValueError as exc:
        _design_error(result, "contrast_invalid", str(exc))
        return result
    _evaluate_contrasts(parsed, fit, settings, adjustment, result)
    return result


def _evaluate_contrasts(parsed: list[tuple[str, np.ndarray]], fit: dict, settings: dict,
                        adjustment: str, result: StatResult) -> None:
    """Строки контрастов, ортогональность, остаток, поправка, вывод в результат."""
    alpha = settings["alpha"]
    t_crit = float(stats.t.ppf(1 - alpha / 2, fit["df"]))
    rows = [_contrast_row(name, coefs, fit, t_crit) for name, coefs in parsed]
    _adjust(rows, adjustment, alpha, fit)
    nonorth = _orthogonality(parsed, fit["n"])
    if nonorth:
        result.add_warning(
            "contrasts_not_orthogonal",
            "Контрасты неортогональны: их SS не складываются, вклад каждого зависит от "
            "остальных. Остаток посчитан по совместной подгонке.",
            Severity.INFO, params={"pairs": len(nonorth)})
    remainder = _remainder(np.vstack([c for _, c in parsed]), fit)
    _insert_partition(result, fit["name"], rows + ([remainder] if remainder else []))
    result.parameters.update({"contrasts": rows, "remainder": remainder,
                              "non_orthogonal_pairs": nonorth, "adjustment": adjustment,
                              "means": means_table(fit, alpha)})
    _contrasts_to_result(rows, 1 - alpha, result)


# --- Полиномиальные контрасты -------------------------------------------------


def orthogonal_polynomials(x: np.ndarray, n: np.ndarray, degree: int) -> np.ndarray:
    """Коэффициенты ортогональных полиномов степени 1…degree (Table 8.24).

    Грам–Шмидт по степеням центрированного x с весами повторности:
    Σ n_j P_q(x_j) P_r(x_j) = 0 и Σ n_j P_q(x_j) = 0.
    """
    xc = x - np.average(x, weights=n)
    basis = [np.ones_like(x)]
    for q in range(1, degree + 1):
        p = xc**q
        for b in basis:
            p = p - (np.sum(n * p * b) / np.sum(n * b * b)) * b
        basis.append(p)
    return np.vstack(basis[1:])


def _numeric_levels(levels: list[str]) -> np.ndarray:
    """Уровни как числа (десятичная запятая допустима); ValueError иначе."""
    try:
        return np.array([float(lv.replace(",", ".")) for lv in levels])
    except ValueError as exc:
        raise ValueError("Уровни фактора должны быть числами (дозы, концентрации)") from exc


def _trend_fits(x: np.ndarray, fit: dict, degree: int) -> list[dict]:
    """Полиномы порядка 1…degree по средним уровней (веса — повторность), стр. 195."""
    fits = []
    for q in range(1, degree + 1):
        coefs = np.polynomial.polynomial.polyfit(x, fit["means"], q, w=np.sqrt(fit["n"]))
        fits.append({"order": q, "coefficients": [float(c) for c in coefs]})
    return fits


def polynomial_contrasts(
    values: list[float],
    group: list[str],
    block: list[str] | None = None,
    degree: int = 2,
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Линейный, квадратичный… тренд по количественным уровням с остатком (Example 8.6)."""
    settings = _settings(alpha, transform_kind, offset)
    result, fit = one_factor_fit(values, group, block, settings, "polynomial_contrasts",
                                 column_names)
    if fit is None:
        return result
    try:
        x = _numeric_levels(fit["levels"])
    except ValueError as exc:
        _design_error(result, "levels_not_numeric", str(exc))
        return result
    degree = int(min(max(degree, 1), len(x) - 1, len(_POLY_NAMES)))
    poly = orthogonal_polynomials(x, fit["n"], degree)
    parsed = [(_POLY_NAMES[q], poly[q]) for q in range(degree)]
    _evaluate_contrasts(parsed, fit, settings, "none", result)
    result.parameters.update({"degree": degree, "x": x.tolist(),
                              "trend_fits": _trend_fits(x, fit, degree)})
    return result


# --- Контрасты в факторной структуре (Example 8.5) ----------------------------


def _subset_effects(cells: pd.DataFrame, cols: list[str]) -> dict[tuple[str, ...], np.ndarray]:
    """Эффекты sum-to-zero величины c по всем подмножествам cols, по строкам cells."""
    effects: dict[tuple[str, ...], np.ndarray] = {(): np.full(len(cells), cells["c"].mean())}
    for size in range(1, len(cols) + 1):
        for subset in combinations(cols, size):
            means = cells.groupby(list(subset))["c"].transform("mean").to_numpy()
            lower = [s for k in range(size) for s in combinations(subset, k)]
            effects[subset] = means - sum(effects[s] for s in lower)
    return effects


def factorial_contrast_rows(
    frame: pd.DataFrame, keys: list[str], names: dict[str, str], spec: str,
    s2: float, df: int,
) -> tuple[list[dict], list[dict]]:
    """Разбиение членов с фактором A на контрасты: ψ и ψ.B, ψ.C… (стр. 187–190).

    Возвращает строки ANOVA (с полем parent) и оценки главных контрастов.
    """
    a, others = keys[0], keys[1:]
    levels = _natural_order(frame[a])
    parsed = parse_contrasts(spec, levels)
    if _orthogonality(parsed, np.ones(len(levels))):
        raise ValueError("Контрасты в факторной структуре должны быть ортогональны")
    n_cell = int(frame.groupby(keys).size().iloc[0])
    table = frame.groupby(keys)["y"].mean().unstack(a)[levels]
    cells = table.index.to_frame(index=False)
    rows, estimates = [], []
    for name, coefs in parsed:
        cells["c"] = table.to_numpy() @ coefs
        weight = n_cell / float(np.sum(coefs**2))
        for subset, eff in _subset_effects(cells, others).items():
            parent = ".".join(names[k] for k in (a, *subset))
            term = ".".join([name if not subset else f"({name})", *(names[k] for k in subset)])
            ss = weight * float(np.sum(eff**2))
            dof = int(np.prod([cells[k].nunique() - 1 for k in subset])) if subset else 1
            rows.append(_partition_row(term, parent, ss, dof, s2, df))
        estimates.append(_main_contrast(name, coefs, cells, n_cell, s2, df))
    return rows, estimates


def _partition_row(term: str, parent: str, ss: float, dof: int, s2: float, df: int) -> dict:
    """Строка разбиения: SS, MS, F против ResMS, P."""
    f = ss / dof / s2
    return {"term": term, "parent": parent, "stratum": "within_term", "df": dof,
            "ss": ss, "ms": ss / dof, "F": f, "p": float(stats.f.sf(f, dof, df))}


def _main_contrast(name: str, coefs: np.ndarray, cells: pd.DataFrame, n_cell: int,
                   s2: float, df: int) -> dict:
    """ψ̂ по маргинальным средним A, SE = √(s² Σl² / (n · число ячеек прочих факторов))."""
    psi = float(cells["c"].mean())
    se = float(np.sqrt(s2 * np.sum(coefs**2) / (n_cell * len(cells))))
    t_stat = psi / se
    return {"term": name, "estimate": psi, "se": se, "t": t_stat,
            "p": float(2 * stats.t.sf(abs(t_stat), df)),
            "by_cell": [float(v) for v in cells["c"] - psi]}


def add_remainders(table: list[dict], parts: list[dict]) -> list[dict]:
    """Остаток каждого разбитого члена: SS члена минус SS его контрастов."""
    s2 = next(r["ms"] for r in table if r["term"] == "Residual")
    res_df = next(r["df"] for r in table if r["term"] == "Residual")
    out = []
    for parent in dict.fromkeys(p["parent"] for p in parts):
        mine = [p for p in parts if p["parent"] == parent]
        line = next(r for r in table if r["term"] == parent)
        dof = line["df"] - sum(p["df"] for p in mine)
        out += mine
        if dof > 0:
            ss = max(line["ss"] - sum(p["ss"] for p in mine), 0.0)
            out.append(_partition_row(f"Remainder[{parent}]", parent, ss, dof, s2, res_df))
    return out
