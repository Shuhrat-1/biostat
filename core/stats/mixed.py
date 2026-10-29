"""Линейные смешанные модели (гл. 16 книги).

``mixed_model`` — фиксированная модель из 1–3 факторов (полное скрещивание) и
случайная модель из вложенных структурных факторов (Block/WholePlot, Farm/
Field): REML-компоненты с SE (Table 16.3), фиксированные эффекты first-level-zero
(Table 16.1), инкрементные и маргинальные тесты Вальда и приближённые F
с df Кенварда–Роджера (Table 16.2), маргинальные предсказания с SE и SED
(Example 16.1I), дисперсии и ковариации наблюдений (Example 16.1F), связь с
дисперсиями страт (Example 16.1H), eBLUP и остатки.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import Terms, _frame, _label, _marginal
from core.stats.factorial import _new_result, _settings
from core.stats.regression_checks import _transform_column
from core.stats.reml import blups, kenward_roger_df, refit_fixed, reml_fit, wald

_MAX_N = 2000  # плотные матрицы N × N в браузере
_FIXED = ("A", "B", "C")
_STRATA = ("s1", "s2", "s3")


# --- Матрицы плана ------------------------------------------------------------


def fixed_terms(names: list[str]) -> list[tuple[str, ...]]:
    """Главные эффекты, затем взаимодействия по возрастанию порядка (A, B, A.B …)."""
    return [t for size in range(1, len(names) + 1) for t in combinations(names, size)]


def _independent(design: np.ndarray) -> list[int]:
    """Линейно независимые столбцы слева направо (пустые ячейки дают совпадения)."""
    keep: list[int] = []
    for j in range(design.shape[1]):
        if np.linalg.matrix_rank(design[:, keep + [j]]) > len(keep):
            keep.append(j)
    return keep


def fixed_design(model: Terms, terms: list[tuple[str, ...]]) -> tuple[np.ndarray, dict]:
    """Матрица [1] + члены без вырожденных столбцов и индексы столбцов каждого члена."""
    blocks, owner = [np.ones((len(model.frame), 1))], [None]
    for t in terms:
        cols = model.columns(t)
        blocks.append(cols)
        owner += [t] * cols.shape[1]
    full = np.column_stack(blocks)
    keep = _independent(full)
    index = {t: [i for i, j in enumerate(keep) if owner[j] == t] for t in terms}
    return full[:, keep], {"index": index, "keep": keep, "owner": owner}


def random_design(frame: pd.DataFrame, strata: list[str]) -> list[np.ndarray]:
    """Индикаторы вложенных случайных членов: s1, s1.s2, s1.s2.s3 (стр. 428–429)."""
    zs, key = [], pd.Series([""] * len(frame), index=frame.index)
    for s in strata:
        key = key + "|" + frame[s].astype(str)
        levels = sorted(key.unique())
        zs.append(np.column_stack([(key == lev).to_numpy(float) for lev in levels]))
    return zs


# --- Тесты фиксированных членов -------------------------------------------------


def _test_row(fit: dict, cols: list[int], term: str) -> dict:
    """Вальд, P(χ²), F = W/q, df Кенварда–Роджера и P(F)."""
    q = len(cols)
    contrast = np.zeros((fit["x"].shape[1], q))
    contrast[cols, range(q)] = 1.0
    w = wald(fit, contrast)
    ddf = kenward_roger_df(fit, contrast)
    f = w / q
    return {"term": term, "df": q, "wald": w, "p_wald": float(stats.chi2.sf(w, q)),
            "F": f, "ddf": ddf, "p": float(stats.f.sf(f, q, ddf))}


def incremental_tests(model: Terms, terms: list[tuple[str, ...]], fit: dict, y: np.ndarray,
                      names: dict[str, str]) -> list[dict]:
    """Последовательное добавление членов при компонентах полной модели (Table 16.2)."""
    rows = []
    for k, term in enumerate(terms):
        x, info = fixed_design(model, terms[:k + 1])
        if not info["index"][term]:
            continue
        sub = refit_fixed(fit, y, x)
        rows.append(_test_row(sub, info["index"][term], "+ " + _label(term, names)))
    return rows


def marginal_tests(fit: dict, info: dict, terms: list[tuple[str, ...]],
                   names: dict[str, str]) -> list[dict]:
    """Исключение каждого немаргинального члена из полной модели (стр. 430–431)."""
    return [_test_row(fit, info["index"][t], "- " + _label(t, names))
            for t in terms if not _marginal(t, terms) and info["index"][t]]


# --- Компоненты дисперсии ------------------------------------------------------


def variance_table(fit: dict, labels: list[str]) -> list[dict]:
    """Оценки компонент и SE из ожидаемой информации (Table 16.3)."""
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(fit["info"]))))
    return [{"term": lab, "estimate": float(t), "se": float(s)}
            for lab, t, s in zip(labels, fit["theta"], se)]


def covariance_structure(fit: dict, labels: list[str]) -> list[dict]:
    """Ковариация двух наблюдений = сумма компонент общих случайных эффектов (Example 16.1F)."""
    theta = fit["theta"]
    rows = [{"shared": "—", "covariance": 0.0}]
    for k in range(len(theta) - 1):
        rows.append({"shared": labels[k], "covariance": float(theta[:k + 1].sum())})
    rows.append({"shared": "то же наблюдение (дисперсия)", "covariance": float(theta.sum())})
    return rows


def stratum_variances(frame: pd.DataFrame, strata: list[str], fit: dict) -> list[dict] | None:
    """ξ = σ² + Σ n_k σ²_k для сбалансированной вложенной структуры (стр. 435)."""
    sizes = []
    key = pd.Series([""] * len(frame), index=frame.index)
    for s in strata:
        key = key + "|" + frame[s].astype(str)
        counts = key.value_counts()
        if counts.nunique() != 1:
            return None
        sizes.append(int(counts.iloc[0]))
    theta = fit["theta"]
    rows = []
    for k in range(len(strata) + 1):
        xi = theta[-1] + sum(sizes[j] * theta[j] for j in range(k, len(strata)))
        rows.append({"stratum": ".".join(strata[:k + 1]) if k < len(strata) else "residual",
                     "variance": float(xi)})
    return rows


# --- Предсказания --------------------------------------------------------------


def _prediction_rows(model: Terms, terms: list[tuple[str, ...]], info: dict, fit: dict,
                     factors: list[str]) -> list[dict]:
    """Маргинальные предсказания всех комбинаций уровней с SE (Example 16.1I)."""
    rows = []
    for combo in product(*(model.levels[f] for f in factors)):
        one = pd.DataFrame({f: [lev] for f, lev in zip(factors, combo)})
        probe = Terms.__new__(Terms)
        probe.frame, probe.levels, probe.variates = one, model.levels, model.variates
        full = probe.design(terms)[0]
        row = full[info["keep"]]
        rows.append({"cell": " × ".join(combo), "row": row, "mean": float(row @ fit["beta"]),
                     "se": float(np.sqrt(row @ fit["phi"] @ row))})
    return rows


def factor_means(preds: list[dict], factors: list[str], levels: dict[str, list[str]],
                 fit: dict, names: dict[str, str]) -> list[dict]:
    """Средние уровней каждого фактора с равными весами по прочим и попарные SED (стр. 441)."""
    out = []
    for k, f in enumerate(factors):
        rows = []
        for lev in levels[f]:
            cells = [p for p in preds if p["cell"].split(" × ")[k] == lev]
            row = np.mean([c["row"] for c in cells], axis=0)
            rows.append({"level": lev, "row": row, "mean": float(row @ fit["beta"]),
                         "se": float(np.sqrt(row @ fit["phi"] @ row))})
        pairs = [{"pair": f"{a['level']} − {b['level']}", "difference": a["mean"] - b["mean"],
                  "sed": float(np.sqrt((a["row"] - b["row"]) @ fit["phi"] @ (a["row"] - b["row"])))}
                 for a, b in combinations(rows, 2)]
        out.append({"factor": names.get(f, f), "means": [{k2: v for k2, v in r.items() if k2 != "row"}
                                                       for r in rows], "differences": pairs})
    return out


def _sed_rows(preds: list[dict], fit: dict) -> list[dict]:
    """SED всех пар предсказаний (структурные вклады сокращаются, стр. 437)."""
    out = []
    for a, b in combinations(preds, 2):
        d = a["row"] - b["row"]
        out.append({"pair": f"{a['cell']} − {b['cell']}", "difference": a["mean"] - b["mean"],
                    "sed": float(np.sqrt(d @ fit["phi"] @ d))})
    return out


# --- Подготовка данных ----------------------------------------------------------


def _collect(values: list[float], factor_a: list[str], factor_b: list[str] | None,
             factor_c: list[str] | None, strata: list[list[str] | None]) -> tuple[pd.DataFrame, int]:
    """Полные строки: отклик, фиксированные факторы, структурные факторы."""
    factors = {"A": factor_a, "B": factor_b, "C": factor_c}
    factors.update({s: v for s, v in zip(_STRATA, strata)})
    return _frame(values, {}, factors)


def _names(column_names: dict[str, str] | None, used: list[str]) -> dict[str, str]:
    """Имена колонок пользователя для членов модели."""
    cn = column_names or {}
    slot = {"A": "factor_a", "B": "factor_b", "C": "factor_c", "s1": "stratum_1",
            "s2": "stratum_2", "s3": "stratum_3"}
    return {k: cn.get(slot[k], k) for k in used}


def _guard(frame: pd.DataFrame, fixed: list[str], strata: list[str], result: StatResult) -> bool:
    """Объём, уровни факторов и случайная структура."""
    if len(frame) > _MAX_N:
        result.add_warning("mixed_too_large", f"Больше {_MAX_N} наблюдений", Severity.ERROR)
        return False
    if any(frame[f].nunique() < 2 for f in fixed):
        result.add_warning("factor_one_level", "У фиксированного фактора один уровень",
                           Severity.ERROR)
        return False
    if not strata or frame[strata[0]].nunique() < 2:
        result.add_warning("mixed_no_structure", "Нужен структурный фактор минимум с 2 уровнями",
                           Severity.ERROR)
        return False
    return True


# --- Метод каталога -------------------------------------------------------------


def mixed_model(
    values: list[float],
    factor_a: list[str],
    stratum_1: list[str],
    factor_b: list[str] | None = None,
    factor_c: list[str] | None = None,
    stratum_2: list[str] | None = None,
    stratum_3: list[str] | None = None,
    alpha: float = 0.05,
    transform_kind: str = "none",
    offset: float = 0.0,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Смешанная модель: фиксированные факторы и вложенная случайная структура (гл. 16)."""
    settings = _settings(alpha, transform_kind, offset)
    frame, dropped = _collect(values, factor_a, factor_b, factor_c,
                              [stratum_1, stratum_2, stratum_3])
    fixed = [f for f, v in zip(_FIXED, (factor_a, factor_b, factor_c)) if v is not None]
    strata = [s for s, v in zip(_STRATA, (stratum_1, stratum_2, stratum_3)) if v is not None]
    names = _names(column_names, fixed + strata)
    result = _new_result("mixed_model", frame, settings, column_names=names)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    if not _transform_column(frame, "y", transform_kind, offset, result) or not _guard(
            frame, fixed, strata, result):
        return result
    _fit_and_fill(frame, fixed, strata, names, result)
    return result


def _fit_and_fill(frame: pd.DataFrame, fixed: list[str], strata: list[str],
                  names: dict[str, str], result: StatResult) -> None:
    """REML, тесты, компоненты, предсказания и остатки."""
    y = frame["y"].to_numpy(float)
    model = Terms(frame, fixed, [])
    terms = fixed_terms(fixed)
    x, info = fixed_design(model, terms)
    zs = random_design(frame, strata)
    fit = reml_fit(y, x, [z @ z.T for z in zs] + [np.eye(len(y))])
    if not fit["converged"]:
        result.add_warning("reml_not_converged", "REML не сошёлся: проверь структуру "
                           "(тот же член в фиксированной и случайной модели, 0 df страты).",
                           Severity.ERROR)
        return
    labels = [".".join(names[s] for s in strata[:k + 1]) for k in range(len(strata))]
    _fill_random(frame, strata, fit, labels + ["Residual"], result)
    _fill_fixed(model, terms, x, info, fit, y, names, result)
    _fill_residuals(fit, y, zs, result)


def _fill_random(frame: pd.DataFrame, strata: list[str], fit: dict, labels: list[str],
                 result: StatResult) -> None:
    """Компоненты, ковариации наблюдений и дисперсии страт."""
    table = variance_table(fit, labels)
    result.parameters.update({"variance_components": table,
                              "covariance_structure": covariance_structure(fit, labels),
                              "stratum_variances": stratum_variances(frame, strata, fit)})
    result.statistics.update({f"sigma2_{r['term']}": r["estimate"] for r in table})
    result.statistics["total_variance"] = float(fit["theta"].sum())
    negative = [r["term"] for r in table[:-1] if r["estimate"] < 0]
    if negative:
        result.add_warning("negative_variance_component", "Отрицательная компонента: "
                           f"{', '.join(negative)} — единицы внутри меньше похожи, чем между.",
                           Severity.INFO, params={"terms": ", ".join(negative)})


def _fill_fixed(model: Terms, terms: list[tuple[str, ...]], x: np.ndarray, info: dict, fit: dict,
                y: np.ndarray, names: dict[str, str], result: StatResult) -> None:
    """Фиксированные эффекты, тесты, предсказания и SED."""
    se = np.sqrt(np.diag(fit["phi"]))
    labels = ["[1]"] + [_column_label(model, info["owner"][j], j, info, names)
                        for j in info["keep"][1:]]
    effects = [{"term": lab, "estimate": float(b), "se": float(s)}
               for lab, b, s in zip(labels, fit["beta"], se)]
    preds = _prediction_rows(model, terms, info, fit, list(model.levels))
    seds = _sed_rows(preds, fit)
    result.parameters.update({
        "fixed_effects": effects, "incremental": incremental_tests(model, terms, fit, y, names),
        "marginal": marginal_tests(fit, info, terms, names),
        "predictions": [{k: v for k, v in p.items() if k != "row"} for p in preds],
        "sed": seds, "factor_means": factor_means(preds, list(model.levels), model.levels, fit,
                                                  names)})
    result.statistics.update({"sed_min": min(s["sed"] for s in seds),
                              "sed_max": max(s["sed"] for s in seds)})
    for e in effects:
        result.estimates.append(Estimate(e["term"], e["estimate"], std_error=e["se"]))


def _column_label(model: Terms, term: tuple[str, ...], j: int, info: dict,
                  names: dict[str, str]) -> str:
    """Подпись столбца first-level-zero: «Species Am», «Irrigation yes.Species Ga»."""
    parts = [[names.get(n, n)] if n in model.variates else
             [f"{names.get(n, n)} {lev}" for lev in model.levels[n][1:]] for n in term]
    combos = [".".join(c) for c in product(*parts)]
    first = info["owner"].index(term)
    return combos[j - first]


def _fill_residuals(fit: dict, y: np.ndarray, zs: list[np.ndarray], result: StatResult) -> None:
    """Остатки = eBLUP отклонений; подогнанные — без случайных эффектов (стр. 437)."""
    us = blups(fit, y, zs)
    fitted = fit["x"] @ fit["beta"]
    resid = y - fitted - sum(z @ u for z, u in zip(zs, us))
    blom = stats.norm.ppf((np.arange(1, y.size + 1) - 0.375) / (y.size + 0.25))
    result.parameters["plots"] = {
        "residual_vs_fitted": {"fitted": fitted.tolist(), "residuals": resid.tolist()},
        "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(resid).tolist()},
        "residuals": resid.tolist()}
    result.parameters["blups"] = [u.tolist() for u in us]
