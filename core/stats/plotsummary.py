"""Данные графиков-итогов: средние с ДИ, взаимодействие, ДИ разностей.

Пакет 1 плана графиков (claude/plots-plan.md). Всё берётся из уже
посчитанного результата метода (``StatResult.to_dict()``), без пересчёта:
мост вызывает :func:`summary_plots` после основного расчёта.

Соглашения ядра: разность пары — ``group2 − group1``; при трансформации
отклика таблицы средних несут ``back_*`` — значения на исходной шкале
(ДИ асимметричен, гл. 6).
"""

from __future__ import annotations

import math
import re
from typing import Any

MAX_PAIRS = 30   # больше пар — forest нечитаем, остаётся таблица
MAX_LEVELS = 20  # уровней на оси средних
MAX_LINES = 10   # линий на графике взаимодействия

_CELL = re.compile(r"^\s*([^=]+?)=(.*)$")


def _ok(*values: Any) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _transformed(params: dict[str, Any]) -> bool:
    return params.get("transform") not in (None, "", "none")


def _mean_item(row: dict, label: str, back: bool) -> dict | None:
    """Строка таблицы средних → точка с ДИ (исходная шкала, если есть)."""
    if row.get("estimable") is False:
        return None
    keys = ("back_mean", "back_ci_lower", "back_ci_upper") if back else \
        ("mean", "ci_lower", "ci_upper")
    mean, lo, hi = (row.get(k) for k in keys)
    if not _ok(mean, lo, hi):
        return None
    return {"label": label, "mean": mean, "lower": lo, "upper": hi}


def means_from_rows(rows: list[dict], back: bool, label_key: str = "label",
                    interval: str = "ci") -> dict | None:
    """Средние с ДИ для оси категорий; None — нечего или слишком много."""
    items = [it for r in rows if (it := _mean_item(r, str(r.get(label_key)), back))]
    if not items or len(items) > MAX_LEVELS:
        return None
    return {"items": items, "interval": interval, "numeric": False}


def _cell(label: str) -> list[tuple[str, str]] | None:
    """«A=a × B=b» → [(A, a), (B, b)]."""
    parts = [_CELL.match(p) for p in label.split("×")]
    if not all(parts):
        return None
    return [(m.group(1).strip(), m.group(2).strip()) for m in parts]


def interaction_from_rows(rows: list[dict], back: bool) -> dict | None:
    """График взаимодействия двух факторов по таблице средних ячеек."""
    cells = [(_cell(str(r.get("label", ""))), _mean_item(r, "", back)) for r in rows]
    if not cells or any(c is None or len(c) != 2 for c, _ in cells):
        return None
    (fa, _), (fb, _) = cells[0][0]
    levels_a = list(dict.fromkeys(c[0][1] for c, _ in cells))
    levels_b = list(dict.fromkeys(c[1][1] for c, _ in cells))
    if len(levels_b) > MAX_LINES or len(levels_a) > MAX_LEVELS:
        return None
    grid = {(c[0][1], c[1][1]): it for c, it in cells}
    series = []
    for b in levels_b:
        pts = [grid.get((a, b)) for a in levels_a]
        series.append({"name": b, "values": [p["mean"] if p else None for p in pts],
                       "lower": [p["lower"] if p else None for p in pts],
                       "upper": [p["upper"] if p else None for p in pts]})
    return {"factor_a": fa, "factor_b": fb, "levels_a": levels_a, "series": series}


def differences_from(comparisons: list[dict]) -> dict | None:
    """ДИ разностей пар group2 − group1 (forest); опорная линия 0."""
    items = []
    for c in comparisons:
        diff = c.get("mean_diff")
        lo, hi = c.get("ci_lower"), c.get("ci_upper")
        if not _ok(lo, hi) and _ok(diff, c.get("half_width")):
            lo, hi = diff - c["half_width"], diff + c["half_width"]
        if _ok(diff, lo, hi):
            items.append({"label": f"{c['group2']} − {c['group1']}", "diff": diff,
                          "lower": lo, "upper": hi})
    if not items or len(items) > MAX_PAIRS:
        return None
    return {"items": items, "reference": 0.0, "kind": "diff"}


def _ratios(ratios: list[dict]) -> dict | None:
    """Отношения средних на исходной шкале после log (гл. 6); опорная 1."""
    items = [{"label": f"{r['group2']} / {r['group1']}", "diff": r["ratio"],
              "lower": r["ci_lower"], "upper": r["ci_upper"]}
             for r in ratios if _ok(r.get("ratio"), r.get("ci_lower"), r.get("ci_upper"))]
    if not items or len(items) > MAX_PAIRS:
        return None
    return {"items": items, "reference": 1.0, "kind": "ratio"}


def _equivalence(params: dict) -> dict | None:
    """TOST: разности с контролем, ДИ и границы ±c (гл. 10.5)."""
    margin = params.get("margin")
    if params.get("margin_kind") == "percent" and _ok(margin, params.get("control_mean")):
        margin = margin / 100 * params["control_mean"]
    control = params.get("control", "")
    items = [{"label": f"{r['level']} − {control}", "diff": r["difference"],
              "lower": r["ci_lower"], "upper": r["ci_upper"]}
             for r in params.get("equivalence", [])
             if _ok(r.get("difference"), r.get("ci_lower"), r.get("ci_upper"))]
    if not items or len(items) > MAX_PAIRS or not _ok(margin):
        return None
    return {"items": items, "reference": 0.0, "kind": "diff", "margin": abs(margin)}


def _numeric_levels(means: dict) -> dict:
    """Уровни-дозы (полиномиальный контраст): ось x числовая."""
    try:
        for it in means["items"]:
            it["x"] = float(it["label"].replace(",", "."))
    except ValueError:
        return means
    return {**means, "numeric": True}


def _transformed_means(result: dict) -> dict | None:
    """Средние на исходной шкале с асимметричным ДИ: оценки mean_i + labels."""
    labels = result["parameters"].get("labels", [])
    est = {e["name"]: e for e in result.get("estimates", [])}
    rows = [{"label": lab, **{k: est[f"mean_{i}"].get(v) for k, v in
                              (("mean", "value"), ("ci_lower", "ci_lower"),
                               ("ci_upper", "ci_upper"))}}
            for i, lab in enumerate(labels, 1) if f"mean_{i}" in est]
    return means_from_rows(rows, back=False)


def _adjusted_means(params: dict) -> dict | None:
    """Скорректированные средние ANCOVA: ± SE (df в результате нет)."""
    rows = [{"label": r["group"], "mean": r["mean"],
             "ci_lower": r["mean"] - r["se"], "ci_upper": r["mean"] + r["se"]}
            for r in params.get("adjusted_means", []) if _ok(r.get("mean"), r.get("se"))]
    return means_from_rows(rows, back=False, interval="se")


def _model_tables(params: dict) -> dict[str, Any]:
    """Таблицы средних модели: взаимодействие (2 фактора) и средние с ДИ."""
    out: dict[str, Any] = {}
    back = _transformed(params)
    for table in params.get("tables", []):
        rows = table.get("rows", [])
        if "interaction" not in out and (inter := interaction_from_rows(rows, back)):
            out["interaction"] = inter
            continue  # те же ячейки средними с ДИ не дублируем
        if "means_ci" not in out and (means := means_from_rows(rows, back)):
            out["means_ci"] = {**means, "term": table.get("term", "")}
    return out


_COMPARISONS = {"tukey_hsd", "scheffe_test", "treatment_means", "rcbd_anova",
                "rcbd_subsampling_anova", "bibd_anova", "mean_comparisons", "rm_anova",
                "simple_effects"}
_TABLES = {"factorial_anova", "latin_square_anova", "nested_anova", "unbalanced_anova",
           "split_plot_anova"}


# --- Пакет 2: кривые по группам и графики против x -------------------------------------

def _pairs(xs: list, ys: list) -> list[list[float]]:
    return [[x, y] for x, y in zip(xs, ys) if _ok(x, y)]


def _curve(name: str, cv: dict) -> dict:
    """Кривая группы на сетке x; полоса ДИ — если есть и конечна везде."""
    out = {"name": name, "line": _pairs(cv["x"], cv["fitted"])}
    lo, hi = cv.get("lower"), cv.get("upper")
    if lo and hi and all(_ok(v) for v in list(lo) + list(hi)):
        out["band"] = {"lower": _pairs(cv["x"], lo), "upper": _pairs(cv["x"], hi)}
    return out


def _block_curves(params: dict) -> dict | None:
    """Кривые всех групп (curve_in_blocks, nonlinear_in_blocks), гл. 17 + 15.3."""
    curves = params.get("curves", [])
    pts = params.get("points")
    if len(curves) < 2 or not pts or len(curves) > MAX_LINES:
        return None
    return {"points": [[x, y, None] for x, y in _pairs(pts["x"], pts["y"])],
            "curves": [_curve(str(c.get("group")), c) for c in curves]}


def _group_lines(params: dict) -> dict | None:
    """Данные и линии итоговой модели регрессии с группами (гл. 15)."""
    gl = params.get("group_lines")
    if not gl or len(gl["lines"]) > MAX_LINES:
        return None
    points = [[x, y, g] for x, y, g in zip(gl["x"], gl["y"], gl["group"]) if _ok(x, y)]
    return {"points": points,
            "curves": [{"name": ln["group"], "line": _pairs(ln["x"], ln["fitted"])}
                       for ln in gl["lines"]]}


_CRITERIA = {"adj_r_squared": "R²adj", "aic": "AIC", "sbc": "SBC"}


def _mark_best_models(curves: dict, best: dict) -> dict:
    """Сравнение кривых (Table 17.12): к имени лучшей модели — критерии, по
    которым она лучшая; заголовок — «модели», а не «группы»."""
    renamed = []
    for cv in curves["curves"]:
        wins = [label for key, label in _CRITERIA.items() if best.get(key) == cv["name"]]
        renamed.append({**cv, "name": f"{cv['name']} ({', '.join(wins)})" if wins else cv["name"]})
    return {**curves, "curves": renamed, "title": "models"}


def _extra_scatters(params: dict) -> list[dict]:
    """Графики против x и индексные: остатки, |остатки|, расхождения CV, рычаги, Кук."""
    plots, thr = params.get("plots") or {}, params.get("thresholds") or {}
    names = params.get("variables") or ["x1", "x2"]
    out = []

    def add(kind: str, xs: list, ys: list, ref: list[float], x_name: str = "") -> None:
        if points := _pairs(xs, ys):
            out.append({"kind": kind, "points": points, "ref": ref, "x_name": x_name})

    if rx := plots.get("residual_vs_x"):
        add("residual_vs_x", rx["x"], rx["residuals"], [0.0])
    # Поверхность отклика: x — список значений, остатки — std_residuals (порядок строк).
    for i, key in enumerate(("residual_vs_x1", "residual_vs_x2")):
        if isinstance(xs := plots.get(key), list) and plots.get("std_residuals"):
            add("residual_vs_x", xs, plots["std_residuals"], [0.0],
                str(names[i]) if i < len(names) else "")
    if ov := plots.get("observed_vs_fitted"):
        add("observed_vs_fitted", ov["fitted"], ov["observed"], [])
    if dv := plots.get("discrepancy_vs_x"):
        add("discrepancy_vs_x", dv["x"], dv["discrepancy"], [0.0])
    if ar := plots.get("abs_residual_vs_fitted"):
        add("abs_residual_vs_fitted", ar["fitted"], ar["abs_residuals"], [])
    if lv := plots.get("leverage_vs_x"):
        add("leverage_vs_x", lv["x"], lv["leverage"], [thr["leverage"]] if _ok(thr.get("leverage")) else [])
    if ck := plots.get("cook_vs_fitted"):
        ref = [thr["modified_cook"]] if _ok(thr.get("modified_cook")) else []
        add("cook_vs_fitted", ck["fitted"], ck["cook"], ref)
    return out


# --- Пакет 3: GLM, порядковые и категориальные данные, парные измерения --------------

MAX_PAIR_LINES = 60  # больше объектов — линии до/после сливаются, остаётся гистограмма


def _glm_predictions(params: dict) -> dict | None:
    """Предсказания по уровням первого фактора на натуральной шкале (гл. 18):
    обратное связующее преобразование, ДИ асимметричен."""
    preds = params.get("factor_predictions") or []
    if not preds:
        return None
    factor = preds[0]["factor"]
    rows = [{"label": p["level"], "mean": p.get("natural"), "ci_lower": p.get("natural_lower"),
             "ci_upper": p.get("natural_upper")} for p in preds if p["factor"] == factor]
    means = means_from_rows(rows, back=False)
    if means is None:
        # ДИ не посчитаны (вырожденная ковариация, напр. ZINB) — только точки.
        rows = [{**r, "ci_lower": r["mean"], "ci_upper": r["mean"]} for r in rows]
        means = means_from_rows(rows, back=False, interval="none")
    return {**means, "term": factor, "title": "predictions"} if means else None


def _category_probabilities(params: dict) -> dict | None:
    """Вероятности категорий порядкового отклика по уровням (составные столбцы)."""
    cats = params.get("categories") or []
    probs = params.get("category_probabilities") or []
    if not cats or not probs or len(probs) > MAX_LEVELS:
        return None
    several = len({p["factor"] for p in probs}) > 1
    rows = [{"label": f"{p['factor']}={p['level']}" if several else str(p["level"]),
             "values": [p["probabilities"].get(c) for c in cats]} for p in probs]
    if not all(_ok(*r["values"]) for r in rows):
        return None
    return {"categories": [str(c) for c in cats], "rows": rows}


def _contingency(params: dict) -> dict | None:
    """Теплокарта таблицы сопряжённости: цвет — скорректированный остаток,
    подпись — наблюдённая частота (|остаток| > 2 — ячейка выделяется), гл. 18.6."""
    table, cells = params.get("table") or {}, params.get("cells") or []
    rows, cols = table.get("rows") or [], table.get("columns") or []
    if not cells or len(rows) > MAX_LEVELS or len(cols) > MAX_LEVELS:
        return None
    ri = {str(r): i for i, r in enumerate(rows)}
    ci = {str(c): i for i, c in enumerate(cols)}
    data = [[ci[str(c["column"])], ri[str(c["row"])], c["adjusted_residual"], c["observed"],
             c["expected"]] for c in cells
            if _ok(c.get("adjusted_residual"), c.get("observed"), c.get("expected"))]
    return {"rows": [str(r) for r in rows], "columns": [str(c) for c in cols],
            "cells": data} if data else None


def paired_plot(before: list[float], after: list[float], names: tuple[str, str]) -> dict:
    """До/после по объектам и гистограмма разностей до − после (как в ядре)."""
    from core.stats.plotdata import histogram_data

    pairs = [[b, a] for b, a in zip(before, after) if _ok(b, a)]
    diffs = [b - a for b, a in pairs]
    return {"labels": list(names), "pairs": pairs if len(pairs) <= MAX_PAIR_LINES else None,
            "diff_hist": histogram_data(diffs)}


# --- Пакет 4: планирование и отбор моделей, поверхность отклика ---------------------------

MAX_SUBSETS = 500  # точек на графике Cp / R²adj


def _power_curve(params: dict) -> dict | None:
    """Мощность против повторности n и целевая мощность (гл. 10)."""
    rows = params.get("replication_table") or []
    points = _pairs([r.get("n") for r in rows], [r.get("power") for r in rows])
    if len(points) < 2:
        return None
    target = params.get("target_power")
    return {"kind": "power_vs_n", "points": points, "ref": [target] if _ok(target) else [],
            "x_name": "", "line": True}


def _subset_scatters(params: dict) -> list[dict]:
    """Все подмножества: Cp (с линией Cp = p) и R²adj против числа параметров p."""
    subsets = (params.get("subsets") or [])[:MAX_SUBSETS]
    labels = [", ".join(s.get("terms", [])) for s in subsets]
    p = [s.get("p") for s in subsets]
    out = []
    for kind, key, diagonal in (("cp_vs_p", "cp", True), ("adj_r2_vs_p", "adj_r_squared", False)):
        ys = [s.get(key) for s in subsets]
        keep = [i for i in range(len(subsets)) if _ok(p[i], ys[i])]
        if keep:
            out.append({"kind": kind, "points": [[p[i], ys[i]] for i in keep], "ref": [],
                        "x_name": "", "diagonal": diagonal, "labels": [labels[i] for i in keep]})
    return out


def _cv_models(params: dict) -> dict | None:
    """MSEP моделей при кросс-валидации (гл. 14.4); лучшая — отмечена."""
    models = params.get("models") or []
    best = ", ".join(params.get("best") or [])
    items = [{"label": ", ".join(m.get("terms", [])) or "—", "value": m["msep"]}
             for m in models if _ok(m.get("msep"))]
    if not items or len(items) > MAX_PAIRS:
        return None
    return {"kind": "msep", "items": items, "best": best}


def _surface(params: dict) -> dict | None:
    """Поверхность отклика на сетке x1 × x2 (гл. 16.3); прочие x — на средних."""
    sf = params.get("surface") or {}
    x1, x2, z = sf.get("x1") or [], sf.get("x2") or [], sf.get("fitted") or []
    if not x1 or not x2 or len(z) != len(x1) or not all(len(row) == len(x2) for row in z):
        return None
    cells = [[i, j, z[i][j]] for i in range(len(x1)) for j in range(len(x2)) if _ok(z[i][j])]
    names = params.get("variables") or ["x1", "x2"]
    return {"x1": x1, "x2": x2, "cells": cells, "names": [str(n) for n in names[:2]]}


def _comparison_plots(method: str, result: dict) -> dict[str, Any]:
    """Пакет 1: средние, взаимодействие, разности."""
    params = result.get("parameters", {})
    out: dict[str, Any] = {}
    if method in _COMPARISONS and (diff := differences_from(params.get("comparisons", []))):
        out["differences"] = diff
    if method in _TABLES:
        out.update(_model_tables(params))
    if method in ("mean_comparisons", "polynomial_contrasts"):
        means = means_from_rows(params.get("means", []), back=False, label_key="level")
        if means:
            out["means_ci"] = _numeric_levels(means) if method == "polynomial_contrasts" else means
    if method == "transformed_means":
        if means := _transformed_means(result):
            out["means_ci"] = means
        if ratios := _ratios(params.get("ratios", [])):
            out["differences"] = ratios
    if method == "ancova" and (means := _adjusted_means(params)):
        out["means_ci"] = means
    if method == "equivalence_test" and (eq := _equivalence(params)):
        out["differences"] = eq
    if method == "simple_effects" and (inter := interaction_from_rows(
            params.get("cell_means", []), back=False)):
        out["interaction"] = inter
    if method == "chi2_goodness_of_fit" and (shares := _gof_shares(params)):
        out["means_ci"] = shares
    return out


def _gof_shares(params: dict) -> dict | None:
    """Наблюдаемые доли категорий с ДИ Уилсона (χ² согласия, Witte гл. 19)."""
    rows = [{"label": c["category"], "mean": c.get("p_observed"), "ci_lower": c.get("ci_lower"),
             "ci_upper": c.get("ci_upper")} for c in params.get("cells", [])]
    return means_from_rows(rows, back=False)


def _model_plots(method: str, result: dict) -> dict[str, Any]:
    """Пакеты 2–4: кривые, графики против x, GLM, категории, отбор, поверхность."""
    params = result.get("parameters", {})
    out: dict[str, Any] = {}
    if curves := _block_curves(params) or _group_lines(params):
        if method == "curve_comparison":
            curves = _mark_best_models(curves, params.get("best") or {})
        out.update({"group_curves": curves, "scatter": None})
    extra = _extra_scatters(params) + _subset_scatters(params)
    if method == "replication_power" and (power := _power_curve(params)):
        extra.append(power)
    if extra:
        out["extra_scatter"] = extra
    if method in ("glm_analysis", "glmm_analysis", "count_regression") and \
            (pred := _glm_predictions(params)):
        out["means_ci"] = pred
    if method == "ordinal_regression" and (stack := _category_probabilities(params)):
        out["stacked"] = stack
    if method == "contingency_test" and (heat := _contingency(params)):
        out["heatmap"] = heat
    if method == "model_cross_validation" and (bars := _cv_models(params)):
        out["bars"] = bars
    if method == "response_surface" and (surface := _surface(params)):
        out["surface"] = surface
    return out


def summary_plots(method: str, result: dict) -> dict[str, Any]:
    """Графики-итоги метода из готового результата. Ключ со значением None —
    убрать график моста (кривые по группам заменяют одиночную кривую)."""
    return {**_model_plots(method, result), **_comparison_plots(method, result)}
