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


_COMPARISONS = {"tukey_hsd", "treatment_means", "rcbd_anova", "rcbd_subsampling_anova",
                "bibd_anova", "mean_comparisons"}
_TABLES = {"factorial_anova", "latin_square_anova", "nested_anova", "unbalanced_anova",
           "split_plot_anova"}


def summary_plots(method: str, result: dict) -> dict[str, Any]:
    """Графики-итоги метода: ключи means_ci, interaction, differences."""
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
    return out
