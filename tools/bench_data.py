"""Синтетические данные и случаи замера методов (общие для CLI и браузера).

Используют: `tools/bench_methods.py` (CPython, через шаблоны моста) и
страница `web/bench.html` (Pyodide, через сам мост). Только numpy — чтобы
модуль грузился и в Pyodide. Размер — число строк основной таблицы
(округляется вверх до кратного 12: так сбалансированы все схемы сразу).
"""

from __future__ import annotations

import json

import numpy as np

# ── Синтетические данные ─────────────────────────────────────────────────


def main_table(n: int, seed: int = 1) -> dict[str, list]:
    """Таблица на n строк (кратно 12): все сбалансированные схемы сразу.

    RCBD (trt × blk), факториал A×B×C в блоках по 12, субвыборки, split-plot,
    вложенная, повторные измерения, регрессии, счёты, доли, категории.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    trt, a, b, c = i % 4, i % 2, (i // 2) % 3, (i // 6) % 2
    farm, site = i // 20, i % 6
    x = rng.uniform(0.5, 10, n)
    farm_eff = rng.normal(0, 1, farm.max() + 1)[farm]
    y = 50 + 2 * trt + 1.5 * a + b + rng.normal(0, 1, n // 4 + 1)[i // 4] + rng.normal(0, 2, n)
    p = 1 / (1 + np.exp(-(-0.5 + 0.15 * x + 0.3 * a + 0.5 * farm_eff)))
    return {
        "y": y.round(3), "trt": [f"t{v}" for v in trt], "blk": [f"b{v}" for v in i // 4],
        "g2": [f"g{v}" for v in a], "A": [f"a{v}" for v in a], "B": [f"b{v}" for v in b],
        "C": [f"c{v}" for v in c], "fblk": [f"f{v}" for v in i // 12],
        "trt_s": [f"t{v}" for v in (i // 2) % 4], "unit": [f"u{v}" for v in i // 2],
        "blk_s": [f"s{v}" for v in i // 8], "sA": [f"w{v}" for v in (i // 3) % 2],
        "sB": [f"p{v}" for v in i % 3], "sblk": [f"r{v}" for v in i // 6],
        "wplot": [f"wp{v}" for v in i // 3], "ngrp": [f"n{v}" for v in i % 3],
        "nmem": [f"m{v % 3}_{(v // 3) % 4}" for v in i], "subj": [f"s{v}" for v in i // 3],
        "cond": [f"c{v}" for v in i % 3], "farm": [f"farm{v}" for v in farm],
        "site": [f"site{v}" for v in site], **_numeric_columns(rng, n, x, a, p),
    }


def _numeric_columns(rng: np.random.Generator, n: int, x: np.ndarray, a: np.ndarray,
                     p: np.ndarray) -> dict[str, list]:
    """Регрессии, парные, счёты, доли, порядковый балл, категории."""
    x2, x3 = rng.uniform(0, 5, n), rng.normal(10, 2, n)
    month = np.arange(n) % 12 + 1
    pre = rng.normal(20, 3, n)
    score = np.digitize(0.3 * x + rng.logistic(size=n), [1, 2.5, 4])
    return {
        "x": x.round(4), "x2": x2.round(4), "x3": x3.round(4),
        "ylin": (2 + 0.8 * x + 0.3 * x2 - 0.2 * x3 + rng.normal(0, 1, n)).round(4),
        "ycurve": (5 + 10 * np.exp(-0.35 * x) + rng.normal(0, 0.4, n)).round(4),
        "month": month, "ytrig": (10 + 3 * np.cos(2 * np.pi * month / 12)
                                  + rng.normal(0, 0.5, n)).round(4),
        "pre": pre.round(3), "post": (pre + 0.8 + rng.normal(0, 1, n)).round(3),
        "count": rng.poisson(np.exp(1 + 0.3 * a)), "k": rng.binomial(20, p), "m": 20,
        "score": np.array(["none", "low", "mid", "high"])[score],
        "cat1": rng.choice(["red", "green", "blue"], n, p=[0.5, 0.3, 0.2]),
        "yn": rng.choice(["yes", "no"], n), "yn2": rng.choice(["yes", "no"], n),
    }


def latin_table(t: int = 6, seed: int = 2) -> dict[str, list]:
    """Латинский квадрат t × t (размер задан схемой, не числом строк)."""
    rng = np.random.default_rng(seed)
    r, c = np.divmod(np.arange(t * t), t)
    trt = (r + c) % t
    return {"y": (20 + trt + rng.normal(0, 1, t * t)).round(3), "trt": [f"t{v}" for v in trt],
            "row": [f"r{v}" for v in r], "col": [f"c{v}" for v in c]}


def bibd_table(n: int, seed: int = 3) -> dict[str, list]:
    """BIBD t=7, k=3, b=7 (плоскость Фано), повторённый до ~n строк."""
    fano = [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)]
    reps = max(1, round(n / 21))
    rng = np.random.default_rng(seed)
    blk, trt = zip(*[(f"b{r}_{j}", f"t{v}") for r in range(reps)
                     for j, block in enumerate(fano) for v in block])
    y = 30 + np.array([int(t[1:]) for t in trt]) + rng.normal(0, 1, len(trt))
    return {"y": y.round(3), "trt": list(trt), "blk": list(blk)}


def to_csv(cols: dict[str, list]) -> bytes:
    names = list(cols)
    rows = zip(*(np.broadcast_to(np.asarray(cols[k], dtype=object), (len(cols["y"]),))
                 for k in names))
    return (",".join(names) + "\n" + "\n".join(",".join(map(str, r)) for r in rows)
            + "\n").encode()


# ── Случаи: метод → таблица, колонки, параметры ──────────────────────────

G = {"value": "y", "group": "trt"}
XY = {"x": "x", "y": "ylin"}
CURVE = {"x": "x", "y": "ycurve"}
RCBD = {"value": "y", "group": "trt", "block": "blk"}
FAC = {"value": "y", "factor_a": "A", "factor_b": "B"}
RM = {"value": "y", "condition": "cond", "subject": "subj"}
MULTI = {"variables": "x, x2, x3"}

CASES: dict[str, tuple[str, dict, dict]] = {
    "describe_all": ("main", {}, {}),
    "describe_by_group": ("main", G, {}),
    "chi2_goodness_of_fit": ("main", {"response": "cat1"}, {}),
    "frequency_table": ("main", {"response": "x"}, {}),
    "mean_ci": ("main", {"value": "y"}, {}),
    "normality_test": ("main", {"value": "y"}, {}),
    "one_proportion": ("main", {"response": "yn"}, {"success": "yes"}),
    "two_proportions": ("main", {"response": "yn", "group": "g2"}, {"success": "yes"}),
    "mcnemar_test": ("main", {"first": "yn", "second": "yn2"}, {"success": "yes"}),
    "one_sample_t": ("main", {"value": "y"}, {"popmean": 53}),
    "one_sample_wilcoxon": ("main", {"value": "y"}, {"popmean": 53}),
    "two_sample_t": ("main", {"value": "y", "group": "g2"}, {}),
    "mann_whitney": ("main", {"value": "y", "group": "g2"}, {}),
    "one_way_anova": ("main", G, {}),
    "rm_anova": ("main", RM, {}),
    "friedman": ("main", RM, {}),
    "welch_anova": ("main", G, {}),
    "tukey_hsd": ("main", G, {}),
    "scheffe_test": ("main", G, {}),
    "kruskal_wallis": ("main", G, {}),
    "treatment_means": ("main", G, {}),
    "anova_permutation_test": ("main", G, {}),
    "bartlett_test": ("main", G, {}),
    "anova_residuals": ("main", G, {}),
    "transformed_means": ("main", G, {}),
    "rcbd_anova": ("main", RCBD, {}),
    "rcbd_subsampling_anova": ("main", {"value": "y", "group": "trt_s", "block": "blk_s",
                                        "unit": "unit"}, {}),
    "factorial_anova": ("main", {**FAC, "factor_c": "C", "block": "fblk"}, {}),
    "simple_effects": ("main", FAC, {}),
    "nested_anova": ("main", {"value": "y", "group": "ngrp", "member": "nmem"}, {}),
    "treatment_contrasts": ("main", RCBD, {"contrasts": "t0 vs t1, t2, t3"}),
    "polynomial_contrasts": ("main", {"value": "y", "group": "month"}, {}),
    "mean_comparisons": ("main", RCBD, {}),
    "latin_square_anova": ("latin", {"value": "y", "factor_a": "trt", "row": "row",
                                     "column": "col"}, {}),
    "split_plot_anova": ("main", {"value": "y", "factor_a": "sA", "factor_b": "sB",
                                  "block": "sblk", "whole_plot": "wplot"}, {}),
    "bibd_anova": ("bibd", {"value": "y", "group": "trt", "block": "blk"}, {}),
    "replication_power": ("main", RCBD, {"delta": 2}),
    "equivalence_test": ("main", G, {"control": "t0", "margin": 3}),
    "unbalanced_anova": ("main", FAC, {}),
    "paired_t": ("main", {"before": "pre", "after": "post"}, {}),
    "wilcoxon": ("main", {"before": "pre", "after": "post"}, {}),
    "simple_linear_regression": ("main", XY, {}),
    "regression_details": ("main", XY, {"lack_of_fit": "no"}),
    "inverse_prediction": ("main", XY, {"y_new": "8"}),
    "regression_diagnostics": ("main", XY, {}),
    "regression_cross_validation": ("main", XY, {}),
    "x_transform_regression": ("main", CURVE, {}),
    "polynomial_regression": ("main", CURVE, {}),
    "trigonometric_regression": ("main", {"x": "month", "y": "ytrig"}, {}),
    "nonlinear_regression": ("main", CURVE, {}),
    "smooth_curve": ("main", CURVE, {}),
    "curve_comparison": ("main", CURVE, {}),
    "multiple_regression": ("main", {"y": "ylin"}, MULTI),
    "regression_subsets": ("main", {"y": "ylin"}, MULTI),
    "stepwise_selection": ("main", {"y": "ylin"}, MULTI),
    "model_cross_validation": ("main", {"y": "ylin"}, {"models": "x; x, x2; x, x2, x3"}),
    "response_surface": ("main", {"y": "ylin"}, {"variables": "x, x2"}),
    "regression_with_groups": ("main", {"y": "ylin", "group": "site"}, {"variables": "x"}),
    "blocked_regression": ("main", {"value": "ylin", "x": "x", "block": "site"}, {}),
    "ancova": ("main", {"value": "y", "group": "trt", "covariate": "x"}, {}),
    "glm_analysis": ("main", {"value": "k", "trials": "m", "factor_a": "A",
                              "covariate": "x"}, {}),
    "count_regression": ("main", {"value": "count", "factor_a": "A"}, {}),
    "glmm_analysis": ("main", {"value": "k", "trials": "m", "factor_a": "A",
                               "covariate": "x", "stratum_1": "farm"}, {}),
    "ordinal_regression": ("main", {"response": "score", "factor_a": "A", "covariate": "x"},
                           {"levels": "none, low, mid, high"}),
    "curve_in_blocks": ("main", {"value": "ycurve", "x": "x", "block": "site"}, {}),
    "nonlinear_in_blocks": ("main", {"value": "ycurve", "x": "x", "block": "site"}, {}),
    "mixed_model": ("main", {"value": "y", "factor_a": "trt", "stratum_1": "farm"}, {}),
    "contingency_test": ("main", {"factor_a": "cat1", "factor_b": "yn"}, {}),
    "sampling_scheme_power": ("main", {}, {"components": "Блок=0.5; Делянка=2.0; Образец=1.0",
                                           "schemes": "4 3 2; 4 4 1", "delta": 2}),
    "spearman_correlation": ("main", XY, {}),
    "pearson_correlation": ("main", XY, {}),
}



RCBD_ROLES = {**RCBD, "x": "x"}  # роли для замера распознавателей карты
_TABLES: dict[tuple[str, int], bytes] = {}


def table_bytes(name: str, n: int) -> bytes:
    """CSV таблицы ``name`` на ~n строк (кэш в процессе)."""
    if (name, n) not in _TABLES:
        cols = {"main": lambda: main_table(-(-n // 12) * 12), "latin": latin_table,
                "bibd": lambda: bibd_table(n)}[name]()
        _TABLES[(name, n)] = to_csv(cols)
    return _TABLES[(name, n)]


def cases_json() -> str:
    """CASES и роли распознавателей для страницы замера в браузере."""
    return json.dumps({"cases": CASES, "roles": RCBD_ROLES})
