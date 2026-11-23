"""Числа примера для страниц методов — расчётом ядра, не вручную.

Данные — content/examples/<name>.json (копия ROWS из web/src/exampleData.ts; тест
сверяет копию). Для каждого метода — свой расчёт в RUNNERS: какие функции ядра
вызвать и какие величины отдать странице (⟦имя⟧ в тексте и TeX).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLES = ROOT / "content" / "examples"


def load_example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def column(example: dict, name: str) -> list[str]:
    i = example["columns"].index(name)
    return [row[i] for row in example["rows"]]


def numbers(values: list[str]) -> list[float]:
    """Числа с десятичной запятой, как в примере сайта («5,2»)."""
    return [float(v.replace(",", ".")) for v in values]


def groups_of(example: dict, response: str, factor: str) -> list[tuple[str, list[float]]]:
    """(уровень фактора, значения отклика) в порядке первого появления — как в ядре."""
    out: dict[str, list[float]] = {}
    for level, y in zip(column(example, factor), numbers(column(example, response))):
        out.setdefault(level, []).append(y)
    return list(out.items())


def _one_way(cfg: dict) -> dict:
    """one_way_anova + SED и LSD из treatment_means (в one_way_anova их нет)."""
    from core.stats.anova import one_way_anova, treatment_means

    groups = groups_of(load_example(cfg["data"]), cfg["response"], cfg["factor"])
    labels, samples = [g for g, _ in groups], [y for _, y in groups]
    result = one_way_anova(*samples, labels=labels)
    means = treatment_means(*samples, labels=labels)
    values = {k: v for k, v in result.statistics.items() if isinstance(v, (int, float))}
    values |= {"p": result.p_value, "eta2": values["eta_squared"], "omega2": values["omega_squared"],
               "sed": means.statistics["sed"], "lsd": means.statistics["lsd"],
               "t": len(samples), "N": sum(len(s) for s in samples),
               "anova_table": result.parameters["anova_table"], "groups": groups}
    for est in result.estimates:
        label = est.name.removeprefix("mean_")
        values[f"mean_{label}"], values[f"se_{label}"] = est.value, est.std_error
    sizes = {len(s) for s in samples}
    if len(sizes) == 1:
        values["n_group"] = sizes.pop()
    return values


def _levels(cfg: dict) -> list[tuple[str, list[float]]]:
    """Группы примера; levels в cfg — взять только эти уровни (в их порядке)."""
    groups = dict(groups_of(load_example(cfg["data"]), cfg["response"], cfg["factor"]))
    return [(lv, groups[lv]) for lv in cfg.get("levels", groups)]


def _scalars(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if isinstance(v, (int, float))}


def _two_sample(cfg: dict) -> dict:
    """two_sample_t на двух уровнях примера (cfg["levels"])."""
    from core.stats.hypothesis import two_sample_t

    groups = _levels(cfg)
    (_, a), (_, b) = groups
    result = two_sample_t(a, b)
    return _scalars(result.statistics) | {"p": result.p_value, "variant": result.parameters["variant"],
                                          "groups": groups, "n_a": len(a), "n_b": len(b)}


def _welch(cfg: dict) -> dict:
    """welch_anova; для сравнения — обычный F и Бартлетт (variance_rule) на тех же данных."""
    from core.stats.anova import one_way_anova
    from core.stats.oneway import welch_anova
    import numpy as np

    from core.stats.variance_rule import bartlett, spreads_of

    groups = _levels(cfg)
    labels, samples = [g for g, _ in groups], [y for _, y in groups]
    result, classic = welch_anova(*samples, labels=labels), one_way_anova(*samples, labels=labels)
    x2, crit = bartlett(spreads_of([np.asarray(y) for y in samples]))
    values = _scalars(result.statistics) | {
        "p": result.p_value, "F_classic": classic.statistics["F"], "p_classic": classic.p_value,
        "df_classic_between": classic.statistics["df_between"], "df_classic_within": classic.statistics["df_within"],
        "bartlett_x2": x2, "bartlett_crit": crit, "groups": groups}
    return values | _sizes(samples)


def _permutation(cfg: dict) -> dict:
    """anova_permutation_test с фиксированным seed (страница воспроизводима)."""
    import numpy as np

    from core.stats.anova_checks import _n_arrangements, anova_permutation_test

    groups = _levels(cfg)
    result = anova_permutation_test(*[y for _, y in groups], labels=[g for g, _ in groups],
                                    n_permutations=cfg.get("n_permutations", 999), seed=cfg["seed"])
    samples = [np.asarray(y, dtype=float) for _, y in groups]
    return _scalars(result.statistics) | {"p": result.p_value, "seed": result.seed,
                                          "n_permutations": result.parameters["n_permutations"],
                                          "exact": result.parameters["exact"],
                                          "arrangements": _n_arrangements(samples),
                                          "groups": groups} | _sizes([y for _, y in groups])


def _sizes(samples: list[list[float]]) -> dict:
    sizes = {len(s) for s in samples}
    return {"t": len(samples), "N": sum(len(s) for s in samples)} | ({"n_group": sizes.pop()} if len(sizes) == 1 else {})


def _estimates(result, prefix: str = "") -> dict:
    """Оценки результата: <name> — значение, <name>_lo / _hi — границы ДИ (если есть)."""
    out = {}
    for est in result.estimates:
        out[prefix + est.name] = est.value
        if est.ci_lower is not None:
            out[f"{prefix}{est.name}_lo"], out[f"{prefix}{est.name}_hi"] = est.ci_lower, est.ci_upper
    return out


def _kruskal(cfg: dict) -> dict:
    """kruskal_wallis: H, поправка на совпадения, суммы рангов, медианы групп."""
    from core.stats.anova import kruskal_wallis

    groups = _levels(cfg)
    result = kruskal_wallis(*[y for _, y in groups], labels=[g for g, _ in groups])
    return (_scalars(result.statistics) | _estimates(result) | {"p": result.p_value, "groups": groups}
            | _sizes([y for _, y in groups]))


# Предупреждения о дисперсиях one_way_anova (_check_variances, _warn_two_groups) — их же
# выдаёт transformed_means на шкале анализа; тест сверяет набор с исходником ядра.
VARIANCE_WARNINGS = {"variance_heterogeneity", "variance_grows_with_mean", "variance_grows_with_mean_three",
                     "variance_two_groups", "unequal_variance_small_n"}


def _transformed(cfg: dict) -> dict:
    """transformed_means (log10 и т. п.) + арифметические средние treatment_means на исходной шкале."""
    from core.stats.anova import treatment_means
    from core.stats.transforms import transformed_means

    groups = _levels(cfg)
    labels, samples = [g for g, _ in groups], [y for _, y in groups]
    result = transformed_means(*samples, labels=labels, transform_kind=cfg.get("transform", "log10"))
    raw = treatment_means(*samples, labels=labels)
    log = result.parameters["transformed"]
    values = (_scalars(result.statistics) | _estimates(result) | _estimates(raw, "arith_")
              | {f"log_{k}": v for k, v in log["means"].items()}
              | {f"log_{k}": v for k, v in _scalars(log["statistics"]).items()})
    for r in result.parameters.get("ratios", []):
        key = f"ratio_{r['group2']}_{r['group1']}"
        values |= {key: r["ratio"], f"{key}_lo": r["ci_lower"], f"{key}_hi": r["ci_upper"]}
    n_var = sum(w.code in VARIANCE_WARNINGS for w in result.warnings)
    return values | {"p": result.p_value, "anova_table": result.parameters["anova_table"],
                     "n_variance_warnings": n_var, "groups": groups} | _sizes(samples)


def _rcbd(cfg: dict) -> dict:
    """rcbd_anova: отклик, вариант и блок из колонок примера."""
    from core.stats.blocking import rcbd_anova

    example = load_example(cfg["data"])
    result = rcbd_anova(numbers(column(example, cfg["response"])), column(example, cfg["factor"]),
                        column(example, cfg["block"]))
    labels, blocks = result.parameters["labels"], result.parameters["blocks"]
    return (_scalars(result.statistics) | _estimates(result)
            | {"p": result.p_value, "anova_table": result.parameters["anova_table"],
               "t": len(labels), "n_blocks": len(blocks), "N": len(labels) * len(blocks)})


RUNNERS = {"one_way_anova": _one_way, "two_sample_t": _two_sample, "welch_anova": _welch,
           "anova_permutation_test": _permutation, "kruskal_wallis": _kruskal,
           "transformed_means": _transformed, "rcbd_anova": _rcbd}


def example_values(method: str, cfg: dict) -> dict:
    """Величины примера для страницы метода method."""
    if method not in RUNNERS:
        raise KeyError(f"{method}: нет расчёта примера в tools/method_examples.py (RUNNERS)")
    return RUNNERS[method](cfg)
