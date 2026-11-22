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


RUNNERS = {"one_way_anova": _one_way}


def example_values(method: str, cfg: dict) -> dict:
    """Величины примера для страницы метода method."""
    if method not in RUNNERS:
        raise KeyError(f"{method}: нет расчёта примера в tools/method_examples.py (RUNNERS)")
    return RUNNERS[method](cfg)
