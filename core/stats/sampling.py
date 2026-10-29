"""Планирование вложенной выборки по компонентам дисперсии (Case Study 19.3 книги).

``sampling_scheme_power`` — RCBD с n_R блоками и n_T вариантами (линиями) на
экспериментальных единицах и вложенными подвыборками n_1, n_2, … (ветви на
растении, стручки на ветви). По компонентам дисперсии (из пилотного опыта,
разд. 16.4.1) предсказывает дисперсии страт фиктивной ANOVA (Table 19.4):
страта уровня j — Σ_{i ≥ j} σ²_i·Π(числа ниже i); для единиц
s²_U = n_1 n_2 σ²_U + n_2 σ²_1 + σ²_2. SE средней варианта = √(s²_U/(n_R·Πn)),
SED = √2·SE, df = (n_R − 1)(n_T − 1), мощность для разности δ — по гл. 10.
Несколько схем сравниваются при фиксированном бюджете измерений.
"""

from __future__ import annotations

import numpy as np

from core.report.schema import Severity, StatResult
from core.stats.power import power_t

_YES_NO = ("yes", "no")
_MAX_LEVELS = 6


def parse_components(text: str) -> list[tuple[str, float]]:
    """«Rep=0.005; Unit=0.0046; …» сверху вниз: блок, единица, подуровни."""
    out = []
    for part in [p for p in (text or "").replace(",", ";").split(";") if p.strip()]:
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Ожидалось имя=значение: {part.strip()!r}")
        out.append((name.strip(), float(value.replace(" ", ""))))
    if not 2 <= len(out) <= _MAX_LEVELS or any(v < 0 for _, v in out):
        raise ValueError(f"Нужно 2–{_MAX_LEVELS} неотрицательных компонент: блок, единица, …")
    return out


def parse_schemes(text: str, n_sub: int) -> list[list[int]]:
    """«60 3 2 2; 110 3 2 1» → [n_T, n_R, n_1 … n_k] для каждой схемы."""
    schemes = []
    for chunk in [c for c in (text or "").split(";") if c.strip()]:
        nums = [int(float(v)) for v in chunk.replace(",", " ").split()]
        if len(nums) != 2 + n_sub or min(nums) < 1 or nums[0] < 2 or nums[1] < 2:
            raise ValueError(f"Схема {chunk.strip()!r}: нужно n_T ≥ 2, n_R ≥ 2 и {n_sub} "
                             "чисел подвыборки ≥ 1")
        schemes.append(nums)
    if not schemes:
        raise ValueError("Задай хотя бы одну схему")
    return schemes


def stratum_table(components: list[tuple[str, float]], scheme: list[int],
                  positions: bool) -> list[dict]:
    """Фиктивная ANOVA со стратами, df и предсказанными дисперсиями (Table 19.4a)."""
    n_t, n_r, subs = scheme[0], scheme[1], scheme[2:]
    sigma = [v for _, v in components]
    counts = [n_r, n_t, *subs]  # число элементов уровня внутри родителя

    def below(j: int) -> int:
        return int(np.prod(counts[j + 1:])) if j + 1 < len(counts) else 1

    def variance(level: int) -> float:
        return float(sum(sigma[i] * below(i) for i in range(level, len(sigma))))

    rows = [{"stratum": components[0][0], "df": n_r - 1, "variance": variance(0)},
            {"stratum": components[1][0], "treatment_df": n_t - 1,
             "df": (n_r - 1) * (n_t - 1), "variance": variance(1)}]
    above = n_r * n_t
    for k, n_k in enumerate(subs, start=2):
        if n_k > 1:
            df = (above - 1) * (n_k - 1) if positions else above * (n_k - 1)
            rows.append({"stratum": components[k][0], "position_df": n_k - 1 if positions else 0,
                         "df": df, "variance": variance(k)})
        above *= n_k
    return rows


def evaluate(components: list[tuple[str, float]], scheme: list[int], delta: float,
             alpha: float, positions: bool) -> dict:
    """s²_U, SE, SED, df и мощность для разности δ двух вариантов."""
    table = stratum_table(components, scheme, positions)
    unit = table[1]
    per_treatment = scheme[1] * int(np.prod(scheme[2:]))
    se = float(np.sqrt(unit["variance"] / per_treatment))
    sed = float(np.sqrt(2) * se)
    return {"n_treatments": scheme[0], "n_blocks": scheme[1], "subsamples": scheme[2:],
            "measurements": scheme[0] * per_treatment, "s2_unit": unit["variance"],
            "df_unit": unit["df"], "se_mean": se, "sed": sed,
            "power": power_t(delta, sed, unit["df"], alpha) if delta > 0 else None,
            "strata": table}


def sampling_scheme_power(
    components: str = "",
    schemes: str = "",
    delta: float = 0.0,
    alpha: float = 0.05,
    position_terms: str = "yes",
) -> StatResult:
    """Сравнение схем вложенной выборки: SE, SED и мощность (Case Study 19.3, стр. 532–535)."""
    if position_terms not in _YES_NO or not 0 < alpha < 1 or delta < 0:
        raise ValueError("position_terms ∈ (yes, no), 0 < alpha < 1, delta ≥ 0")
    comps = parse_components(components)
    plans = parse_schemes(schemes, len(comps) - 2)
    rows = [evaluate(comps, p, delta, alpha, position_terms == "yes") for p in plans]
    result = StatResult(method="sampling_scheme_power", n=len(rows), parameters={
        "alpha": alpha, "delta": delta, "components": [{"term": n, "variance": v}
                                                       for n, v in comps], "schemes": rows})
    if delta > 0:
        best = max(range(len(rows)), key=lambda i: rows[i]["power"])
        result.statistics.update({"best_scheme": best + 1, "best_power": rows[best]["power"]})
    _df_warnings(rows, result)
    return result


def _df_warnings(rows: list[dict], result: StatResult) -> None:
    """Правило 10–20 остаточных df в страте сравнений вариантов (стр. 521)."""
    low = [str(i + 1) for i, r in enumerate(rows) if r["df_unit"] < 10]
    if low:
        result.add_warning("sampling_low_df", f"Схемы {', '.join(low)}: меньше 10 остаточных df в "
                           "страте вариантов — мощность мала, добавь блоки или варианты.",
                           Severity.WARNING, params={"schemes": ", ".join(low)})
