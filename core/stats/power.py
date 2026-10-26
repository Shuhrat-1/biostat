"""Повторность, мощность и эквивалентность (гл. 10 книги).

- ``power_t`` — мощность t-теста разности по нецентральному t (10.3).
- ``replication_table`` — SED, LSD и мощность для n = 2…n_max в CRD,
  RCBD или латинском квадрате (10.1, Tables 10.1, 10.3).
- ``split_plot_power`` — SED, df Саттертуэйта и мощность сравнений
  в split-plot (Example 10.2, Table 10.4).
- ``replication_power`` — метод каталога: s² и %CV из пилотных данных
  (или заданные вручную) → таблица повторностей и минимальное n.
- ``equivalence_test`` — TOST каждого варианта против стандарта (10.5).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.contrasts import one_factor_fit
from core.stats.factorial import _design_error, _settings

DESIGNS = ("crd", "rcbd", "latin_square")
_ALTERNATIVES = ("two-sided", "greater", "less")


# --- Мощность -----------------------------------------------------------------


def power_t(delta: float, sed: float, df: float, alpha: float = 0.05,
            alternative: str = "two-sided") -> float:
    """Вероятность признать разность δ значимой: нецентральное t с c = δ/SED (стр. 248).

    Boost в scipy 1.12 выдаёт ложное «divide by zero» при малых df — значения верны.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _power_t(delta / sed, df, alpha, alternative)


def _power_t(nc: float, df: float, alpha: float, alternative: str) -> float:
    """Мощность по нецентральности nc: одно- или двусторонняя."""
    if alternative == "greater":
        return float(stats.nct.sf(stats.t.ppf(1 - alpha, df), df, nc))
    if alternative == "less":
        return float(stats.nct.cdf(-stats.t.ppf(1 - alpha, df), df, nc))
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))


def residual_df(design: str, n: int, t: int) -> int:
    """ResDF: CRD N − t, RCBD (n − 1)(t − 1), латинский квадрат (t − 1)(t − 2) (стр. 244–245)."""
    if design == "rcbd":
        return (n - 1) * (t - 1)
    if design == "latin_square":
        return (t - 1) * (t - 2)
    return n * t - t


def replication_table(s2: float, t: int, delta: float, alpha: float = 0.05,
                      design: str = "crd", n_max: int = 20,
                      alternative: str = "two-sided") -> list[dict]:
    """Строки n = 2…n_max: N, ResDF, t-критическое, SED = √(2s²/n), LSD, δ/SED, мощность."""
    if design not in DESIGNS:
        raise ValueError(f"design ∈ {DESIGNS}")
    rows = []
    tails = 2 if alternative == "two-sided" else 1
    for n in range(2, n_max + 1):
        if design == "latin_square" and n != t:
            continue
        df = residual_df(design, n, t)
        if df < 1:
            continue
        sed = float(np.sqrt(2 * s2 / n))
        t_crit = float(stats.t.ppf(1 - alpha / tails, df))
        rows.append({"n": n, "N": n * t, "df": df, "t_crit": t_crit, "sed": sed,
                     "lsd": t_crit * sed, "noncentrality": delta / sed,
                     "power": power_t(delta, sed, df, alpha, alternative)})
    return rows


def sigma_from_range(minimum: float, maximum: float) -> float:
    """σ ≈ (max − min)/4: 95 % нормальных наблюдений в μ ± 2σ (стр. 245)."""
    return (maximum - minimum) / 4


def split_plot_power(s2: float, sw2: float, t_a: int, t_b: int, m: int, delta: float,
                     alpha: float = 0.05) -> dict:
    """Мощность сравнений A внутри уровня B и главного эффекта A в split-plot (стр. 250–252)."""
    wp_df, res_df = (t_a - 1) * (m - 1), t_a * (t_b - 1) * (m - 1)
    combined = sw2 + (t_b - 1) * s2
    sed_within = float(np.sqrt(2 * combined / (t_b * m)))
    df_within = combined**2 / (sw2**2 / wp_df + ((t_b - 1) * s2) ** 2 / res_df)
    sed_main = float(np.sqrt(2 * sw2 / (t_b * m)))
    return {
        "within_b": {"sed": sed_within, "df": float(df_within),
                     "power": power_t(delta, sed_within, df_within, alpha)},
        "main_a": {"sed": sed_main, "df": wp_df, "power": power_t(delta, sed_main, wp_df, alpha)},
    }


# --- Планирование по пилотным данным ------------------------------------------


def replication_power(
    values: list[float],
    group: list[str],
    block: list[str] | None = None,
    delta: float = 0.0,
    alpha: float = 0.05,
    target_power: float = 0.8,
    n_max: int = 30,
    s2_override: float = 0.0,
    alternative: str = "two-sided",
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Повторность по пилоту: LSD ≤ δ и мощность ≥ цели (10.1–10.3).

    s² — ResMS пилота или ``s2_override``; дизайн (CRD/RCBD) и t — как в пилоте.
    """
    settings = _settings(alpha, "none", 0.0)
    result, fit = one_factor_fit(values, group, block, settings, "replication_power",
                                 column_names)
    if fit is None or not _planning_args_ok(delta, target_power, alternative, result):
        return result
    s2 = s2_override if s2_override > 0 else fit["s2"]
    design = "rcbd" if block is not None else "crd"
    t = len(fit["levels"])
    rows = replication_table(s2, t, delta, alpha, design, max(n_max, 2), alternative)
    _fill_planning(rows, fit, s2, delta, target_power, result)
    result.parameters.update({"design": design, "n_treatments": t, "delta": delta,
                              "target_power": target_power, "alternative": alternative,
                              "s2_source": "override" if s2_override > 0 else "pilot"})
    return result


def _planning_args_ok(delta: float, target: float, alternative: str,
                      result: StatResult) -> bool:
    """δ > 0, целевая мощность в (0, 1), известный вид гипотезы."""
    if delta > 0 and 0 < target < 1 and alternative in _ALTERNATIVES:
        return True
    return _design_error(result, "power_invalid",
                         "Задай разность δ > 0, целевую мощность в (0, 1) и вид гипотезы")


def _fill_planning(rows: list[dict], fit: dict, s2: float, delta: float, target: float,
                   result: StatResult) -> None:
    """Таблица повторностей, минимальные n, %CV и LSD в долях среднего (10.1.2)."""
    grand = float(np.average(fit["means"], weights=fit["n"]))
    cv = 100 * np.sqrt(s2) / grand if grand > 0 else None
    for row in rows:
        row["lsd_fraction_of_mean"] = row["lsd"] / grand if grand > 0 else None
    n_lsd = next((r["n"] for r in rows if r["lsd"] <= delta), None)
    n_power = next((r["n"] for r in rows if r["power"] >= target), None)
    result.parameters["replication_table"] = rows
    result.statistics.update({"s2_planning": s2, "grand_mean": grand,
                              "n_required_lsd": n_lsd, "n_required_power": n_power})
    if cv is not None:
        result.statistics["cv_pct"] = float(cv)
    if n_power is None:
        result.add_warning("power_not_reached",
                           "Целевая мощность не достигнута в пределах n_max: уменьши число "
                           "вариантов, снизь фоновую вариацию или повтори опыт.",
                           Severity.INFO, params={"n_max": rows[-1]["n"] if rows else None})
    if rows and rows[0]["df"] < 10:
        result.add_warning("few_residual_df",
                           "Меньше 10 остаточных df при малых n: оценка фоновой вариации "
                           "ненадёжна (правило стр. 245).", Severity.INFO)


# --- Эквивалентность (TOST) ---------------------------------------------------


def _combine(group: list[str], factor_b: list[str] | None, factor_c: list[str] | None,
             column_names: dict[str, str] | None) -> tuple[list[str], dict[str, str] | None]:
    """Вариант как комбинация до трёх факторов: «L × 20 × 30» (имя — Treatment)."""
    parts = [p for p in (group, factor_b, factor_c) if p is not None]
    labels = [" × ".join(str(v).strip() for v in combo) if all(str(v).strip() for v in combo)
              else "" for combo in zip(*parts)]
    if len(parts) > 1:
        column_names = (column_names or {}) | {"group": "Treatment"}
    return labels, column_names


def _match_level(control: str, levels: list[str]) -> int:
    """Номер уровня-стандарта; «S 200 60» совпадает с «S × 200 × 60»."""
    def norm(text: str) -> str:
        return " ".join(text.replace("×", " ").replace(",", " ").lower().split())
    lookup = {norm(lv): i for i, lv in enumerate(levels)}
    if norm(control) not in lookup:
        raise ValueError(f"Стандарт «{control}» не найден; уровни: {', '.join(levels)}")
    return lookup[norm(control)]


def equivalence_test(
    values: list[float],
    group: list[str],
    factor_b: list[str] | None = None,
    factor_c: list[str] | None = None,
    block: list[str] | None = None,
    control: str = "",
    margin: float = 0.0,
    margin_kind: str = "absolute",
    alpha: float = 0.05,
    adjustment: str = "none",
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """TOST каждого варианта против стандарта на ±c, c — абсолютно или в % (Ex. 10.3).

    Эквивалентность: 100(1 − 2α)% ДИ внутри (−c, c); неэквивалентность — целиком вне.
    """
    settings = _settings(alpha, "none", 0.0)
    labels, column_names = _combine(group, factor_b, factor_c, column_names)
    result, fit = one_factor_fit(values, labels, block, settings, "equivalence_test",
                                 column_names)
    if fit is None:
        return result
    ctrl = _equivalence_args(fit, control, margin, margin_kind, adjustment, result)
    if ctrl is None:
        return result
    c = margin if margin_kind == "absolute" else margin / 100 * fit["means"][ctrl]
    rows = _tost_rows(fit, ctrl, c, alpha, adjustment)
    _fill_tost(rows, fit, ctrl, c, settings, result)
    result.parameters.update({"adjustment": adjustment, "margin_kind": margin_kind})
    return result


def _equivalence_args(fit: dict, control: str, margin: float, margin_kind: str,
                      adjustment: str, result: StatResult) -> int | None:
    """Номер стандарта или None с ошибкой equivalence_invalid."""
    try:
        ctrl = _match_level(control, fit["levels"])
        if margin <= 0 or margin_kind not in ("absolute", "percent") \
                or adjustment not in ("none", "bonferroni"):
            raise ValueError("Задай границу эквивалентности c > 0 (absolute или percent)")
    except ValueError as exc:
        _design_error(result, "equivalence_invalid", str(exc))
        return None
    return ctrl


def _tost_rows(fit: dict, ctrl: int, c: float, alpha: float, adjustment: str) -> list[dict]:
    """t_a = (d + c)/SED, t_b = (d − c)/SED, P TOST = max(P_a, P_b), ДИ 100(1 − 2α)% (стр. 253)."""
    others = [j for j in range(len(fit["levels"])) if j != ctrl]
    a = alpha / len(others) if adjustment == "bonferroni" else alpha
    t_crit = float(stats.t.ppf(1 - a, fit["df"]))
    rows = []
    for j in others:
        d = float(fit["means"][j] - fit["means"][ctrl])
        sed = float(np.sqrt(fit["s2"] * (1 / fit["n"][j] + 1 / fit["n"][ctrl])))
        t_a, t_b = (d + c) / sed, (d - c) / sed
        p_tost = max(stats.t.sf(t_a, fit["df"]), stats.t.cdf(t_b, fit["df"]))
        lower, upper = d - t_crit * sed, d + t_crit * sed
        rows.append({"level": fit["levels"][j], "mean": float(fit["means"][j]),
                     "difference": d, "sed": sed, "t_a": t_a, "t_b": t_b,
                     "p_tost": float(p_tost), "ci_lower": lower, "ci_upper": upper,
                     "equivalent": bool(-c < lower and upper < c),
                     "inequivalent": bool(upper < -c or lower > c)})
    return rows


def _fill_tost(rows: list[dict], fit: dict, ctrl: int, c: float, settings: dict,
               result: StatResult) -> None:
    """Таблица TOST, оценки разностей, мощность обычного t-теста для δ = c."""
    alpha = settings["alpha"]
    level = 1 - 2 * alpha
    for row in rows:
        result.estimates.append(Estimate(name=f"{row['level']} − {fit['levels'][ctrl]}",
                                         value=row["difference"], ci_lower=row["ci_lower"],
                                         ci_upper=row["ci_upper"], ci_level=level,
                                         std_error=row["sed"]))
    sed = rows[0]["sed"] if rows else None
    result.parameters.update({"equivalence": rows, "control": fit["levels"][ctrl],
                              "control_mean": float(fit["means"][ctrl]), "margin": c,
                              "ci_level": level})
    result.statistics.update({
        "n_equivalent": sum(r["equivalent"] for r in rows),
        "n_inequivalent": sum(r["inequivalent"] for r in rows),
    })
    if sed:
        result.statistics["power_difference_c"] = power_t(c, sed, fit["df"], alpha)
    if not any(r["equivalent"] or r["inequivalent"] for r in rows):
        result.add_warning("equivalence_inconclusive",
                           "Ни эквивалентность, ни неэквивалентность не доказаны: опыту не "
                           "хватает точности для выбранной границы c.", Severity.INFO)
