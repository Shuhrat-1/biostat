"""Однофакторный ANOVA повторных измерений (Witte, гл. 17; пакет W4).

Каждый субъект проходит все k условий. Разбиение SS_total = SS_between +
SS_subject + SS_error (ф. 17.1), F = MS_between / MS_error (ф. 17.5),
частичная η² (ф. 17.6).

Сферичность [Witte, стр. 336]: тест Маучли выводится как информация, рядом с
обычным P всегда P с поправками Гринхауса–Гейссера и Хюйнха–Фельдта
[предл.: Mauchly 1940; Greenhouse & Geisser 1959; Huynh & Feldt 1976].
Переключения по Маучли нет (решение владельца: двухшаговая процедура
искажает ошибку I рода).

Сравнения пар — два блока, оба всегда: HSD Тьюки на MS_error с d (ф. 17.7–17.8,
при условии сферичности) и парные t по разностям с поправкой Холма и d_z
[предл.: Maxwell & Delaney 2004] — сферичность им не нужна.

При пропусках неполные субъекты исключаются; для таких данных — mixed_model
(Welham, гл. 16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from core.report.schema import Severity, StatResult


def _matrix(
    values: list[float], condition: list[str], subject: list[str]
) -> tuple[pd.DataFrame, list[str], int]:
    """Длинный формат → матрица субъекты × условия; неполные субъекты убраны.

    Условия — в порядке первого появления. Возвращает (матрица, условия,
    число исключённых субъектов). ValueError при повторе ячейки.
    """
    if not len(values) == len(condition) == len(subject):
        raise ValueError("Колонки должны быть одной длины")
    frame = pd.DataFrame({"y": pd.to_numeric(pd.Series(values), errors="coerce"),
                          "cond": pd.Series(condition).astype(str).str.strip(),
                          "subj": pd.Series(subject).astype(str).str.strip()})
    frame = frame[(frame["cond"] != "") & (frame["subj"] != "")
                  & ~frame["cond"].isin(["nan", "None"]) & ~frame["subj"].isin(["nan", "None"])]
    if frame.duplicated(["subj", "cond"]).any():
        raise ValueError("У субъекта несколько значений в одном условии: нужна одна ячейка "
                         "субъект × условие (усредните повторности заранее)")
    levels = list(dict.fromkeys(frame["cond"]))
    wide = frame.pivot(index="subj", columns="cond", values="y").reindex(columns=levels)
    complete = wide.dropna()
    return complete, levels, int(len(wide) - len(complete))


def anova_table(y: np.ndarray) -> dict[str, float]:
    """SS, df, MS, F, P для матрицы n × k (Table 17.2, 17.4)."""
    n, k = y.shape
    grand = y.mean()
    ss_between = float(n * np.sum((y.mean(axis=0) - grand) ** 2))
    ss_subject = float(k * np.sum((y.mean(axis=1) - grand) ** 2))
    ss_total = float(np.sum((y - grand) ** 2))
    ss_error = max(ss_total - ss_between - ss_subject, 0.0)
    df_b, df_s, df_e = k - 1, n - 1, (k - 1) * (n - 1)
    ms_b, ms_e = ss_between / df_b, ss_error / df_e
    f = ms_b / ms_e if ms_e > 0 else float("inf")
    return {"ss_between": ss_between, "ss_subject": ss_subject, "ss_error": ss_error,
            "ss_within": ss_subject + ss_error, "ss_total": ss_total,
            "df_between": df_b, "df_subject": df_s, "df_error": df_e,
            "ms_between": ms_b, "ms_error": ms_e, "F": f,
            "p": float(stats.f.sf(f, df_b, df_e)),
            "eta_squared_partial": ss_between / (ss_between + ss_error)
            if ss_between + ss_error > 0 else float("nan")}


def _orthonormal_contrasts(k: int) -> np.ndarray:
    """(k − 1) × k ортонормированные контрасты (нормированный Хелмерт)."""
    c = np.zeros((k - 1, k))
    for i in range(1, k):
        c[i - 1, :i] = 1.0
        c[i - 1, i] = -float(i)
        c[i - 1] /= np.sqrt(i * (i + 1))
    return c


def sphericity(y: np.ndarray) -> dict[str, float | None]:
    """ε ГГ, ε ХФ (обрезан до 1), нижняя граница 1/(k − 1) и тест Маучли.

    M = C·S·Cᵀ, S — ковариация условий, C — ортонормированные контрасты.
    ε_GG = tr(M)² / ((k − 1)·tr(M²)); ε_HF = (n(k − 1)ε_GG − 2) / ((k − 1)(n − 1 − (k − 1)ε_GG)).
    Маучли: W = det M / (tr M / (k − 1))^(k − 1), p = k − 1, df = p(p + 1)/2 − 1;
    χ² и P — в _mauchly. При n − 1 < k − 1 или det M ≤ 0 W не определён (None).
    """
    n, k = y.shape
    p = k - 1
    m = _orthonormal_contrasts(k) @ np.cov(y, rowvar=False, ddof=1) @ _orthonormal_contrasts(k).T
    tr = float(np.trace(m))
    eps_gg = tr**2 / (p * float(np.trace(m @ m))) if tr > 0 else float("nan")
    denom = p * (n - 1 - p * eps_gg)
    eps_hf = min(1.0, (n * p * eps_gg - 2) / denom) if denom > 0 else 1.0
    # ε_HF − ε_GG = (pε − 1)(pε + 2)/знаменатель ≥ 0, т. к. pε_GG ≥ 1: обрезка только сверху.
    out: dict[str, float | None] = {"epsilon_gg": eps_gg, "epsilon_hf": eps_hf,
                                    "epsilon_lower_bound": 1 / p,
                                    "mauchly_w": None, "mauchly_chi2": None,
                                    "mauchly_df": p * (p + 1) / 2 - 1, "mauchly_p": None}
    if n - 1 >= p and tr > 0:
        sign, logdet = np.linalg.slogdet(m)
        if sign > 0:
            out.update(_mauchly(logdet - p * np.log(tr / p), n - 1, p, k))
    return out


def _mauchly(log_w: float, df: int, p: int, k: int) -> dict[str, float]:
    """χ² Маучли с поправкой второго порядка — как R mauchly.test [предл.: Anderson 2003].

    f = 1 − (2p² + p + 2)/(6p·df); χ² = −df·f·ln W; P = P₁ + ω₂(P₂ − P₁), где P₁, P₂ —
    хвосты χ² с ν и ν + 4 df; ω₂ = 0 при p = 2 (k = 3).
    """
    f = 1 - (2 * p**2 + p + 2) / (6 * p * df)
    nu = p * (p + 1) / 2 - 1
    w2 = ((p + 2) * (p - 1) * (p - 2) * (2 * p**3 + 6 * p**2 + 3 * k + 2)
          / (288 * (df * p * f) ** 2))
    chi2 = -df * f * log_w
    p1, p2 = stats.chi2.sf(chi2, nu), stats.chi2.sf(chi2, nu + 4)
    return {"mauchly_w": float(np.exp(log_w)), "mauchly_chi2": float(chi2),
            "mauchly_p": float(p1 + w2 * (p2 - p1))}


def _tukey_rows(y: np.ndarray, levels: list[str], table: dict, alpha: float) -> list[dict]:
    """HSD Тьюки на MS_error (ф. 17.7), d = разность/√MS_error (ф. 17.8)."""
    n, k = y.shape
    se = np.sqrt(table["ms_error"] / n)
    hsd = float(stats.studentized_range.ppf(1 - alpha, k, table["df_error"]) * se)
    means, sd = y.mean(axis=0), np.sqrt(table["ms_error"])
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(means[j] - means[i])
            p = float(stats.studentized_range.sf(abs(diff) / se, k, table["df_error"]))
            rows.append({"group1": levels[i], "group2": levels[j], "mean_diff": diff,
                         "p_adj": p, "ci_lower": diff - hsd, "ci_upper": diff + hsd,
                         "reject": bool(p < alpha), "cohens_d": diff / sd if sd > 0 else float("nan")})
    return rows


def _paired_rows(y: np.ndarray, levels: list[str], alpha: float) -> list[dict]:
    """Парные t по разностям, P по Холму, ДИ по Бонферрони, d_z = D̄/s_D."""
    n, k = y.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    t_crit = float(stats.t.ppf(1 - alpha / (2 * len(pairs)), n - 1))
    rows = []
    for i, j in pairs:
        d = y[:, j] - y[:, i]
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        se = sd / np.sqrt(n)
        t = mean / se if se > 0 else float("nan")
        rows.append({"group1": levels[i], "group2": levels[j], "mean_diff": mean, "t": t,
                     "df": n - 1, "p": float(2 * stats.t.sf(abs(t), n - 1)),
                     "ci_lower": mean - t_crit * se, "ci_upper": mean + t_crit * se,
                     "cohens_dz": mean / sd if sd > 0 else float("nan")})
    p_holm = multipletests([r["p"] for r in rows], alpha=alpha, method="holm")[1]
    for row, p_adj in zip(rows, p_holm):
        row["p_adj"] = float(p_adj)
        row["reject"] = bool(p_adj < alpha)
    return rows


def _corrected_p(table: dict, eps: dict) -> dict[str, float]:
    """P с df, умноженными на ε (ГГ, ХФ)."""
    out = {}
    for key, name in (("epsilon_gg", "p_gg"), ("epsilon_hf", "p_hf")):
        e = eps[key]
        out[name] = float(stats.f.sf(table["F"], e * table["df_between"], e * table["df_error"]))
    return out


def _guard(y: pd.DataFrame, levels: list[str], dropped: int, result: StatResult) -> bool:
    """Ошибки: < 2 условий, < 2 полных субъектов; предупреждение об исключённых."""
    if dropped:
        result.add_warning(
            "rm_incomplete_subjects",
            f"Исключено субъектов с пропусками: {dropped}. При пропусках используйте "
            f"mixed_model (Welham, гл. 16) — он не требует полных данных.",
            Severity.WARNING, params={"dropped": dropped})
    if len(levels) < 2:
        result.add_warning("too_few_groups", "Нужно ≥ 2 условий", Severity.ERROR)
        return False
    if len(y) < 2:
        result.add_warning("n_too_small", "Нужно ≥ 2 субъектов с полными данными",
                           Severity.ERROR)
        return False
    return True


def _sphericity_block(y: np.ndarray, table: dict, alpha: float, result: StatResult) -> None:
    """ε и P с поправками в statistics; k = 2 — ε = 1 и пояснение."""
    if y.shape[1] == 2:
        result.statistics.update({"epsilon_gg": 1.0, "epsilon_hf": 1.0,
                                  "epsilon_lower_bound": 1.0,
                                  "p_gg": table["p"], "p_hf": table["p"]})
        result.add_warning(
            "rm_two_conditions",
            "Два условия: сферичность выполняется всегда (одна разность), ε = 1, "
            "тест Маучли не нужен. Анализ равносилен парному t-тесту.",
            Severity.INFO)
        return
    eps = sphericity(y)
    result.statistics.update({**eps, **_corrected_p(table, eps)})
    if eps["mauchly_w"] is None:
        result.add_warning(
            "rm_mauchly_undefined",
            "Тест Маучли не определён: субъектов слишком мало или ковариация разностей "
            "вырождена. Поправки ε посчитаны.",
            Severity.INFO)
    if (table["p"] < alpha) != (result.statistics["p_gg"] < alpha):
        result.add_warning(
            "rm_sphericity_conflict",
            f"Вывод зависит от сферичности: обычный P = {table['p']:.4f}, "
            f"с поправкой Гринхауса–Гейссера P = {result.statistics['p_gg']:.4f} "
            f"(α = {alpha}). Сравните также блок парных t с поправкой Холма.",
            Severity.WARNING,
            params={"p": round(table["p"], 4), "p_gg": round(result.statistics["p_gg"], 4),
                    "alpha": alpha})


def rm_anova(
    values: list[float],
    condition: list[str],
    subject: list[str],
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """ANOVA повторных измерений: k условий у одних и тех же субъектов (Witte, гл. 17).

    Длинный формат: значение, условие, субъект. Неполные субъекты исключаются.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")
    wide, levels, dropped = _matrix(values, condition, subject)
    result = StatResult(method="rm_anova", n=int(wide.size),
                        parameters={"alpha": alpha, "labels": levels,
                                    "column_names": column_names or {}})
    if not _guard(wide, levels, dropped, result):
        return result
    y = wide.to_numpy(dtype=float)
    table = anova_table(y)
    if table["ms_error"] == 0:
        result.add_warning("rm_zero_error",
                           "MS_error = 0: все субъекты меняются одинаково, F не определён.",
                           Severity.ERROR)
        return result
    result.p_value = table.pop("p")
    result.statistics = {**table, "n_subjects": len(wide), "n_conditions": len(levels),
                         "n_excluded_subjects": dropped}
    _sphericity_block(y, {**table, "p": result.p_value}, alpha, result)
    _fill_parameters(y, wide, levels, table, alpha, result)
    return result


def _fill_parameters(y: np.ndarray, wide: pd.DataFrame, levels: list[str], table: dict,
                     alpha: float, result: StatResult) -> None:
    """Таблица ANOVA, средние, два блока сравнений, профили субъектов."""
    result.parameters.update({
        "anova_table": [
            {"term": "Between", "ss": table["ss_between"], "df": table["df_between"],
             "ms": table["ms_between"], "F": table["F"], "p": result.p_value},
            {"term": "Subject", "ss": table["ss_subject"], "df": table["df_subject"]},
            {"term": "Error", "ss": table["ss_error"], "df": table["df_error"],
             "ms": table["ms_error"]},
            {"term": "Total", "ss": table["ss_total"],
             "df": table["df_between"] + table["df_subject"] + table["df_error"]},
        ],
        "means": [{"level": lv, "n": int(y.shape[0]), "mean": float(m)}
                  for lv, m in zip(levels, y.mean(axis=0))],
        # Блок 1 — по книге (ф. 17.7–17.8), корректен при сферичности.
        "comparisons": _tukey_rows(y, levels, table, alpha),
        "comparisons_assumption": "sphericity",
        # Блок 2 — без допущения сферичности [предл.: Maxwell & Delaney 2004].
        "paired_comparisons": _paired_rows(y, levels, alpha),
        "profiles": {"subjects": [str(s) for s in wide.index], "conditions": levels,
                     "values": y.tolist()},
    })
