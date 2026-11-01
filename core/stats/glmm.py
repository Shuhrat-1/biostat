"""Обобщённые линейные смешанные модели (GLMM) — экспериментальный метод (гл. 18 + 16).

``glmm_analysis`` — биномиальная или пуассоновская GLM (как ``glm_analysis``)
плюс случайные интерсепты по 1–3 структурным факторам (скрещённым или
вложенным: ферма, ферма.поле …). Книга упоминает GLMM как обобщение гл. 16 на
гл. 18 (стр. 512–513), но без численного примера; GenStat/R считают их
приближённо — так же и здесь (движок ``glmm_engine``: Лаплас или адаптивная
квадратура Гаусса–Эрмита для одного фактора).

- Компоненты дисперсии σ²_k (шкала предиктора), LR-тест σ²_k = 0 против модели
  без компоненты с поправкой на границу (P/2 — смесь χ²₀ и χ²₁).
- Фиксированные параметры: z-тесты Вальда (SE по гессиану −2ℓ), χ² Вальда
  членов, предсказания и сравнения уровней при b = 0 («типичная» группа, не
  среднее по популяции групп).
- Условные моды b по группам с условной SD; проверка сверхдисперсии по χ²
  Пирсона (условные остатки) — при её наличии добавить фактор «наблюдение».

Результат ВСЕГДА несёт предупреждение ``glmm_experimental``: оценки
приближённые, для ответственных решений сверять с R (lme4) или GenStat.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from scipy import sparse, stats

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import _frame, _label
from core.stats.factorial import _new_result
from core.stats.glm import (
    Model,
    _check_exposure,
    _check_options,
    _check_response,
    _family,
    comparisons,
    factor_predictions,
    model_terms,
    parameter_rows,
)
from core.stats.glmm_engine import (
    GLMMData,
    fit_glmm,
    fit_scaled,
    inverse_link,
    laplace,
    scaling_matrix,
)
from core.stats.regression_checks import _transform_column

EXPERIMENTAL = ("Экспериментальный движок, оценки приближённые (Лаплас/квадратура), для "
                "ответственных решений сверяйте с R/GenStat.")
_RANDOM = ("r1", "r2", "r3")
_SLOTS = {"A": "factor_a", "B": "factor_b", "x": "covariate", "r1": "stratum_1",
          "r2": "stratum_2", "r3": "stratum_3"}
_STRUCTURES = ("crossed", "nested")
_METHODS = ("laplace", "agq")
_MAX_N, _MAX_Q = 5000, 2000
_FEW_LEVELS = 5
_FIXED_DISP = {"phi": 1.0, "estimated": False}


# --- Данные и план ----------------------------------------------------------------


def _collect(values: list[float], slots: dict[str, list | None], trials: list | None,
             exposure: list | None, n_trials: int) -> tuple[pd.DataFrame, int]:
    """Полные строки: отклик, m, ковариата, экспозиция, фиксированные и случайные факторы."""
    numeric = {k: v for k, v in (("x", slots["x"]), ("m", trials), ("e", exposure))
               if v is not None}
    factors = {k: slots[k] for k in ("A", "B", *_RANDOM)}
    frame, dropped = _frame(values, numeric, factors)
    if trials is None and n_trials > 0:
        frame["m"] = float(n_trials)
    if "e" in frame and (frame["e"] > 0).all():
        frame["off"] = np.log(frame["e"])
    return frame, dropped


def random_blocks(frame: pd.DataFrame, keys: list[str], nested: bool) -> list[tuple]:
    """Коды уровней каждого случайного члена: r1, r2, … или r1, r1.r2, … (вложенные)."""
    out, key = [], pd.Series([""] * len(frame), index=frame.index)
    for k in keys:
        key = key + "|" + frame[k] if nested else frame[k]
        codes, levels = pd.factorize(key, sort=True)
        out.append((codes, [str(v).strip("|").replace("|", ".") for v in levels]))
    return out


def random_design(blocks: list[tuple]) -> tuple[sparse.csr_matrix, list[np.ndarray]]:
    """Разреженная Z из индикаторов и индексы столбцов каждого члена."""
    parts, groups, start = [], [], 0
    for codes, levels in blocks:
        n, q = codes.size, len(levels)
        parts.append(sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, q)))
        groups.append(np.arange(start, start + q))
        start += q
    return sparse.hstack(parts, format="csr"), groups


def glmm_data(frame: pd.DataFrame, x: np.ndarray, blocks: list[tuple], distribution: str,
              link: str) -> GLMMData:
    """Данные движка: y, m (1 для Пуассона), X, Z, смещение."""
    z, groups = random_design(blocks)
    m = frame["m"].to_numpy(float) if "m" in frame else np.ones(len(frame))
    off = frame["off"].to_numpy(float) if "off" in frame else None
    return GLMMData(frame["y"].to_numpy(float), m, x, z, groups, distribution, link, off)


def _drop(d: GLMMData, k: int) -> GLMMData:
    """Данные без k-го случайного члена (для LR-теста его дисперсии)."""
    keep = np.concatenate([g for i, g in enumerate(d.groups) if i != k])
    groups, start = [], 0
    for i, g in enumerate(d.groups):
        if i != k:
            groups.append(np.arange(start, start + g.size))
            start += g.size
    return GLMMData(d.y, d.m, d.x, d.z[:, keep], groups, d.family, d.link, d.offset)


# --- Тесты и сводки ---------------------------------------------------------------


def null_m2ll(d: GLMMData, beta: np.ndarray) -> float:
    """−2ℓ обычной GLM (θ = 0) в тех же константах, что и GLMM."""
    return laplace(d, np.zeros(len(d.groups)), beta)["m2ll"]


def boundary_p(lr: float, k: int) -> float:
    """P для k дисперсий на границе 0: смесь Σ C(k, j)/2ᵏ χ²_j (χ²₀ — точка 0)."""
    if lr <= 1e-8:
        return 1.0
    return float(sum(comb(k, j) / 2 ** k * stats.chi2.sf(lr, j) for j in range(1, k + 1)))


def variance_tests(d: GLMMData, fit: dict, glm_beta: np.ndarray, n_agq: int) -> list[float]:
    """−2ℓ моделей без каждого случайного члена (один член — обычная GLM)."""
    if len(d.groups) == 1:
        return [null_m2ll(d, glm_beta)]
    xs, m = scaling_matrix(d.x)
    out = []
    for k in range(len(d.groups)):
        sub = _drop(d, k)
        sub.x = xs
        out.append(fit_glmm(sub, np.linalg.solve(m, fit["beta"]), n_agq)["m2ll"])
    return out


def variance_rows(fit: dict, blocks: list[tuple], reduced: list[float],
                  labels: list[str]) -> list[dict]:
    """σ², SD, число уровней и LR-тест σ² = 0 для каждого случайного члена."""
    rows = []
    for k, (theta, (_, levels)) in enumerate(zip(fit["theta"], blocks)):
        lr = max(reduced[k] - fit["m2ll"], 0.0)
        rows.append({"term": labels[k], "variance": float(theta ** 2), "sd": float(theta),
                     "levels": len(levels), "lr": lr, "p": boundary_p(lr, 1)})
    return rows


def wald_terms(model: Model, glm_fit: dict, cov: np.ndarray, beta: np.ndarray,
               names: dict[str, str]) -> list[dict]:
    """χ² Вальда каждого члена (все его столбцы = 0) при прочих членах модели."""
    rows = []
    for term in glm_fit["terms"]:
        idx = glm_fit["info"]["index"][term]
        if not idx:
            continue
        b, v = beta[idx], cov[np.ix_(idx, idx)]
        chi2 = float(b @ np.linalg.pinv(v) @ b)
        rows.append({"term": _label(term, names), "df": len(idx), "chi2": chi2,
                     "p": float(stats.chi2.sf(chi2, len(idx))),
                     "marginal": any(set(term) < set(o) for o in glm_fit["terms"])})
    return rows


def conditional_modes(fit: dict, blocks: list[tuple], labels: list[str]) -> list[dict]:
    """b_j = θ_k·u_j и условная SD θ_k·√[(I + ΛZᵀWZΛ)⁻¹]_jj по уровням."""
    lu, q = fit["mode"]["lu"], fit["u"].size
    diag = np.concatenate([np.diag(lu.solve(np.eye(q)[:, s:s + 200]), -s)[:min(200, q - s)]
                           for s in range(0, q, 200)])
    lam = fit["mode"]["lam"]
    out, start = [], 0
    for label, (_, levels) in zip(labels, blocks):
        for j, lev in enumerate(levels, start):
            out.append({"term": label, "level": lev, "b": float(lam[j] * fit["u"][j]),
                        "sd": float(lam[j] * np.sqrt(diag[j]))})
        start += len(levels)
    return out


def pearson_check(d: GLMMData, fit: dict) -> dict:
    """Χ² Пирсона по условным остаткам против χ²(n − p − k) (как overdisp_fun Болкера)."""
    p = inverse_link(fit["eta"], d.link)[0]
    mean = d.m * p
    var = mean if d.family == "poisson" else d.m * p * (1 - p)
    r = (d.y - mean) / np.sqrt(np.maximum(var, 1e-300))
    df = d.y.size - d.x.shape[1] - len(d.groups)
    chi2 = float(r @ r)
    return {"chi2": chi2, "df": df, "ratio": chi2 / df if df > 0 else float("nan"),
            "p": float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan"),
            "residuals": r, "fitted": mean}


def residual_plots(check: dict) -> dict:
    """Условные остатки Пирсона против условных подогнанных и Q–Q (Блом)."""
    r = check["residuals"]
    blom = stats.norm.ppf((np.arange(1, r.size + 1) - 0.375) / (r.size + 0.25))
    return {"residual_vs_fitted": {"fitted": check["fitted"].tolist(), "residuals": r.tolist()},
            "qq": {"theoretical_quantiles": blom.tolist(), "sample_quantiles": np.sort(r).tolist()},
            "std_residuals": r.tolist()}


# --- Метод каталога -------------------------------------------------------------


def glmm_analysis(
    values: list[float],
    stratum_1: list[str],
    factor_a: list[str] | None = None,
    factor_b: list[str] | None = None,
    covariate: list | None = None,
    stratum_2: list[str] | None = None,
    stratum_3: list[str] | None = None,
    trials: list | None = None,
    exposure: list | None = None,
    distribution: str = "binomial",
    link: str = "canonical",
    n_trials: int = 0,
    covariate_transform: str = "none",
    covariate_offset: float = 0.0,
    interactions: str = "yes",
    random_structure: str = "crossed",
    method: str = "laplace",
    n_agq: int = 10,
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """GLMM: биномиальная/пуассоновская GLM + случайные интерсепты 1–3 факторов (экспериментально)."""
    link = _check_options(distribution, link, "fixed", interactions, alpha)
    if random_structure not in _STRUCTURES or method not in _METHODS or not 1 <= n_agq <= 50:
        raise ValueError(f"random_structure ∈ {_STRUCTURES}, method ∈ {_METHODS}, n_agq 1–50")
    slots = {"A": factor_a, "B": factor_b, "x": covariate, "r1": stratum_1, "r2": stratum_2,
             "r3": stratum_3}
    frame, dropped = _collect(values, slots, trials, exposure, int(n_trials))
    fixed = [k for k in ("A", "B") if slots[k] is not None]
    variates = ["x"] if covariate is not None else []
    rand = [k for k in _RANDOM if slots[k] is not None]
    names = {k: (column_names or {}).get(_SLOTS[k], k) for k in (*fixed, *variates, *rand)}
    result = _new_result("glmm_analysis", frame, {"alpha": alpha, "transform_kind": "none",
                                                  "offset": 0.0},
                         distribution=distribution, link=link, column_names=names,
                         random_structure=random_structure, approximation=method)
    result.add_warning("glmm_experimental", EXPERIMENTAL, Severity.WARNING)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    ok = (_check_exposure(frame, distribution, result)
          and _check_response(frame, distribution, result)
          and _check_design(frame, fixed, variates, rand, result)
          and (not variates or _transform_column(frame, "x", covariate_transform,
                                                 covariate_offset, result)))
    if ok:
        spec = {"fixed": fixed, "variates": variates, "random": rand,
                "interactions": interactions == "yes", "nested": random_structure == "nested",
                "n_agq": _agq_nodes(method, n_agq, len(rand), result)}
        _analyse(frame, spec, names, result)
    return result


def _check_design(frame: pd.DataFrame, fixed: list[str], variates: list[str], rand: list[str],
                  result: StatResult) -> bool:
    """Уровни факторов, вариация ковариаты, ≥ 2 уровней случайных членов, размер задачи."""
    if any(frame[f].nunique() < 2 for f in fixed) or (variates and frame["x"].nunique() < 2):
        result.add_warning("factor_one_level", "У фактора один уровень или ковариата постоянна.",
                           Severity.ERROR)
        return False
    levels = [frame[k].nunique() for k in rand]
    if min(levels) < 2:
        result.add_warning("glmm_random_levels", "У случайного фактора должно быть ≥ 2 уровней.",
                           Severity.ERROR)
        return False
    if len(frame) > _MAX_N or sum(levels) > _MAX_Q:
        result.add_warning("glmm_too_large", f"Слишком большая задача для браузера: n ≤ {_MAX_N}, "
                           f"уровней случайных членов ≤ {_MAX_Q}.", Severity.ERROR,
                           params={"n": _MAX_N, "q": _MAX_Q})
        return False
    if min(levels) < _FEW_LEVELS:
        result.add_warning("glmm_few_levels", f"У случайного фактора < {_FEW_LEVELS} уровней: "
                           "дисперсия оценивается ненадёжно.", Severity.INFO,
                           params={"k": _FEW_LEVELS})
    return True


def _agq_nodes(method: str, n_agq: int, n_random: int, result: StatResult) -> int:
    """Число узлов квадратуры: 1 (Лаплас) или n_agq для одного случайного фактора."""
    if method == "laplace":
        return 1
    if n_random > 1:
        result.add_warning("glmm_agq_single_factor", "Квадратура — только для одного случайного "
                           "фактора; использована аппроксимация Лапласа.", Severity.WARNING)
        return 1
    result.parameters["n_agq"] = int(n_agq)
    return int(n_agq)


def _analyse(frame: pd.DataFrame, spec: dict, names: dict[str, str], result: StatResult) -> None:
    """Подгонка GLM (старт и нулевая модель), затем GLMM и отчёт."""
    p = result.parameters
    model = Model(frame, spec["fixed"], spec["variates"], _family(p["distribution"], p["link"]))
    terms = model_terms(spec["fixed"], spec["variates"], spec["interactions"])
    glm_fit = model.fit(terms)
    blocks = random_blocks(frame, spec["random"], spec["nested"])
    labels = [".".join(names[k] for k in spec["random"][:i + 1]) if spec["nested"] else names[r]
              for i, r in enumerate(spec["random"])]
    d = glmm_data(frame, glm_fit["x"], blocks, p["distribution"], p["link"])
    fit = fit_scaled(d, glm_fit["beta"], spec["n_agq"])
    if not fit["converged"]:
        result.add_warning("glmm_not_converged", "Оптимизация не сошлась: оценки ненадёжны.",
                           Severity.WARNING)
    reduced = variance_tests(d, fit, glm_fit["beta"], spec["n_agq"])
    p.update({"terms": [_label(t, names) for t in terms], "random_terms": labels,
              "variance_components": variance_rows(fit, blocks, reduced, labels),
              "glm_m2ll": null_m2ll(d, glm_fit["beta"])})
    if any(t < 1e-4 for t in fit["theta"]):
        result.add_warning("glmm_boundary", "Дисперсия случайного члена ≈ 0 (на границе): "
                           "модель сводится к GLM по этому члену.", Severity.INFO)
    _fixed_report(model, glm_fit, fit, names, result)
    _random_report(d, fit, blocks, labels, result)


def _fixed_report(model: Model, glm_fit: dict, fit: dict, names: dict[str, str],
                  result: StatResult) -> None:
    """Параметры, χ² Вальда членов, предсказания и сравнения уровней при b = 0."""
    alpha, k = result.parameters["alpha"], fit["theta"].size
    cov = fit["cov"][k:, k:]
    as_glm = {**glm_fit, "beta": fit["beta"], "cov1": cov}
    params = parameter_rows(model, as_glm, _FIXED_DISP, names, alpha)
    preds = factor_predictions(model, as_glm, _FIXED_DISP, names, alpha)
    result.parameters.update({
        "parameter_tests": params, "wald_terms": wald_terms(model, glm_fit, cov, fit["beta"], names),
        "factor_predictions": [{k: v for k, v in r.items() if k != "row"} for r in preds],
        "comparisons": comparisons(preds, as_glm, _FIXED_DISP, result.parameters["link"], alpha)})
    if preds:
        result.add_warning("glmm_predictions_b0", "Предсказания — для «типичной» группы (b = 0), "
                           "а не средние по популяции групп.", Severity.INFO)
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))


def _random_report(d: GLMMData, fit: dict, blocks: list[tuple], labels: list[str],
                   result: StatResult) -> None:
    """Условные моды, сверхдисперсия, графики и плоские статистики."""
    check = pearson_check(d, fit)
    binary = bool(np.all(d.m == 1)) and d.family == "binomial"
    result.parameters.update({"conditional_modes": conditional_modes(fit, blocks, labels),
                              "plots": residual_plots(check),
                              "overdispersion": {k: check[k] for k in ("chi2", "df", "ratio", "p")}})
    if not binary and check["p"] < result.parameters["alpha"]:
        result.add_warning("glmm_overdispersion", f"Сверхдисперсия: χ² Пирсона / df = "
                           f"{check['ratio']:.2f}; добавьте случайный фактор «наблюдение».",
                           Severity.WARNING, params={"ratio": round(check["ratio"], 2)})
    n_par = d.x.shape[1] + fit["theta"].size
    lr = max(result.parameters["glm_m2ll"] - fit["m2ll"], 0.0)
    result.p_value = boundary_p(lr, fit["theta"].size)
    result.statistics.update({
        "m2ll": fit["m2ll"], "loglik": -fit["m2ll"] / 2, "aic": fit["m2ll"] + 2 * n_par,
        "bic": fit["m2ll"] + np.log(d.y.size) * n_par, "lr_random": lr,
        "p_random": result.p_value, "pearson_ratio": check["ratio"],
        **{f"variance_{i + 1}": float(t ** 2) for i, t in enumerate(fit["theta"])}})
