"""Модели счётов сверх Пуассона (гл. 18.3.4 книги).

``count_regression`` — регрессия счётов методом максимального правдоподобия
(statsmodels): пуассоновская, отрицательная биномиальная NB2 (Var = μ + αμ²,
кластеризация), с нулевым раздуванием (ZIP, ZINB: доля «структурных» нулей π с
постоянной вероятностью). Объясняющие члены — как в glm_analysis (блок первым,
до двух факторов, ковариата), смещение log(экспозиция). Выход: параметры с SE
(z), α или π, LR-тесты модели и членов, AIC/BIC, сравнение с пуассоновской
моделью (LR для α = 0 на границе — P/2), наблюдённые и ожидаемые нули, средние
по уровням факторов и отношения средних.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import _label
from core.stats.factorial import _new_result
from core.stats.glm import (
    Model,
    _check_exposure,
    _check_response,
    _check_terms,
    _collect,
    _names,
    comparisons,
    eligible,
    factor_predictions,
    model_terms,
    parameter_rows,
)
from core.stats.mixed import fixed_design
from core.stats.regression_checks import _transform_column

COUNT_MODELS = ("negbin", "zip", "zinb", "poisson")
_FIXED = {"phi": 1.0, "estimated": False}


def _ml_fit(model_name: str, y: np.ndarray, x: np.ndarray, offset: np.ndarray | None,
            start: np.ndarray | None = None) -> dict:
    """ML-подгонка; разделяет параметры среднего, α и логит π (старт ZI — от базовой модели)."""
    kw = {"offset": offset} if offset is not None else {}
    p = x.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name == "poisson":
            res = sm.Poisson(y, x, **kw).fit(disp=False, maxiter=200)
        elif model_name == "negbin":
            res = sm.NegativeBinomial(y, x, loglike_method="nb2", **kw).fit(
                disp=False, maxiter=500, method="bfgs", gtol=1e-8)
        else:
            cls = ZeroInflatedPoisson if model_name == "zip" else ZeroInflatedNegativeBinomialP
            extra = {} if model_name == "zip" else {"p": 2}
            res = cls(y, x, exog_infl=np.ones((len(y), 1)), inflation="logit", **extra, **kw).fit(
                start_params=start, disp=False, maxiter=2000, method="bfgs", gtol=1e-8)
    params = np.asarray(res.params)
    try:
        cov = np.asarray(res.cov_params())
        ok = bool(np.all(np.isfinite(np.diag(cov))) and np.all(np.diag(cov) >= 0))
    except ValueError:
        cov, ok = np.full((params.size, params.size), np.nan), False
    infl = 1 if model_name in ("zip", "zinb") else 0
    mean = slice(infl, infl + p)
    return {"res": res, "beta": params[mean], "cov1": cov[mean, mean], "llf": float(res.llf),
            "n_par": params.size, "df": len(y) - params.size, "params": params, "cov": cov,
            "infl": infl, "p": p,
            "converged": ok and bool(res.mle_retvals.get("converged", True))}


class CountModel(Model):
    """План по членам (как в GLM) и ML-подгонка выбранной модели счётов."""

    def __init__(self, frame: pd.DataFrame, factors: list[str], variates: list[str],
                 model_name: str) -> None:
        super().__init__(frame, factors, variates, sm.families.Poisson())
        self.model_name = model_name

    def fit(self, terms: list[tuple[str, ...]], model_name: str | None = None) -> dict:
        """Подгонка [1] + члены (кэш по модели и набору членов)."""
        name = model_name or self.model_name
        key = (name, tuple(terms))
        if key not in self._cache:
            x, info = fixed_design(self.terms, terms)
            off = self.frame["off"].to_numpy(float) if "off" in self.frame else None
            start = None
            if name in ("zip", "zinb"):  # π ≈ 0.05 и параметры базовой модели
                base = self.fit(terms, "poisson" if name == "zip" else "negbin")
                start = np.r_[-3.0, base["params"]]
            fit = _ml_fit(name, self.frame["y"].to_numpy(float), x, off, start)
            self._cache[key] = {**fit, "info": info, "terms": list(terms), "x": x, "off": off}
        return self._cache[key]


def _lr(full: dict, reduced: dict) -> dict:
    """LR = 2(ℓ₁ − ℓ₀) против χ²(Δ параметров)."""
    df = full["n_par"] - reduced["n_par"]
    lr = max(2 * (full["llf"] - reduced["llf"]), 0.0)
    return {"df": df, "lr": float(lr), "p": float(stats.chi2.sf(lr, df)) if df > 0 else 1.0}


def _extra_parameters(fit: dict, model_name: str, alpha: float) -> dict:
    """α (NB, ключ nb_alpha) и π = logit⁻¹(γ) (ZI) с ДИ; ДИ π — обратным преобразованием."""
    out, crit = {}, float(stats.norm.ppf(1 - alpha / 2))
    params, cov = fit["params"], fit["cov"]
    if model_name in ("negbin", "zinb"):
        a, se = float(params[-1]), float(np.sqrt(cov[-1, -1]))
        out["nb_alpha"] = {"estimate": a, "se": se}
    if fit["infl"]:
        g, se = float(params[0]), float(np.sqrt(cov[0, 0]))
        expit = stats.logistic.cdf
        out["zero_inflation"] = {"logit": g, "se": se, "pi": float(expit(g)),
                                 "lower": float(expit(g - crit * se)),
                                 "upper": float(expit(g + crit * se))}
    return out


def expected_zeros(fit: dict, model_name: str) -> float:
    """Σ P(y = 0): Пуассон e^(−μ), NB2 (1 + αμ)^(−1/α), с раздуванием π + (1 − π)P₀."""
    eta = fit["x"] @ fit["beta"] + (0.0 if fit["off"] is None else fit["off"])
    mu = np.exp(eta)
    if model_name in ("negbin", "zinb"):
        a = fit["params"][-1]
        p0 = (1 + a * mu) ** (-1 / a)
    else:
        p0 = np.exp(-mu)
    if fit["infl"]:
        pi = stats.logistic.cdf(fit["params"][0])
        p0 = pi + (1 - pi) * p0
    return float(p0.sum())


# --- Метод каталога -------------------------------------------------------------


def count_regression(
    values: list[float],
    factor_a: list[str] | None = None,
    factor_b: list[str] | None = None,
    covariate: list | None = None,
    block: list[str] | None = None,
    exposure: list | None = None,
    model: str = "negbin",
    covariate_transform: str = "none",
    covariate_offset: float = 0.0,
    interactions: str = "yes",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Отрицательная биномиальная и модели с нулевым раздуванием для счётов (стр. 512)."""
    if model not in COUNT_MODELS or interactions not in ("yes", "no") or not 0 < alpha < 1:
        raise ValueError(f"model ∈ {COUNT_MODELS}, interactions ∈ (yes, no)")
    slots = {"blk": block, "A": factor_a, "B": factor_b, "x": covariate, "e": exposure}
    frame, dropped = _collect(values, slots, None, 0)
    factors = [k for k in ("blk", "A", "B") if slots[k] is not None]
    variates = ["x"] if covariate is not None else []
    names = _names(column_names, factors, variates)
    result = _new_result("count_regression", frame, {"alpha": alpha, "transform_kind": "none",
                                                     "offset": 0.0},
                         model=model, link="log", column_names=names)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    ok = (_check_exposure(frame, "poisson", result)
          and _check_response(frame, "poisson", result)
          and _check_terms(frame, factors, variates, result)
          and (not variates or _transform_column(frame, "x", covariate_transform,
                                                 covariate_offset, result)))
    if ok:
        cm = CountModel(frame, factors, variates, model)
        _fill(cm, model_terms(factors, variates, interactions == "yes"), names, result)
    return result


def _fill(cm: CountModel, terms: list[tuple[str, ...]], names: dict[str, str],
          result: StatResult) -> None:
    """Подгонка, тесты, параметры, сравнение с Пуассоном, нули и средние уровней."""
    alpha, name = result.parameters["alpha"], cm.model_name
    fit, null = cm.fit(terms), cm.fit([])
    if fit["infl"] and stats.logistic.cdf(fit["params"][0]) < 1e-4:
        result.add_warning("count_no_zero_inflation", "Доля структурных нулей π → 0: нулевого "
                           "раздувания нет, используй модель без него.")
    elif not fit["converged"]:
        result.add_warning("count_not_converged", "Оптимизация не сошлась: оценки ненадёжны "
                           "(мало ненулевых счётов или избыточная модель).", Severity.WARNING)
    _count_statistics(cm, fit, null, result)
    params = parameter_rows(cm, fit, _FIXED, names, alpha)
    preds = factor_predictions(cm, fit, _FIXED, names, alpha)
    result.parameters.update({
        "parameter_tests": params, **_extra_parameters(fit, name, alpha),
        "term_tests": [{"term": "- " + _label(t, names), **_lr(fit, cm.fit([o for o in terms
                                                                            if o != t]))}
                       for t in eligible(terms)],
        "factor_predictions": [{k: v for k, v in p.items() if k != "row"} for p in preds],
        "comparisons": comparisons(preds, fit, _FIXED, "log", alpha)})
    if name != "poisson":
        _versus_poisson(cm, terms, fit, name, result)
    for row in params:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))


def _count_statistics(cm: CountModel, fit: dict, null: dict, result: StatResult) -> None:
    """LR модели, log L, AIC, BIC, наблюдённые и ожидаемые нули."""
    test = _lr(fit, null)
    result.p_value = test["p"]
    n = len(cm.frame)
    result.statistics.update({
        "lr_model": test["lr"], "df_model": test["df"], "p_model": test["p"], "loglik": fit["llf"],
        "aic": 2 * fit["n_par"] - 2 * fit["llf"],
        "bic": float(np.log(n) * fit["n_par"] - 2 * fit["llf"]),
        "observed_zeros": int((cm.frame["y"] == 0).sum()),
        "expected_zeros": expected_zeros(fit, cm.model_name)})


def _versus_poisson(cm: CountModel, terms: list[tuple[str, ...]], fit: dict, name: str,
                    result: StatResult) -> None:
    """LR против пуассоновской модели; α = 0 и π = 0 — на границе, P делится пополам."""
    pois = cm.fit(terms, "poisson")
    test = _lr(fit, pois)
    p_half = test["p"] / 2
    result.statistics.update({"lr_vs_poisson": test["lr"], "df_vs_poisson": test["df"],
                              "p_vs_poisson": p_half, "aic_poisson": 2 * pois["n_par"]
                              - 2 * pois["llf"]})
    if p_half >= result.parameters["alpha"]:
        result.add_warning("count_poisson_sufficient", "Пуассоновская модель не хуже "
                           f"(LR = {test['lr']:.2f}, P = {p_half:.3f}): лишний параметр не нужен.",
                           params={"lr": round(test["lr"], 2), "p": round(p_half, 4)})
