"""Таблицы сопряжённости и порядковая регрессия (гл. 18.4 книги).

- ``contingency_test`` — таблица двух классификаций (по строкам наблюдений или с
  колонкой частот): χ² Пирсона (с поправкой Йейтса для 2 × 2 по выбору), G²
  отношения правдоподобия, точный тест Фишера для 2 × 2, V Крамера, ожидаемые
  частоты и скорректированные остатки. Книга (стр. 513): для двух классификаций —
  χ² Пирсона, для большего числа — лог-линейная пуассоновская GLM
  (glm_analysis); G² равен отклонению модели независимости.
- ``ordinal_regression`` — модель пропорциональных шансов для упорядоченных
  исходов (здоров < слабо < умеренно < сильно; стр. 513, Agresti, 2010):
  P(Y ≤ k) = F(θ_k − xβ), связь logit или probit; пороги, коэффициенты,
  LR-тесты модели и членов, вероятности категорий по уровням факторов и
  проверка пропорциональности по отдельным бинарным разрезам.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from core.report.schema import Estimate, Severity, StatResult
from core.stats.ancova import Terms, _frame, _label, _to_float
from core.stats.factorial import _natural_order, _new_result
from core.stats.glm import eligible, model_terms
from core.stats.mixed import _column_label, fixed_design
from core.stats.regression_checks import _transform_column

_CORRECTIONS = ("auto", "yes", "no")
_MIN_EXPECTED = 5.0
_ORD_LINKS = ("logit", "probit")


# --- contingency_test -----------------------------------------------------------


def _table(factor_a: list[str], factor_b: list[str], count: list | None) -> pd.DataFrame:
    """Таблица частот: строки — уровни A, столбцы — уровни B (пустые метки — пропуск)."""
    w = np.ones(len(factor_a)) if count is None else _to_float(count)
    frame = pd.DataFrame({"a": factor_a, "b": factor_b, "w": w})
    frame = frame[(frame["a"] != "") & (frame["b"] != "") & frame["w"].notna()]
    tab = frame.pivot_table(index="a", columns="b", values="w", aggfunc="sum", fill_value=0)
    return tab.loc[_natural_order(frame["a"]), _natural_order(frame["b"])]


def adjusted_residuals(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """(O − E)/√(E(1 − r/n)(1 − c/n)) — приближённо N(0, 1) при независимости."""
    n = obs.sum()
    r, c = obs.sum(axis=1, keepdims=True) / n, obs.sum(axis=0, keepdims=True) / n
    return (obs - exp) / np.sqrt(exp * (1 - r) * (1 - c))


def _cells(tab: pd.DataFrame, exp: np.ndarray) -> list[dict]:
    """Наблюдённые и ожидаемые частоты и скорректированные остатки по ячейкам."""
    obs = tab.to_numpy(float)
    adj = adjusted_residuals(obs, exp)
    return [{"row": str(r), "column": str(c), "observed": float(obs[i, j]),
             "expected": float(exp[i, j]), "adjusted_residual": float(adj[i, j])}
            for i, r in enumerate(tab.index) for j, c in enumerate(tab.columns)]


def _small_expected(exp: np.ndarray, result: StatResult) -> None:
    """Правило Кокрена: > 20 % ожидаемых < 5 или любая < 1 — χ² ненадёжен."""
    share = float((exp < _MIN_EXPECTED).mean())
    if share > 0.2 or exp.min() < 1:
        result.add_warning("contingency_small_expected", f"Ожидаемые частоты малы ({share:.0%} "
                           "ячеек < 5): χ² ненадёжен; для 2 × 2 — точный тест Фишера, иначе "
                           "объедини уровни.", params={"share": f"{share:.0%}"})


def contingency_test(
    factor_a: list[str],
    factor_b: list[str],
    count: list | None = None,
    correction: str = "auto",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """χ² независимости двух классификаций, G², Фишер для 2 × 2, V Крамера (стр. 513)."""
    if correction not in _CORRECTIONS or not 0 < alpha < 1:
        raise ValueError(f"correction ∈ {_CORRECTIONS}")
    tab = _table(factor_a, factor_b, count)
    cn = column_names or {}
    result = _new_result("contingency_test", tab, {"alpha": alpha, "transform_kind": "none",
                                                   "offset": 0.0},
                         rows=cn.get("factor_a", "A"), columns=cn.get("factor_b", "B"))
    if min(tab.shape) < 2 or tab.to_numpy().sum() == 0:
        result.add_warning("contingency_too_small", "Нужно ≥ 2 уровней в каждой классификации.",
                           Severity.ERROR)
        return result
    _fill_contingency(tab, correction, result)
    return result


def _fill_contingency(tab: pd.DataFrame, correction: str, result: StatResult) -> None:
    """Статистики, таблицы и предупреждения."""
    obs = tab.to_numpy(float)
    two_by_two = obs.shape == (2, 2)
    yates = correction == "yes" or (correction == "auto" and two_by_two)
    chi2, p, df, exp = stats.chi2_contingency(obs, correction=yates)
    g2, p_g2, _, _ = stats.chi2_contingency(obs, correction=False, lambda_="log-likelihood")
    chi2_raw = stats.chi2_contingency(obs, correction=False)[0]
    n = obs.sum()
    result.n = int(n)
    result.p_value = float(p)
    result.statistics.update({
        "chi2": float(chi2), "df": int(df), "p": float(p), "yates": bool(yates and two_by_two),
        "g2": float(g2), "p_g2": float(p_g2), "n_total": float(n),
        "cramers_v": float(np.sqrt(chi2_raw / (n * (min(obs.shape) - 1))))})
    if two_by_two:
        odds, p_f = stats.fisher_exact(obs)
        result.statistics.update({"fisher_p": float(p_f), "odds_ratio": float(odds)})
    result.parameters.update({"table": {"rows": [str(r) for r in tab.index],
                                        "columns": [str(c) for c in tab.columns],
                                        "counts": obs.tolist()},
                              "cells": _cells(tab, exp)})
    _small_expected(exp, result)


# --- ordinal_regression ---------------------------------------------------------


def response_order(response: list[str], levels: str) -> list[str]:
    """Порядок категорий: заданный списком через запятую или естественный."""
    given = [v.strip() for v in (levels or "").split(",") if v.strip()]
    present = sorted({r for r in response if r != ""})
    if not given:
        return _natural_order(pd.Series(present))
    unknown = [r for r in present if r not in given]
    if unknown:
        raise ValueError(f"Категории не из списка уровней: {', '.join(unknown)}")
    return [g for g in given if g in present]


def _ordinal_frame(response: list[str], slots: dict[str, list | None],
                   order: list[str]) -> tuple[pd.DataFrame, int]:
    """Полные строки: код категории y = 0 … K − 1, ковариата и факторы."""
    codes = [order.index(r) if r in order else np.nan for r in response]
    numeric = {"x": slots["x"]} if slots["x"] is not None else {}
    factors = {k: slots[k] for k in ("blk", "A", "B")}
    frame, dropped = _frame(codes, numeric, factors)
    frame["y"] = frame["y"].astype(int)
    return frame, dropped


def _ordered_fit(frame: pd.DataFrame, x: np.ndarray, link: str, k: int) -> dict:
    """ML-подгонка OrderedModel (без константы); пороги и их ковариация (дельта-метод)."""
    exog = x[:, 1:]
    if exog.shape[1] == 0:
        return _null_fit(frame, k)
    res = OrderedModel(frame["y"].to_numpy(), exog, distr=link).fit(
        method="bfgs", disp=False, maxiter=2000, gtol=1e-8)
    p = exog.shape[1]
    raw = np.asarray(res.params)
    jac = np.zeros((k - 1, raw.size))  # θ1 = c1, θj = c1 + Σ exp(c_i)
    jac[:, p] = 1.0
    for j in range(1, k - 1):
        jac[j:, p + j] = np.exp(raw[p + j])
    cov = np.asarray(res.cov_params())
    return {"beta": raw[:p], "cov_beta": cov[:p, :p], "theta": np.cumsum(
        np.r_[raw[p], np.exp(raw[p + 1:])]), "cov_theta": jac @ cov @ jac.T,
        "llf": float(res.llf), "converged": bool(res.mle_retvals.get("converged", True)),
        "n_par": raw.size}


def _null_fit(frame: pd.DataFrame, k: int) -> dict:
    """Только пороги: log L = Σ n_k log(n_k/N)."""
    n_k = np.bincount(frame["y"], minlength=k).astype(float)
    pos = n_k[n_k > 0]
    return {"llf": float((pos * np.log(pos / pos.sum())).sum()), "n_par": k - 1,
            "beta": np.zeros(0), "converged": True}


class Ordinal:
    """Члены модели и подгонки порядковой регрессии по набору членов (с кэшем)."""

    def __init__(self, frame: pd.DataFrame, factors: list[str], variates: list[str], link: str,
                 k: int) -> None:
        self.terms, self.frame, self.link, self.k = Terms(frame, factors, variates), frame, link, k
        self._cache: dict[tuple, dict] = {}

    def fit(self, terms: list[tuple[str, ...]]) -> dict:
        """Подгонка [пороги] + члены без вырожденных столбцов."""
        key = tuple(terms)
        if key not in self._cache:
            x, info = fixed_design(self.terms, terms)
            self._cache[key] = {**_ordered_fit(self.frame, x, self.link, self.k), "info": info,
                                "terms": list(terms), "x": x}
        return self._cache[key]


def _lr(full: dict, reduced: dict) -> dict:
    """LR = 2(ℓ_full − ℓ_reduced) против χ²(Δ параметров)."""
    df = full["n_par"] - reduced["n_par"]
    lr = 2 * (full["llf"] - reduced["llf"])
    return {"df": df, "lr": float(lr), "p": float(stats.chi2.sf(lr, df)) if df > 0 else 1.0}


def _coefficients(model: Ordinal, fit: dict, names: dict[str, str], alpha: float) -> list[dict]:
    """β с SE, z, P, ДИ; exp(β) — кумулятивное отношение шансов (logit)."""
    info, crit = fit["info"], float(stats.norm.ppf(1 - alpha / 2))
    rows = []
    for j, (b, v) in enumerate(zip(fit["beta"], np.diag(fit["cov_beta"])), start=1):
        se = float(np.sqrt(v))
        row = {"term": _column_label(model.terms, info["owner"][info["keep"][j]],
                                     info["keep"][j], info, names),
               "estimate": float(b), "se": se, "z": float(b / se),
               "p": float(2 * stats.norm.sf(abs(b / se))), "ci_lower": float(b - crit * se),
               "ci_upper": float(b + crit * se)}
        if model.link == "logit":
            row.update({"odds_ratio": float(np.exp(b)), "or_lower": float(np.exp(row["ci_lower"])),
                        "or_upper": float(np.exp(row["ci_upper"]))})
        rows.append(row)
    return rows


def _thresholds(fit: dict, order: list[str]) -> list[dict]:
    """Пороги θ_k между соседними категориями с SE."""
    se = np.sqrt(np.diag(fit["cov_theta"]))
    return [{"between": f"{order[k]} | {order[k + 1]}", "estimate": float(t), "se": float(s)}
            for k, (t, s) in enumerate(zip(fit["theta"], se))]


def _category_probs(model: Ordinal, fit: dict, names: dict[str, str],
                    order: list[str]) -> list[dict]:
    """Вероятности категорий для уровней факторов (усреднение η по прочим, ковариата — среднее)."""
    dist = stats.logistic if model.link == "logit" else stats.norm
    factors = [f for f in ("A", "B") if any(f in t for t in fit["terms"])]
    others = [f for f in ("blk", "A", "B") if any(f in t for t in fit["terms"])]
    out = []
    for f in factors:
        for lev in model.terms.levels[f]:
            eta = np.mean([_eta(model, fit, {**dict(zip(others, c)), f: lev})
                           for c in product(*(model.terms.levels[o] for o in others))
                           if dict(zip(others, c)).get(f, lev) == lev])
            cum = np.r_[dist.cdf(fit["theta"] - eta), 1.0]
            out.append({"factor": names[f], "level": lev, "eta": float(eta),
                        "probabilities": dict(zip(order, np.diff(np.r_[0.0, cum]).tolist()))})
    return out


def _eta(model: Ordinal, fit: dict, setting: dict) -> float:
    """xβ для уровней факторов (ковариата — среднее)."""
    one = pd.DataFrame({k: [v] for k, v in setting.items()})
    if "x" in model.terms.variates:
        one["x"] = float(model.frame["x"].mean())
    probe = Terms.__new__(Terms)
    probe.frame, probe.levels, probe.variates = one, model.terms.levels, model.terms.variates
    row = probe.design(fit["terms"])[0][fit["info"]["keep"]][1:]
    return float(row @ fit["beta"])


def _proportionality(model: Ordinal, fit: dict, names: dict[str, str]) -> list[dict]:
    """Отдельные бинарные модели P(Y > k): близкие наклоны поддерживают пропорциональность."""
    family = sm.families.Binomial(link=sm.families.links.Logit() if model.link == "logit"
                                  else sm.families.links.Probit())
    labels = [r["term"] for r in _coefficients(model, fit, names, 0.05)]
    out = []
    for k in range(model.k - 1):
        y = (model.frame["y"] > k).astype(float).to_numpy()
        if y.min() == y.max():
            continue
        res = sm.GLM(y, fit["x"], family=family).fit()
        out.append({"cut": k + 1, "slopes": dict(zip(labels, np.asarray(res.params)[1:].tolist()))})
    return out


def ordinal_regression(
    response: list[str],
    factor_a: list[str] | None = None,
    factor_b: list[str] | None = None,
    covariate: list | None = None,
    block: list[str] | None = None,
    levels: str = "",
    link: str = "logit",
    covariate_transform: str = "none",
    covariate_offset: float = 0.0,
    interactions: str = "yes",
    alpha: float = 0.05,
    column_names: dict[str, str] | None = None,
) -> StatResult:
    """Модель пропорциональных шансов для упорядоченного отклика (стр. 513)."""
    if link not in _ORD_LINKS or interactions not in ("yes", "no") or not 0 < alpha < 1:
        raise ValueError(f"link ∈ {_ORD_LINKS}, interactions ∈ (yes, no)")
    order = response_order(response, levels)
    slots = {"blk": block, "A": factor_a, "B": factor_b, "x": covariate}
    frame, dropped = _ordinal_frame(response, slots, order)
    factors = [k for k in ("blk", "A", "B") if slots[k] is not None]
    variates = ["x"] if covariate is not None else []
    cn = column_names or {}
    keys = {"blk": "block", "A": "factor_a", "B": "factor_b", "x": "covariate"}
    names = {k: cn.get(keys[k], k) for k in (*factors, *variates)}
    result = _new_result("ordinal_regression", frame, {"alpha": alpha, "transform_kind": "none",
                                                       "offset": 0.0},
                         categories=order, link=link, column_names=names)
    if dropped:
        result.add_warning("missing_dropped", f"Удалено строк с пропусками: {dropped}",
                           Severity.INFO)
    if _ordinal_guard(frame, order, factors, variates, result) and (
            not variates or _transform_column(frame, "x", covariate_transform, covariate_offset,
                                              result)):
        model = Ordinal(frame, factors, variates, link, len(order))
        _fill_ordinal(model, model_terms(factors, variates, interactions == "yes"), names, order,
                      result)
    return result


def _ordinal_guard(frame: pd.DataFrame, order: list[str], factors: list[str],
                   variates: list[str], result: StatResult) -> bool:
    """≥ 3 категорий (2 — это бинарная GLM), объясняющие члены, уровни факторов."""
    if len(order) < 3:
        result.add_warning("ordinal_two_categories", "Нужно ≥ 3 упорядоченных категорий; для двух "
                           "— биномиальная GLM (glm_analysis).", Severity.ERROR)
        return False
    if not [f for f in factors if f != "blk"] and not variates:
        result.add_warning("glm_no_terms", "Нужен фактор или ковариата.", Severity.ERROR)
        return False
    if any(frame[f].nunique() < 2 for f in factors):
        result.add_warning("factor_one_level", "У фактора один уровень.", Severity.ERROR)
        return False
    return True


def _fill_ordinal(model: Ordinal, terms: list[tuple[str, ...]], names: dict[str, str],
                  order: list[str], result: StatResult) -> None:
    """Подгонка, тесты, коэффициенты, пороги, вероятности и проверка пропорциональности."""
    alpha = result.parameters["alpha"]
    fit, null = model.fit(terms), model.fit([])
    if not fit["converged"]:
        result.add_warning("ordinal_not_converged", "Оптимизация не сошлась: возможно полное "
                           "разделение категорий.", Severity.WARNING)
    test = _lr(fit, null)
    result.p_value = test["p"]
    result.statistics.update({"lr_model": test["lr"], "df_model": test["df"], "p_model": test["p"],
                              "loglik": fit["llf"], "aic": 2 * fit["n_par"] - 2 * fit["llf"]})
    coefs = _coefficients(model, fit, names, alpha)
    result.parameters.update({
        "coefficients": coefs, "thresholds": _thresholds(fit, order),
        "term_tests": [{"term": "- " + _label(t, names),
                        **_lr(fit, model.fit([o for o in terms if o != t]))}
                       for t in eligible(terms)],
        "category_probabilities": _category_probs(model, fit, names, order),
        "proportionality": _proportionality(model, fit, names)})
    for row in coefs:
        result.estimates.append(Estimate(name=row["term"], value=row["estimate"],
                                         ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                                         ci_level=1 - alpha, std_error=row["se"]))
